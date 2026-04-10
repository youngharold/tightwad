"""Tests for SSRF protection (tightwad.ssrf + config integration).

Covers:
  - Scheme validation (only http/https allowed)
  - Private/internal IPv4 address blocking
  - Private/internal IPv6 address blocking
  - Localhost / loopback blocking
  - Link-local / IMDS blocking (169.254.0.0/16)
  - DNS-rebinding protection (resolved IP checked against blocklist)
  - allow_private=True bypass for homelab LAN deployments
  - Public URLs pass validation
  - Config integration: bad URLs rejected at load_config() time
  - Environment variable integration: bad URLs rejected via load_proxy_from_env()
  - Edge cases: missing scheme, missing hostname, empty string

Audit reference: SEC-5 / GitHub issue #7
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tightwad.ssrf import (
    ALLOWED_SCHEMES,
    BLOCKED_SCHEMES,
    PRIVATE_NETWORKS,
    _is_private_ip,
    _resolve_host,
    validate_upstream_url,
)
from tightwad.config import load_config, load_proxy_from_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cluster_yaml(tmp_path: Path, proxy_dict: dict) -> Path:
    """Write a minimal cluster.yaml with the given proxy block and return its path."""
    cfg = {
        "coordinator": {
            "host": "0.0.0.0",
            "port": 8080,
            "backend": "hip",
            "gpus": [{"name": "XTX", "vram_gb": 24}],
        },
        "models": {"m": {"path": "/m.gguf", "default": True}},
        "proxy": proxy_dict,
    }
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.dump(cfg))
    return p


# ---------------------------------------------------------------------------
# 1.  Module-level sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    def test_http_in_allowed(self):
        assert "http" in ALLOWED_SCHEMES

    def test_https_in_allowed(self):
        assert "https" in ALLOWED_SCHEMES

    def test_file_in_blocked(self):
        assert "file" in BLOCKED_SCHEMES

    def test_gopher_in_blocked(self):
        assert "gopher" in BLOCKED_SCHEMES

    def test_private_networks_not_empty(self):
        assert len(PRIVATE_NETWORKS) > 0

    def test_loopback_in_private_networks(self):
        import ipaddress
        loopback = ipaddress.ip_network("127.0.0.0/8")
        assert loopback in PRIVATE_NETWORKS

    def test_rfc1918_class_a_in_private_networks(self):
        import ipaddress
        assert ipaddress.ip_network("10.0.0.0/8") in PRIVATE_NETWORKS

    def test_rfc1918_class_b_in_private_networks(self):
        import ipaddress
        assert ipaddress.ip_network("172.16.0.0/12") in PRIVATE_NETWORKS

    def test_rfc1918_class_c_in_private_networks(self):
        import ipaddress
        assert ipaddress.ip_network("192.168.0.0/16") in PRIVATE_NETWORKS

    def test_link_local_in_private_networks(self):
        import ipaddress
        assert ipaddress.ip_network("169.254.0.0/16") in PRIVATE_NETWORKS

    def test_ipv6_loopback_in_private_networks(self):
        import ipaddress
        assert ipaddress.ip_network("::1/128") in PRIVATE_NETWORKS

    def test_ipv6_ula_in_private_networks(self):
        import ipaddress
        assert ipaddress.ip_network("fc00::/7") in PRIVATE_NETWORKS


# ---------------------------------------------------------------------------
# 2.  _is_private_ip() unit tests
# ---------------------------------------------------------------------------

class TestIsPrivateIp:
    @pytest.mark.parametrize("addr", [
        "127.0.0.1",
        "127.255.255.255",
        "10.0.0.1",
        "10.255.255.255",
        "172.16.0.1",
        "172.31.255.255",
        "192.168.0.1",
        "192.168.255.255",
        "169.254.1.1",
        "169.254.169.254",    # AWS/GCP IMDS
        "::1",                 # IPv6 loopback
        "fc00::1",             # IPv6 ULA
        "fd12:3456:789a::1",   # IPv6 ULA (fd range)
        "fe80::1",             # IPv6 link-local
        "0.0.0.1",             # "this" network
        "255.255.255.255",     # broadcast
    ])
    def test_private_ip_detected(self, addr):
        assert _is_private_ip(addr) is True

    @pytest.mark.parametrize("addr", [
        "8.8.8.8",
        "1.1.1.1",
        "93.184.216.34",      # example.com
        "2001:4860:4860::8888",  # Google Public DNS IPv6
    ])
    def test_public_ip_not_private(self, addr):
        assert _is_private_ip(addr) is False

    def test_unparseable_address_returns_true(self):
        # Unparseable addresses should be treated conservatively as private.
        assert _is_private_ip("not-an-ip-address") is True


# ---------------------------------------------------------------------------
# 3.  Scheme validation
# ---------------------------------------------------------------------------

class TestSchemeValidation:
    @pytest.mark.parametrize("url", [
        "http://8.8.8.8:80",
        "https://8.8.8.8:443",
        "HTTP://8.8.8.8:80",   # case-insensitive
        "HTTPS://8.8.8.8:443",
    ])
    def test_valid_schemes_pass(self, url):
        # resolve_hostname=False to skip DNS lookup for IP literals that
        # happen to be public and don't need DNS.
        validate_upstream_url(url, allow_private=False, resolve_hostname=False)

    @pytest.mark.parametrize("bad_scheme_url", [
        "file:///etc/passwd",
        "file://localhost/etc/shadow",
        "gopher://evil.com/",
        "ftp://evil.com/",
        "ftps://evil.com/",
        "sftp://evil.com/",
        "ldap://internal/",
        "ldaps://internal/",
        "dict://localhost/",
        "telnet://127.0.0.1:23/",
        "smtp://mail.internal/",
        "data:text/plain;base64,SGVsbG8=",
        "javascript:alert(1)",
    ])
    def test_dangerous_schemes_rejected(self, bad_scheme_url):
        with pytest.raises(ValueError, match="scheme"):
            validate_upstream_url(bad_scheme_url, allow_private=False, resolve_hostname=False)

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_upstream_url("xyz://example.com/", resolve_hostname=False)

    def test_missing_scheme_rejected(self):
        with pytest.raises(ValueError):
            validate_upstream_url("192.168.1.10:11434", resolve_hostname=False)

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            validate_upstream_url("", resolve_hostname=False)

    def test_non_string_rejected(self):
        with pytest.raises((ValueError, AttributeError)):
            validate_upstream_url(None, resolve_hostname=False)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4.  Private / internal IP blocking (literal IPs)
# ---------------------------------------------------------------------------

class TestPrivateIpBlocking:
    """validate_upstream_url with allow_private=False should block private IPs."""

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:11434",
        "http://127.0.0.2:8080",
        "http://10.0.0.1:8080",
        "http://10.255.255.254:8080",
        "http://172.16.0.1:8080",
        "http://172.31.255.255:8080",
        "http://192.168.0.1:8080",
        "http://192.168.255.254:8080",
        "http://169.254.169.254/latest/meta-data",  # AWS IMDS
        "http://169.254.0.1/",
    ])
    def test_private_ipv4_blocked(self, url):
        with pytest.raises(ValueError, match="private"):
            validate_upstream_url(url, allow_private=False, resolve_hostname=False)

    @pytest.mark.parametrize("url", [
        "http://[::1]:8080/",
        "http://[fc00::1]:8080/",
        "http://[fd12:3456:789a::1]:8080/",
        "http://[fe80::1]:8080/",
    ])
    def test_private_ipv6_blocked(self, url):
        with pytest.raises(ValueError, match="private"):
            validate_upstream_url(url, allow_private=False, resolve_hostname=False)

    @pytest.mark.parametrize("url", [
        "http://8.8.8.8:80",
        "http://1.1.1.1:443",
        "https://93.184.216.34:443",
    ])
    def test_public_ipv4_allowed(self, url):
        # Public IPs should not raise (resolve_hostname=False — they're literals).
        validate_upstream_url(url, allow_private=False, resolve_hostname=False)


# ---------------------------------------------------------------------------
# 5.  allow_private=True bypass (homelab mode)
# ---------------------------------------------------------------------------

class TestAllowPrivateBypass:
    """When allow_private=True, private IPs and LAN addresses must be allowed."""

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:11434",
        "http://192.168.1.10:11434",
        "http://10.0.0.5:8080",
        "http://172.16.1.1:8080",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]:8080/",
        "http://[fc00::1]:8080/",
    ])
    def test_private_ip_allowed_when_flag_set(self, url):
        # Must not raise.
        validate_upstream_url(url, allow_private=True, resolve_hostname=False)

    def test_bad_scheme_still_rejected_even_with_allow_private(self):
        """Scheme check is always enforced regardless of allow_private."""
        with pytest.raises(ValueError, match="scheme"):
            validate_upstream_url("file:///etc/passwd", allow_private=True)

    def test_gopher_still_rejected_with_allow_private(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_upstream_url("gopher://127.0.0.1/", allow_private=True)


# ---------------------------------------------------------------------------
# 6.  DNS rebinding protection
# ---------------------------------------------------------------------------

class TestDnsRebindingProtection:
    """Hostname-based URLs must have their resolved IPs checked."""

    def test_hostname_resolving_to_private_ip_blocked(self):
        """If DNS resolves to a private IP, the URL should be blocked."""
        with patch("tightwad.ssrf._resolve_host", return_value=["127.0.0.1"]):
            with pytest.raises(ValueError, match="private"):
                validate_upstream_url(
                    "http://evil.internal.lan/",
                    allow_private=False,
                    resolve_hostname=True,
                )

    def test_hostname_resolving_to_imds_blocked(self):
        """A domain that resolves to 169.254.169.254 (IMDS) is blocked."""
        with patch("tightwad.ssrf._resolve_host", return_value=["169.254.169.254"]):
            with pytest.raises(ValueError, match="private"):
                validate_upstream_url(
                    "http://metadata.google.internal/",
                    allow_private=False,
                    resolve_hostname=True,
                )

    def test_hostname_resolving_to_rfc1918_blocked(self):
        """Domain resolving to 192.168.x.x is blocked."""
        with patch("tightwad.ssrf._resolve_host", return_value=["192.168.0.1"]):
            with pytest.raises(ValueError, match="private"):
                validate_upstream_url(
                    "http://not-so-public.example.com/",
                    allow_private=False,
                    resolve_hostname=True,
                )

    def test_hostname_resolving_to_public_ip_allowed(self):
        """Domain resolving to a public IP should pass."""
        with patch("tightwad.ssrf._resolve_host", return_value=["93.184.216.34"]):
            # Should not raise.
            validate_upstream_url(
                "http://example.com/",
                allow_private=False,
                resolve_hostname=True,
            )

    def test_hostname_resolving_to_any_private_ip_blocked(self):
        """If ANY resolved address is private, the URL is blocked (multi-A record)."""
        with patch("tightwad.ssrf._resolve_host", return_value=["8.8.8.8", "10.0.0.1"]):
            with pytest.raises(ValueError, match="private"):
                validate_upstream_url(
                    "http://tricky.example.com/",
                    allow_private=False,
                    resolve_hostname=True,
                )

    def test_dns_failure_raises_value_error(self):
        """Unresolvable hostname should raise ValueError (fail-closed)."""
        import socket
        with patch("tightwad.ssrf._resolve_host", side_effect=socket.gaierror("NXDOMAIN")):
            with pytest.raises(ValueError, match="could not be resolved"):
                validate_upstream_url(
                    "http://this-does-not-exist.invalid/",
                    allow_private=False,
                    resolve_hostname=True,
                )

    def test_resolve_hostname_false_skips_dns(self):
        """resolve_hostname=False must skip the DNS lookup entirely."""
        # If DNS were called, _resolve_host would raise — but it should not be called.
        with patch("tightwad.ssrf._resolve_host", side_effect=RuntimeError("should not call")):
            # Public IP, allow_private=False — should just pass without DNS.
            validate_upstream_url(
                "http://8.8.8.8:80",
                allow_private=False,
                resolve_hostname=False,
            )

    def test_allow_private_skips_dns_resolution(self):
        """When allow_private=True, DNS resolution should also be skipped."""
        with patch("tightwad.ssrf._resolve_host", side_effect=RuntimeError("should not call")):
            validate_upstream_url(
                "http://lan-server.home/",
                allow_private=True,
                resolve_hostname=True,  # True but should be bypassed by allow_private
            )


# ---------------------------------------------------------------------------
# 7.  Localhost / loopback-specific scenarios
# ---------------------------------------------------------------------------

class TestLocalhostBlocking:
    @pytest.mark.parametrize("url", [
        "http://localhost:11434",
        "http://localhost:8080",
        "http://localhost/",
    ])
    def test_localhost_hostname_blocked_via_dns(self, url):
        """'localhost' resolves to 127.0.0.1 (or ::1) and should be blocked."""
        with patch("tightwad.ssrf._resolve_host", return_value=["127.0.0.1"]):
            with pytest.raises(ValueError, match="private"):
                validate_upstream_url(url, allow_private=False, resolve_hostname=True)

    def test_localhost_allowed_with_allow_private(self):
        """localhost should be reachable when allow_private=True."""
        with patch("tightwad.ssrf._resolve_host", return_value=["127.0.0.1"]):
            validate_upstream_url(
                "http://localhost:11434",
                allow_private=True,
                resolve_hostname=True,
            )

    def test_loopback_literal_blocked(self):
        validate_upstream_url.__module__  # ensure imported
        with pytest.raises(ValueError, match="private"):
            validate_upstream_url("http://127.0.0.1:11434", allow_private=False, resolve_hostname=False)


# ---------------------------------------------------------------------------
# 8.  URL edge cases
# ---------------------------------------------------------------------------

class TestUrlEdgeCases:
    def test_no_hostname_rejected(self):
        with pytest.raises(ValueError, match="hostname"):
            validate_upstream_url("http:///path", resolve_hostname=False)

    def test_url_with_path_and_port_validated(self):
        """Extra path/port components should not interfere with validation."""
        validate_upstream_url(
            "https://8.8.8.8:443/api/v1/endpoint",
            allow_private=False,
            resolve_hostname=False,
        )

    def test_url_with_credentials_validated(self):
        """URLs with embedded user:pass should still be checked on the hostname."""
        with pytest.raises(ValueError, match="private"):
            validate_upstream_url(
                "http://user:pass@192.168.1.10:8080/",
                allow_private=False,
                resolve_hostname=False,
            )

    def test_uppercase_scheme_accepted(self):
        validate_upstream_url("HTTP://8.8.8.8:80", allow_private=False, resolve_hostname=False)


# ---------------------------------------------------------------------------
# 9.  Config integration — load_config() rejects bad upstream URLs
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_bad_scheme_in_draft_url_rejected(self, tmp_path):
        p = _make_cluster_yaml(tmp_path, {
            "draft": {"url": "file:///etc/passwd", "model_name": "draft"},
            "target": {"url": "http://8.8.8.8:8080", "model_name": "target"},
        })
        with pytest.raises(ValueError, match="scheme"):
            load_config(p)

    def test_bad_scheme_in_target_url_rejected(self, tmp_path):
        p = _make_cluster_yaml(tmp_path, {
            "draft": {"url": "http://8.8.8.8:8081", "model_name": "draft"},
            "target": {"url": "gopher://evil.com/", "model_name": "target"},
        })
        with pytest.raises(ValueError, match="scheme"):
            load_config(p)

    def test_private_ip_in_draft_blocked_when_not_allowed(self, tmp_path):
        p = _make_cluster_yaml(tmp_path, {
            "allow_private_upstream": False,
            "draft": {"url": "http://127.0.0.1:8081", "model_name": "draft"},
            "target": {"url": "http://8.8.8.8:8080", "model_name": "target"},
        })
        with pytest.raises(ValueError, match="private"):
            load_config(p)

    def test_private_ip_in_target_blocked_when_not_allowed(self, tmp_path):
        p = _make_cluster_yaml(tmp_path, {
            "allow_private_upstream": False,
            "draft": {"url": "http://8.8.8.8:8081", "model_name": "draft"},
            "target": {"url": "http://192.168.1.10:8080", "model_name": "target"},
        })
        with pytest.raises(ValueError, match="private"):
            load_config(p)

    def test_private_ip_allowed_in_lan_mode(self, tmp_path):
        """Default homelab mode (allow_private_upstream: true) accepts LAN URLs."""
        p = _make_cluster_yaml(tmp_path, {
            "allow_private_upstream": True,
            "draft": {"url": "http://192.168.1.101:11434", "model_name": "draft"},
            "target": {"url": "http://192.168.1.100:8080", "model_name": "target"},
        })
        config = load_config(p)
        assert config.proxy is not None
        assert config.proxy.draft.url == "http://192.168.1.101:11434"
        assert config.proxy.allow_private_upstream is True

    def test_default_allow_private_upstream_is_true(self, tmp_path):
        """When allow_private_upstream is omitted from YAML it defaults to True."""
        p = _make_cluster_yaml(tmp_path, {
            "draft": {"url": "http://192.168.1.101:11434", "model_name": "draft"},
            "target": {"url": "http://192.168.1.100:8080", "model_name": "target"},
        })
        config = load_config(p)
        assert config.proxy is not None
        assert config.proxy.allow_private_upstream is True

    def test_no_proxy_section_still_loads(self, tmp_path):
        """Configs without a proxy block continue to work unaffected."""
        cfg = {
            "coordinator": {
                "host": "0.0.0.0", "port": 8080, "backend": "hip",
                "gpus": [{"name": "XTX", "vram_gb": 24}],
            },
            "models": {"m": {"path": "/m.gguf", "default": True}},
        }
        p = tmp_path / "cluster.yaml"
        p.write_text(yaml.dump(cfg))
        config = load_config(p)
        assert config.proxy is None


# ---------------------------------------------------------------------------
# 10.  Environment variable integration — load_proxy_from_env()
# ---------------------------------------------------------------------------

class TestEnvVarIntegration:
    def test_bad_draft_scheme_rejected(self, monkeypatch):
        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "file:///etc/passwd")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://8.8.8.8:8080")
        monkeypatch.setenv("TIGHTWAD_ALLOW_PRIVATE_UPSTREAM", "false")
        with pytest.raises(ValueError, match="scheme"):
            load_proxy_from_env()

    def test_bad_target_scheme_rejected(self, monkeypatch):
        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://8.8.8.8:8081")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "gopher://evil.com/")
        monkeypatch.setenv("TIGHTWAD_ALLOW_PRIVATE_UPSTREAM", "false")
        with pytest.raises(ValueError, match="scheme"):
            load_proxy_from_env()

    def test_private_ip_blocked_when_disabled(self, monkeypatch):
        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://127.0.0.1:8081")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://8.8.8.8:8080")
        monkeypatch.setenv("TIGHTWAD_ALLOW_PRIVATE_UPSTREAM", "false")
        with pytest.raises(ValueError, match="private"):
            load_proxy_from_env()

    def test_private_ip_allowed_when_enabled(self, monkeypatch):
        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://192.168.1.101:11434")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://192.168.1.100:8080")
        monkeypatch.setenv("TIGHTWAD_ALLOW_PRIVATE_UPSTREAM", "true")
        result = load_proxy_from_env()
        assert result is not None
        assert result.allow_private_upstream is True

    def test_allow_private_defaults_to_true(self, monkeypatch):
        """When TIGHTWAD_ALLOW_PRIVATE_UPSTREAM is unset it should default True."""
        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://192.168.1.101:11434")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://192.168.1.100:8080")
        monkeypatch.delenv("TIGHTWAD_ALLOW_PRIVATE_UPSTREAM", raising=False)
        result = load_proxy_from_env()
        assert result is not None
        assert result.allow_private_upstream is True

    def test_missing_draft_url_returns_none(self, monkeypatch):
        monkeypatch.delenv("TIGHTWAD_DRAFT_URL", raising=False)
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://8.8.8.8:8080")
        assert load_proxy_from_env() is None

    def test_missing_target_url_returns_none(self, monkeypatch):
        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://8.8.8.8:8081")
        monkeypatch.delenv("TIGHTWAD_TARGET_URL", raising=False)
        assert load_proxy_from_env() is None

    @pytest.mark.parametrize("false_val", ["false", "False", "FALSE", "0", "no", "No"])
    def test_allow_private_false_variants(self, monkeypatch, false_val):
        """Various "falsy" string values should all disable private upstream."""
        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://127.0.0.1:8081")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://8.8.8.8:8080")
        monkeypatch.setenv("TIGHTWAD_ALLOW_PRIVATE_UPSTREAM", false_val)
        with pytest.raises(ValueError, match="private"):
            load_proxy_from_env()


# ---------------------------------------------------------------------------
# 11.  Dangerous scheme error message quality
# ---------------------------------------------------------------------------

class TestErrorMessages:
    def test_file_scheme_error_mentions_http_https(self):
        with pytest.raises(ValueError) as exc_info:
            validate_upstream_url("file:///etc/passwd", resolve_hostname=False)
        assert "http" in str(exc_info.value).lower()

    def test_private_ip_error_mentions_allow_private(self):
        with pytest.raises(ValueError) as exc_info:
            validate_upstream_url("http://127.0.0.1/", allow_private=False, resolve_hostname=False)
        assert "allow_private" in str(exc_info.value)

    def test_dns_rebinding_error_mentions_ssrf(self):
        with patch("tightwad.ssrf._resolve_host", return_value=["10.0.0.1"]):
            with pytest.raises(ValueError) as exc_info:
                validate_upstream_url(
                    "http://evil.rebind.net/",
                    allow_private=False,
                    resolve_hostname=True,
                )
        err = str(exc_info.value)
        # Should mention the hostname AND the resolved address
        assert "evil.rebind.net" in err
        assert "10.0.0.1" in err
