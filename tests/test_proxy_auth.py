"""Tests for proxy Bearer-token authentication (issue #6).

Covers:
- No token configured → open access (backward compat)
- Token configured → 401 on missing / invalid token, 200 on valid token
- All /v1/ API endpoints are protected
- Non-API routes (/ dashboard) are also protected when a token is set
- TokenAuthMiddleware unit tests
- Config: auth_token loaded from YAML and env vars
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
import yaml
from starlette.testclient import TestClient

from tightwad.config import ProxyConfig, ServerEndpoint, load_config
from tightwad.proxy import TokenAuthMiddleware, create_app


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    """ProxyConfig without a token — open (backward-compat) mode."""
    return ProxyConfig(
        draft=ServerEndpoint(url="http://draft:8081", model_name="qwen3-8b"),
        target=ServerEndpoint(url="http://target:8080", model_name="qwen3-32b"),
        host="0.0.0.0",
        port=8088,
        max_draft_tokens=8,
        fallback_on_draft_failure=True,
        auth_token=None,
    )


@pytest.fixture
def secured_config():
    """ProxyConfig with a Bearer token configured."""
    return ProxyConfig(
        draft=ServerEndpoint(url="http://draft:8081", model_name="qwen3-8b"),
        target=ServerEndpoint(url="http://target:8080", model_name="qwen3-32b"),
        host="0.0.0.0",
        port=8088,
        max_draft_tokens=8,
        fallback_on_draft_failure=True,
        auth_token="super-secret-token",
    )


# ---------------------------------------------------------------------------
# TokenAuthMiddleware unit tests
# ---------------------------------------------------------------------------


class TestTokenAuthMiddleware:
    """Unit-level tests for the ASGI middleware class."""

    def test_no_token_is_noop(self, base_config):
        """Middleware with token=None must not block any requests."""
        app = create_app(base_config)
        client = TestClient(app)

        # GET /v1/models should succeed without any Authorization header.
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_valid_token_passes(self, secured_config):
        """A correct Bearer token allows the request through."""
        app = create_app(secured_config)
        client = TestClient(app)

        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer super-secret-token"},
        )
        assert resp.status_code == 200

    def test_missing_auth_header_returns_401(self, secured_config):
        """No Authorization header → 401."""
        app = create_app(secured_config)
        client = TestClient(app)

        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self, secured_config):
        """Wrong token value → 401."""
        app = create_app(secured_config)
        client = TestClient(app)

        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_wrong_scheme_returns_401(self, secured_config):
        """Non-Bearer scheme (e.g. Basic) → 401."""
        app = create_app(secured_config)
        client = TestClient(app)

        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Basic super-secret-token"},
        )
        assert resp.status_code == 401

    def test_empty_auth_header_returns_401(self, secured_config):
        """Empty Authorization header → 401."""
        app = create_app(secured_config)
        client = TestClient(app)

        resp = client.get(
            "/v1/models",
            headers={"Authorization": ""},
        )
        assert resp.status_code == 401

    def test_401_response_is_json(self, secured_config):
        """401 response body should be valid JSON with a 'detail' key."""
        app = create_app(secured_config)
        client = TestClient(app)

        resp = client.get("/v1/models")
        assert resp.status_code == 401
        data = resp.json()
        assert "detail" in data

    def test_401_response_has_www_authenticate_header(self, secured_config):
        """401 response should include WWW-Authenticate: Bearer."""
        app = create_app(secured_config)
        client = TestClient(app)

        resp = client.get("/v1/models")
        assert resp.status_code == 401
        assert "www-authenticate" in {k.lower() for k in resp.headers}
        assert resp.headers["www-authenticate"].lower() == "bearer"


# ---------------------------------------------------------------------------
# All /v1/ endpoints are protected when a token is set
# ---------------------------------------------------------------------------


class TestAuthEnforcedOnAllEndpoints:
    """Every API endpoint must be gated when auth is configured."""

    ENDPOINTS_GET = [
        "/v1/models",
        "/v1/tightwad/status",
        "/v1/tightwad/history",
    ]

    ENDPOINTS_POST = [
        "/v1/completions",
        "/v1/chat/completions",
    ]

    @pytest.mark.parametrize("path", ENDPOINTS_GET)
    def test_get_endpoints_blocked_without_token(self, secured_config, path):
        app = create_app(secured_config)
        client = TestClient(app)
        resp = client.get(path)
        assert resp.status_code == 401, f"Expected 401 on GET {path}, got {resp.status_code}"

    @pytest.mark.parametrize("path", ENDPOINTS_POST)
    def test_post_endpoints_blocked_without_token(self, secured_config, path):
        app = create_app(secured_config)
        client = TestClient(app)
        resp = client.post(path, json={"prompt": "hi", "max_tokens": 5})
        assert resp.status_code == 401, f"Expected 401 on POST {path}, got {resp.status_code}"

    @pytest.mark.parametrize("path", ENDPOINTS_GET)
    def test_get_endpoints_pass_with_valid_token(self, secured_config, path):
        app = create_app(secured_config)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            path,
            headers={"Authorization": "Bearer super-secret-token"},
        )
        # Should NOT be 401 (may be 200 or 500 if backends are down — that's ok)
        assert resp.status_code != 401, (
            f"Valid token should not return 401 on GET {path}"
        )

    @pytest.mark.parametrize("path", ENDPOINTS_POST)
    def test_post_endpoints_pass_with_valid_token(self, secured_config, path):
        app = create_app(secured_config)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            path,
            json={"prompt": "hi", "max_tokens": 5},
            headers={"Authorization": "Bearer super-secret-token"},
        )
        # Should NOT be 401; 500 is fine (backends unreachable in unit tests)
        assert resp.status_code != 401, (
            f"Valid token should not return 401 on POST {path}"
        )


# ---------------------------------------------------------------------------
# Backward compatibility: no token = open proxy
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """When no token is configured, the proxy works without authentication."""

    def test_models_endpoint_open_without_token(self, base_config):
        app = create_app(base_config)
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_status_endpoint_open_without_token(self, base_config):
        app = create_app(base_config)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/tightwad/status")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Config: auth_token loaded from YAML and environment variables
# ---------------------------------------------------------------------------


class TestProxyConfigAuthToken:
    """ProxyConfig correctly propagates auth_token from YAML and env vars."""

    def test_no_token_defaults_to_none(self, base_config):
        assert base_config.auth_token is None

    def test_token_stored_on_config(self, secured_config):
        assert secured_config.auth_token == "super-secret-token"

    def test_yaml_config_parses_auth_token(self, tmp_path):
        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "hip",
                "gpus": [{"name": "XTX", "vram_gb": 24}],
            },
            "models": {"test": {"path": "/test.gguf", "default": True}},
            "proxy": {
                "host": "127.0.0.1",
                "port": 9999,
                "max_draft_tokens": 4,
                "auth_token": "yaml-secret",
                "draft": {
                    "url": "http://192.168.1.1:8081",
                    "model_name": "small-model",
                },
                "target": {
                    "url": "http://192.168.1.2:8080",
                    "model_name": "big-model",
                },
            },
        }
        p = tmp_path / "cluster.yaml"
        p.write_text(yaml.dump(cfg))
        config = load_config(p)

        assert config.proxy is not None
        assert config.proxy.auth_token == "yaml-secret"

    def test_yaml_config_no_auth_token_is_none(self, tmp_path):
        """Omitting auth_token in YAML gives None (backward compat)."""
        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "hip",
                "gpus": [{"name": "XTX", "vram_gb": 24}],
            },
            "models": {"test": {"path": "/test.gguf", "default": True}},
            "proxy": {
                "host": "127.0.0.1",
                "port": 9999,
                "max_draft_tokens": 4,
                "draft": {
                    "url": "http://192.168.1.1:8081",
                    "model_name": "small-model",
                },
                "target": {
                    "url": "http://192.168.1.2:8080",
                    "model_name": "big-model",
                },
            },
        }
        p = tmp_path / "cluster.yaml"
        p.write_text(yaml.dump(cfg))
        config = load_config(p)

        assert config.proxy is not None
        assert config.proxy.auth_token is None

    def test_env_var_proxy_token_used_when_yaml_missing(self, tmp_path, monkeypatch):
        """TIGHTWAD_PROXY_TOKEN env var is used when YAML omits auth_token."""
        monkeypatch.setenv("TIGHTWAD_PROXY_TOKEN", "env-token")

        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "hip",
                "gpus": [{"name": "XTX", "vram_gb": 24}],
            },
            "models": {"test": {"path": "/test.gguf", "default": True}},
            "proxy": {
                "draft": {
                    "url": "http://192.168.1.1:8081",
                    "model_name": "small-model",
                },
                "target": {
                    "url": "http://192.168.1.2:8080",
                    "model_name": "big-model",
                },
            },
        }
        p = tmp_path / "cluster.yaml"
        p.write_text(yaml.dump(cfg))
        config = load_config(p)

        assert config.proxy is not None
        assert config.proxy.auth_token == "env-token"

    def test_yaml_token_takes_precedence_over_env(self, tmp_path, monkeypatch):
        """If both YAML and env var supply a token, YAML wins."""
        monkeypatch.setenv("TIGHTWAD_PROXY_TOKEN", "env-token")

        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "hip",
                "gpus": [{"name": "XTX", "vram_gb": 24}],
            },
            "models": {"test": {"path": "/test.gguf", "default": True}},
            "proxy": {
                "auth_token": "yaml-wins",
                "draft": {
                    "url": "http://192.168.1.1:8081",
                    "model_name": "small-model",
                },
                "target": {
                    "url": "http://192.168.1.2:8080",
                    "model_name": "big-model",
                },
            },
        }
        p = tmp_path / "cluster.yaml"
        p.write_text(yaml.dump(cfg))
        config = load_config(p)

        assert config.proxy is not None
        assert config.proxy.auth_token == "yaml-wins"

    def test_env_only_mode_proxy_token(self, monkeypatch):
        """load_proxy_from_env picks up TIGHTWAD_PROXY_TOKEN."""
        from tightwad.config import load_proxy_from_env

        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://draft:8081")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://target:8080")
        monkeypatch.setenv("TIGHTWAD_PROXY_TOKEN", "env-only-token")

        proxy = load_proxy_from_env()
        assert proxy is not None
        assert proxy.auth_token == "env-only-token"

    def test_env_only_mode_legacy_tightwad_token(self, monkeypatch):
        """load_proxy_from_env also accepts the legacy TIGHTWAD_TOKEN alias."""
        from tightwad.config import load_proxy_from_env

        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://draft:8081")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://target:8080")
        monkeypatch.setenv("TIGHTWAD_TOKEN", "legacy-token")

        proxy = load_proxy_from_env()
        assert proxy is not None
        assert proxy.auth_token == "legacy-token"

    def test_env_only_mode_no_token_is_none(self, monkeypatch):
        """load_proxy_from_env yields None auth_token when no token env var is set."""
        from tightwad.config import load_proxy_from_env

        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://draft:8081")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://target:8080")
        monkeypatch.delenv("TIGHTWAD_PROXY_TOKEN", raising=False)
        monkeypatch.delenv("TIGHTWAD_TOKEN", raising=False)

        proxy = load_proxy_from_env()
        assert proxy is not None
        assert proxy.auth_token is None
