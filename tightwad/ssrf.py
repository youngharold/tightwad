"""SSRF (Server-Side Request Forgery) protection for upstream proxy URLs.

This module validates upstream URLs before they are used to create httpx
clients.  It enforces:

1. **Scheme allowlist** — only ``http://`` and ``https://`` are permitted.
   ``file://``, ``gopher://``, ``ftp://``, and every other scheme are
   rejected outright.

2. **Private / internal IP blocklist** — requests targeting RFC-1918
   addresses, loopback, link-local (including the AWS/GCP instance-metadata
   service at 169.254.169.254), and IPv6 equivalents are blocked by default.

3. **DNS-rebinding prevention** — hostnames are resolved and the *resolved*
   IP address is checked against the blocklist, not just the literal value
   written in the config file.

4. **Opt-out for homelab deployments** — because Tightwad's most common use
   case is targeting LAN servers (e.g. ``http://192.168.1.10:11434``), the
   private-IP check can be disabled per-endpoint by setting
   ``allow_private_upstream: true`` in ``cluster.yaml``.  The scheme check
   is **always** enforced regardless of this flag.

Usage::

    from tightwad.ssrf import validate_upstream_url

    # Raises ValueError on failure:
    validate_upstream_url("http://192.168.1.10:11434", allow_private=True)
    validate_upstream_url("https://public-api.example.com")

Audit reference: SEC-5
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger("tightwad.ssrf")

# ---------------------------------------------------------------------------
# Blocklist of private / internal IP networks
# ---------------------------------------------------------------------------

#: Networks that must not be targeted by the proxy unless the operator has
#: explicitly opted in with ``allow_private_upstream: true``.
PRIVATE_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    # IPv4 ──────────────────────────────────────────────────────────────────
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("10.0.0.0/8"),        # RFC-1918 class A
    ipaddress.ip_network("172.16.0.0/12"),     # RFC-1918 class B
    ipaddress.ip_network("192.168.0.0/16"),    # RFC-1918 class C
    ipaddress.ip_network("169.254.0.0/16"),    # Link-local / IMDS (AWS, GCP, Azure)
    ipaddress.ip_network("0.0.0.0/8"),         # "This" network
    ipaddress.ip_network("100.64.0.0/10"),     # Shared address space (RFC-6598)
    ipaddress.ip_network("192.0.0.0/24"),      # IETF protocol assignments
    ipaddress.ip_network("192.0.2.0/24"),      # TEST-NET-1 (documentation)
    ipaddress.ip_network("198.51.100.0/24"),   # TEST-NET-2 (documentation)
    ipaddress.ip_network("203.0.113.0/24"),    # TEST-NET-3 (documentation)
    ipaddress.ip_network("240.0.0.0/4"),       # Reserved (class E)
    ipaddress.ip_network("255.255.255.255/32"),# Broadcast
    # IPv6 ──────────────────────────────────────────────────────────────────
    ipaddress.ip_network("::1/128"),           # Loopback
    ipaddress.ip_network("fc00::/7"),          # Unique-local (ULA) — fc00::/7 covers fc00:: and fd00::
    ipaddress.ip_network("fe80::/10"),         # Link-local
    ipaddress.ip_network("::/128"),            # Unspecified
    ipaddress.ip_network("::ffff:0:0/96"),     # IPv4-mapped IPv6
]

#: URL schemes that are explicitly allowed.  Everything else is rejected.
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

#: Schemes that are explicitly dangerous and mentioned in the issue/audit.
BLOCKED_SCHEMES: frozenset[str] = frozenset({
    "file", "gopher", "ftp", "ftps", "sftp", "ldap", "ldaps",
    "dict", "telnet", "tftp", "irc", "smtp", "pop3", "imap",
    "data", "javascript", "vbscript",
})


def _is_private_ip(addr: str) -> bool:
    """Return True if *addr* is within any of :data:`PRIVATE_NETWORKS`.

    Parameters
    ----------
    addr:
        A string representation of an IPv4 or IPv6 address.

    Returns
    -------
    bool
        ``True`` if the address matches a private/internal range.
    """
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        # Not a parseable IP address — treat conservatively as private.
        logger.warning("ssrf: could not parse resolved address %r — treating as private", addr)
        return True
    return any(ip in network for network in PRIVATE_NETWORKS)


def _resolve_host(hostname: str) -> list[str]:
    """Resolve *hostname* to a list of IP address strings.

    Uses :func:`socket.getaddrinfo` so that both IPv4 and IPv6 results are
    returned.  A ``socket.gaierror`` (DNS failure) is re-raised to the
    caller.

    Parameters
    ----------
    hostname:
        The hostname to resolve (e.g. ``"api.example.com"``).

    Returns
    -------
    list[str]
        All resolved IP addresses.

    Raises
    ------
    socket.gaierror
        If the hostname cannot be resolved.
    """
    results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    return [r[4][0] for r in results]


def validate_upstream_url(
    url: str,
    *,
    allow_private: bool = False,
    resolve_hostname: bool = True,
) -> None:
    """Validate *url* for use as an upstream proxy target.

    Performs three checks in order:

    1. **Scheme validation** — scheme must be ``http`` or ``https``.
    2. **Literal IP check** — if the host is already an IP literal and
       ``allow_private`` is ``False``, reject private/internal addresses.
    3. **DNS resolution check** — if the host is a hostname and
       ``resolve_hostname`` is ``True`` and ``allow_private`` is ``False``,
       resolve the hostname and check every returned address against the
       blocklist.

    Parameters
    ----------
    url:
        The upstream URL string to validate (e.g.
        ``"http://192.168.1.10:11434"``).
    allow_private:
        When ``True`` the private-IP and DNS-rebinding checks are skipped.
        The scheme check is **always** performed regardless of this flag.
        Set this when the upstream is intentionally a LAN / loopback server.
    resolve_hostname:
        When ``True`` (the default), hostname-based URLs are resolved via DNS
        and the resolved addresses are checked.  Set to ``False`` in unit
        tests to avoid real DNS lookups.

    Raises
    ------
    ValueError
        If the URL fails any validation check.  The error message is
        human-readable and safe to surface in logs / CLI output.
    """
    if not url or not isinstance(url, str):
        raise ValueError(f"Upstream URL must be a non-empty string, got: {url!r}")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    # ── 1. Scheme check ─────────────────────────────────────────────────────
    if scheme not in ALLOWED_SCHEMES:
        if scheme in BLOCKED_SCHEMES:
            raise ValueError(
                f"Upstream URL uses a dangerous scheme '{scheme}://' — "
                f"only http:// and https:// are permitted. Got: {url!r}"
            )
        raise ValueError(
            f"Upstream URL scheme must be 'http' or 'https', "
            f"got '{scheme}://' in: {url!r}"
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(
            f"Upstream URL is missing a hostname: {url!r}"
        )

    # Private-IP checks are opt-out for homelab users.
    if allow_private:
        logger.debug("ssrf: private upstream allowed for %r (allow_private=True)", url)
        return

    # ── 2. Literal IP address check ──────────────────────────────────────────
    try:
        # urlparse strips brackets from IPv6 literals like [::1]
        ip_obj = ipaddress.ip_address(hostname)
        if _is_private_ip(str(ip_obj)):
            raise ValueError(
                f"Upstream URL targets a private/internal IP address "
                f"({ip_obj}) which is blocked for security reasons. "
                f"Got: {url!r}. "
                f"If this is intentional (e.g. a LAN server), set "
                f"'allow_private_upstream: true' in your cluster.yaml."
            )
        # It's a public IP — all good.
        return
    except ValueError as exc:
        # Re-raise our own ValueError from the IP check above.
        if "Upstream URL" in str(exc):
            raise
        # ip_address() raised ValueError — hostname is not an IP literal.
        # Fall through to DNS resolution.

    # ── 3. DNS resolution + rebinding check ─────────────────────────────────
    if not resolve_hostname:
        logger.debug("ssrf: DNS resolution skipped for %r (resolve_hostname=False)", url)
        return

    try:
        resolved_ips = _resolve_host(hostname)
    except socket.gaierror as exc:
        # DNS failure — treat conservatively as a block.  This prevents a
        # temporarily-broken domain from being silently bypassed.
        raise ValueError(
            f"Upstream URL hostname {hostname!r} could not be resolved: {exc}. "
            f"Got: {url!r}"
        ) from exc

    if not resolved_ips:
        raise ValueError(
            f"Upstream URL hostname {hostname!r} resolved to no addresses. "
            f"Got: {url!r}"
        )

    for addr in resolved_ips:
        if _is_private_ip(addr):
            raise ValueError(
                f"Upstream URL hostname {hostname!r} resolves to a "
                f"private/internal IP address ({addr}) — blocked to prevent "
                f"DNS rebinding / SSRF. Got: {url!r}. "
                f"If this is intentional (e.g. a LAN server), set "
                f"'allow_private_upstream: true' in your cluster.yaml."
            )

    logger.debug(
        "ssrf: validated %r — resolved to %s",
        url,
        ", ".join(resolved_ips),
    )
