# Security

## SSRF Protection (SEC-5)

### Background

Tightwad's speculative decoding proxy makes outbound HTTP requests to upstream
servers (draft model, target model) using URLs supplied in `cluster.yaml` or
via environment variables.  Without validation, a misconfigured or tampered
config could cause the proxy to reach unintended internal infrastructure —
a **Server-Side Request Forgery (SSRF)** vulnerability.

This is especially relevant if URL updates ever become API-driven (e.g. a
future endpoint that lets clients change the target model URL), which would
escalate the threat from "misconfiguration risk" to "fully exploitable SSRF."

### What Is Validated

All upstream URLs are validated at startup (config load time) before any
HTTP clients are created:

| Check | Always enforced? | Notes |
|-------|-----------------|-------|
| Scheme allowlist (`http`, `https` only) | **Yes** | `file://`, `gopher://`, `ftp://`, `ldap://`, `data:`, `javascript:`, etc. are always rejected |
| Private IP blocklist | Configurable | See `allow_private_upstream` below |
| DNS-rebinding prevention | Configurable | Hostnames are resolved; the resolved IP is checked against the blocklist |

### Blocked IP Ranges

When private IP blocking is active, the following ranges are rejected:

| Range | Name |
|-------|------|
| `127.0.0.0/8` | IPv4 loopback |
| `10.0.0.0/8` | RFC-1918 class A |
| `172.16.0.0/12` | RFC-1918 class B |
| `192.168.0.0/16` | RFC-1918 class C |
| `169.254.0.0/16` | Link-local / Instance Metadata Service (AWS, GCP, Azure) |
| `100.64.0.0/10` | Shared address space (RFC-6598) |
| `0.0.0.0/8` | "This" network |
| `240.0.0.0/4` | Reserved (class E) |
| `255.255.255.255/32` | Broadcast |
| `::1/128` | IPv6 loopback |
| `fc00::/7` | IPv6 unique-local (ULA) |
| `fe80::/10` | IPv6 link-local |
| `::/128` | IPv6 unspecified |
| `::ffff:0:0/96` | IPv4-mapped IPv6 |

### `allow_private_upstream` Flag

Tightwad's primary use case is targeting **LAN servers** (e.g.
`http://192.168.1.10:11434`).  Blocking private IPs by default would break
every homelab deployment.  Therefore, the private-IP check is **opt-out**:

```yaml
# cluster.yaml — proxy section
proxy:
  # true  = homelab mode (default) — LAN addresses are allowed
  # false = strict mode — only public IPs/hostnames are allowed
  allow_private_upstream: true
```

Or via environment variable:

```bash
# false / 0 / no → strict mode
TIGHTWAD_ALLOW_PRIVATE_UPSTREAM=false tightwad proxy start
```

> **Note:** The **scheme check is always enforced**, regardless of
> `allow_private_upstream`.  `file://`, `gopher://`, etc. are always blocked.

### DNS Rebinding

When `allow_private_upstream: false`, hostnames are resolved via DNS and
**every resolved address** is checked against the blocklist.  If any resolved
IP is private, the URL is rejected.  This prevents a DNS rebinding attack
where a public-looking hostname (e.g. `evil.example.com`) resolves to an
internal address (`192.168.1.1`) at startup.

### Implementation

| File | Role |
|------|------|
| `tightwad/ssrf.py` | Core validation logic (`validate_upstream_url`, IP checks, DNS resolution) |
| `tightwad/config.py` | Calls validation in `load_config()` and `load_proxy_from_env()` before clients are created |

The validator is intentionally decoupled from the config loader so it can
be imported and called independently in future API-driven URL update paths.

### Future Considerations

If a future feature allows clients to dynamically update upstream URLs via
the API (e.g. `POST /v1/tightwad/config`), **always call
`validate_upstream_url(url, allow_private=False)` on any API-supplied URL**.
Do not inherit the operator's `allow_private_upstream` setting for URLs that
come from API clients.

### References

- OWASP SSRF: <https://owasp.org/Top10/A10_2021-Server-Side_Request_Forgery_%28SSRF%29/>
- Audit finding: SEC-5 (issue #7)
- RFC 1918: Private address space
- RFC 4193: IPv6 unique-local addresses (fc00::/7)
