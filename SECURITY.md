# Security Policy

## Reporting a Vulnerability

**Please do not file public GitHub issues for security vulnerabilities.**

To report a vulnerability privately, use [GitHub's private security advisory feature](https://github.com/youngharold/tightwad/security/advisories/new).

We will acknowledge your report within **48 hours** and aim to release a fix within **30 days** for confirmed vulnerabilities. We'll keep you updated throughout the process and credit you in the release notes if you'd like.

### What to Include

A good vulnerability report helps us reproduce and fix the issue faster. Where possible, please include:

- A clear description of the vulnerability and its potential impact
- Steps to reproduce (minimal proof-of-concept preferred)
- Affected version(s) or commit hash
- Any suggested fix or mitigation

## Scope

Tightwad is a network proxy that binds to local ports and makes outbound HTTP requests to llama.cpp / Ollama backends. Areas of particular interest:

| Area | Notes |
|------|-------|
| Proxy authentication bypass | Token auth protecting swarm endpoints |
| SSRF via proxy forwarding | Proxy passes requests to configurable backend URLs |
| IP allowlist bypass | `--allowed-ips` subnet filtering in swarm seeder |
| Arbitrary file read via swarm | Piece/manifest endpoints serve file chunks |
| Unsafe YAML loading | `cluster.yaml` config parsing |

Out of scope: issues in llama.cpp or Ollama themselves (report those upstream).

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest (`main`) | ✅ |
| older releases | ❌ (upgrade to latest) |

## Disclosure Policy

We follow a **coordinated disclosure** model. Once a fix is released, we will publish a GitHub Security Advisory describing the vulnerability. We ask that you wait until a fix is available before making details public.
