# Contributing to Tightwad

Thanks for wanting to help. Here's how to get started.

## Dev Setup

```bash
git clone https://github.com/youngharold/tightwad.git
cd tightwad
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

All tests should pass. If you see failures in `test_init_wizard.py`, those are known (unimplemented CLI feature) and marked as expected failures.

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Run `pytest` and make sure everything passes
5. Open a PR against `main`

## PR Expectations

- Describe what you changed and why
- Keep PRs focused — one feature or fix per PR
- Tests pass
- No hardcoded IPs, paths, or credentials in committed code
- Update CHANGELOG.md if it's a user-facing change

## Reporting Bugs

Open an issue with:
- Your Tightwad version (`tightwad --version`)
- OS and Python version
- Hardware (GPU models, VRAM)
- Steps to reproduce
- Sanitize your `cluster.yaml` before pasting (remove real IPs and hostnames)

## Security Issues

**Do not file public issues for security vulnerabilities.** See [SECURITY.md](SECURITY.md) for how to report them privately.

## Code Style

- Keep it readable
- Match the style of surrounding code
- No unnecessary dependencies

## Questions?

Open a GitHub Discussion or file an issue. We're happy to help.
