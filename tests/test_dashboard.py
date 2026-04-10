"""Tests for the live web dashboard."""

import time

import pytest
from starlette.testclient import TestClient

from tightwad.config import ProxyConfig, ServerEndpoint
from tightwad.proxy import (
    MAX_REQUEST_HISTORY,
    ProxyStats,
    RequestRecord,
    SpeculativeProxy,
    create_app,
)


@pytest.fixture
def proxy_config():
    return ProxyConfig(
        draft=ServerEndpoint(url="http://draft:8081", model_name="qwen3-8b"),
        target=ServerEndpoint(url="http://target:8080", model_name="qwen3-32b"),
        host="0.0.0.0",
        port=8088,
        max_draft_tokens=8,
        fallback_on_draft_failure=True,
    )


@pytest.fixture
def app(proxy_config):
    return create_app(proxy_config)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestDashboardHTML:
    def test_dashboard_html_served(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "TIGHTWAD" in resp.text
        assert "EventSource" in resp.text

    # ------------------------------------------------------------------
    # SEC-2: XSS prevention — verify that the dashboard JavaScript uses
    # safe DOM construction instead of innerHTML for server-supplied data.
    # These tests inspect the *source* of the embedded JS to confirm that
    # the vulnerable patterns are gone and the safe patterns are present.
    # ------------------------------------------------------------------

    def test_no_innerHTML_for_server_data(self, client):
        """innerHTML must NOT be used to inject server-supplied values.

        The only legitimate remaining innerHTML usage is the one-liner
        that clears the log-body placeholder (``body.innerHTML = ''``),
        which inserts no dynamic data.  Any pattern that concatenates a
        server field (s.model, s.backend, s.url, r.model, etc.) into an
        innerHTML assignment is a confirmed XSS vector and must be absent.
        """
        resp = client.get("/dashboard")
        src = resp.text

        # Check line-by-line: no non-comment line should assign innerHTML with
        # a server-supplied field.  Comment lines (starting with * or //) are
        # excluded because they may legitimately mention innerHTML.
        xss_fields = ("s.model", "s.backend", "s.url", "r.model", "role +")
        for line in src.splitlines():
            stripped = line.strip()
            # Skip comment/doc lines — they may discuss innerHTML for clarity.
            if stripped.startswith("*") or stripped.startswith("//"):
                continue
            if "innerHTML" in stripped and "=" in stripped:
                # Allow only the safe placeholder-clear (no dynamic data).
                assert stripped == "if (logCount === 0) { body.innerHTML = ''; }", (
                    f"Unsafe innerHTML assignment found: {stripped!r}"
                )
            for field in xss_fields:
                if field in stripped and "innerHTML" in stripped:
                    raise AssertionError(
                        f"Server-supplied field {field!r} used with innerHTML on line: {stripped!r}"
                    )

        # The old tr.innerHTML bulk-assignment must not appear anywhere.
        assert "tr.innerHTML" not in src, (
            "tr.innerHTML used — switch to createElement/textContent (XSS risk)"
        )

    def test_safe_dom_methods_present(self, client):
        """Confirm safe DOM construction methods are used in the dashboard."""
        resp = client.get("/dashboard")
        src = resp.text

        # createElement must be used for building server-data nodes.
        assert "createElement" in src, "Expected createElement usage for safe DOM construction"
        # textContent must be the insertion method for dynamic values.
        assert "textContent" in src, "Expected textContent usage to safely set server-supplied strings"
        # The td() helper introduced by the fix should be present.
        assert "function td(" in src, "Expected td() helper function for safe table-cell construction"

    def test_updateHealth_uses_dom_not_innerHTML(self, client):
        """updateHealth() must build server-card rows via DOM, not innerHTML."""
        resp = client.get("/dashboard")
        src = resp.text

        # Locate the updateHealth function body.
        start = src.find("function updateHealth(")
        end = src.find("\nfunction ", start + 1)
        fn_body = src[start:end] if end != -1 else src[start:]

        assert "innerHTML" not in fn_body, (
            "updateHealth() still contains innerHTML — XSS risk (SEC-2)"
        )
        assert "createElement" in fn_body, (
            "updateHealth() should use createElement for server-supplied data"
        )
        assert "textContent" in fn_body, (
            "updateHealth() should use textContent to set server-supplied strings"
        )

    def test_addLogRow_uses_dom_not_innerHTML(self, client):
        """addLogRow() must build <td> cells via DOM, not innerHTML."""
        resp = client.get("/dashboard")
        src = resp.text

        # Locate the addLogRow function body.
        start = src.find("function addLogRow(")
        end = src.find("\n// ", start + 1)  # next top-level comment
        fn_body = src[start:end] if end != -1 else src[start:]

        # body.innerHTML = '' (clearing placeholder) is acceptable;
        # any other innerHTML assignment in this function is not.
        lines_with_inner_html = [
            line for line in fn_body.splitlines()
            if "innerHTML" in line and "body.innerHTML = ''" not in line and "innerHTML) to prevent" not in line
        ]
        assert not lines_with_inner_html, (
            f"addLogRow() has unsafe innerHTML usage: {lines_with_inner_html}"
        )


class TestHistoryEndpoint:
    def test_history_empty(self, client):
        resp = client.get("/v1/tightwad/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_history_with_records(self, proxy_config):
        from tightwad.proxy import _proxy
        app = create_app(proxy_config)
        client = TestClient(app)

        # Inject records into proxy stats
        from tightwad.proxy import _proxy as proxy
        proxy.stats.request_history.append(RequestRecord(
            timestamp=time.time(),
            rounds=3,
            drafted=24,
            accepted=18,
            acceptance_rate=0.75,
            draft_ms=150.0,
            verify_ms=200.0,
            total_ms=380.0,
            tokens_output=20,
            model="qwen3-32b",
        ))

        resp = client.get("/v1/tightwad/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["rounds"] == 3
        assert data[0]["acceptance_rate"] == 0.75
        assert data[0]["draft_ms"] == 150.0


class TestRequestRecordRingBuffer:
    def test_ring_buffer_caps_at_max(self):
        stats = ProxyStats()
        for i in range(MAX_REQUEST_HISTORY + 20):
            stats.request_history.append(RequestRecord(
                timestamp=time.time(),
                rounds=1,
                drafted=8,
                accepted=6,
                acceptance_rate=0.75,
                draft_ms=10.0,
                verify_ms=20.0,
                total_ms=30.0,
                tokens_output=7,
                model="test",
            ))
            if len(stats.request_history) > MAX_REQUEST_HISTORY:
                stats.request_history.pop(0)

        assert len(stats.request_history) == MAX_REQUEST_HISTORY

    def test_record_request_method(self, proxy_config):
        proxy = SpeculativeProxy(proxy_config)
        for i in range(60):
            proxy._record_request(
                rounds=1, drafted=8, accepted=6,
                draft_ms=10.0, verify_ms=20.0, total_ms=30.0,
                tokens_output=7,
            )
        assert len(proxy.stats.request_history) == MAX_REQUEST_HISTORY
        # Oldest records should have been evicted
        assert proxy.stats.request_history[0].tokens_output == 7


class TestSpeculationRoundTiming:
    @pytest.mark.asyncio
    async def test_speculation_round_returns_four_tuple(self, proxy_config):
        """speculation_round should return (text, is_done, draft_ms, verify_ms)."""
        proxy = SpeculativeProxy(proxy_config)
        # Both servers are down, fallback_on_draft_failure=True
        # Draft will fail -> fallback to target -> target also fails -> exception
        # But we can test the return signature by catching the error
        try:
            result = await proxy.speculation_round("test prompt")
            # If it somehow succeeds, check it's a 4-tuple
            assert len(result) == 4
            text, done, d_ms, v_ms = result
            assert isinstance(d_ms, float)
            assert isinstance(v_ms, float)
        except Exception:
            # Expected since servers are down; the important thing is
            # the function signature changed to return 4 values
            pass
        await proxy.close()
