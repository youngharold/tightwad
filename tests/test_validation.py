"""Tests for input validation (issue #8 — CQ-1 + CQ-5).

Covers:
- parse_completion_request: max_tokens bounds, type errors, missing fields,
  invalid stop, invalid temperature, bad body type
- parse_chat_completion_request: messages validation (missing, empty, wrong
  type, missing role/content, wrong-typed role/content), same field checks
- _check_body_size: oversized Content-Length → 413, missing header → pass,
  invalid header → 400
- Integration tests via TestClient: validation errors surface as 400,
  oversized body surfaces as 413, valid requests pass through
- ProxyConfig max_tokens_limit / max_body_size propagation via env vars
  and YAML config
"""

from __future__ import annotations

import json

import pytest
import yaml
from starlette.testclient import TestClient

from tightwad.config import ProxyConfig, ServerEndpoint, load_config
from tightwad.proxy import create_app
from tightwad.validation import (
    ValidationError,
    parse_chat_completion_request,
    parse_completion_request,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def proxy_config():
    """Standard test ProxyConfig with default limits."""
    return ProxyConfig(
        draft=ServerEndpoint(url="http://draft:8081", model_name="qwen3-8b"),
        target=ServerEndpoint(url="http://target:8080", model_name="qwen3-32b"),
        host="0.0.0.0",
        port=8088,
        max_draft_tokens=8,
        fallback_on_draft_failure=True,
    )


@pytest.fixture
def low_limit_config():
    """ProxyConfig with a tight max_tokens_limit for testing."""
    return ProxyConfig(
        draft=ServerEndpoint(url="http://draft:8081", model_name="qwen3-8b"),
        target=ServerEndpoint(url="http://target:8080", model_name="qwen3-32b"),
        max_tokens_limit=100,
        max_body_size=512,
    )


# ---------------------------------------------------------------------------
# parse_completion_request — unit tests
# ---------------------------------------------------------------------------


class TestParseCompletionRequest:
    """Unit-level tests for the /v1/completions request parser."""

    def test_valid_minimal_request(self):
        """Minimal valid body parses successfully."""
        req = parse_completion_request({"prompt": "Hello"})
        assert req.prompt == "Hello"
        assert req.max_tokens == 256  # default
        assert req.temperature == 0.0  # default
        assert req.stream is False
        assert req.stop is None

    def test_valid_full_request(self):
        """All fields present and valid."""
        req = parse_completion_request({
            "prompt": "Hi",
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": False,
            "stop": ["<|end|>", "\n\n"],
        })
        assert req.max_tokens == 512
        assert req.temperature == 0.7
        assert req.stop == ["<|end|>", "\n\n"]

    def test_stop_string_converted_to_list(self):
        """A single stop string is coerced to a one-element list."""
        req = parse_completion_request({"prompt": "x", "stop": "END"})
        assert req.stop == ["END"]

    def test_prompt_defaults_to_empty_string(self):
        """Absent prompt gives empty string (OpenAI compat)."""
        req = parse_completion_request({})
        assert req.prompt == ""

    def test_max_tokens_at_default_limit(self):
        """max_tokens == limit (16384) should be accepted."""
        req = parse_completion_request({"max_tokens": 16384})
        assert req.max_tokens == 16384

    def test_max_tokens_exceeds_limit_raises(self):
        """max_tokens > limit raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"max_tokens": 16385})
        assert "max_tokens" in exc_info.value.message
        assert exc_info.value.field == "max_tokens"

    def test_max_tokens_custom_limit(self):
        """Custom limit is respected."""
        with pytest.raises(ValidationError):
            parse_completion_request({"max_tokens": 101}, max_tokens_limit=100)

    def test_max_tokens_at_custom_limit(self):
        """max_tokens == custom limit is accepted."""
        req = parse_completion_request({"max_tokens": 100}, max_tokens_limit=100)
        assert req.max_tokens == 100

    def test_max_tokens_zero_raises(self):
        """max_tokens=0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"max_tokens": 0})
        assert exc_info.value.field == "max_tokens"

    def test_max_tokens_negative_raises(self):
        """max_tokens=-1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"max_tokens": -1})
        assert exc_info.value.field == "max_tokens"

    def test_max_tokens_string_raises(self):
        """max_tokens as string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"max_tokens": "big"})
        assert exc_info.value.field == "max_tokens"

    def test_max_tokens_float_raises(self):
        """max_tokens as float raises ValidationError (must be int)."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"max_tokens": 100.5})
        assert exc_info.value.field == "max_tokens"

    def test_max_tokens_bool_raises(self):
        """max_tokens=True raises ValidationError (bool is subclass of int but invalid)."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"max_tokens": True})
        assert exc_info.value.field == "max_tokens"

    def test_temperature_out_of_range_high(self):
        """temperature > 2.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"temperature": 9999})
        assert exc_info.value.field == "temperature"

    def test_temperature_out_of_range_low(self):
        """temperature < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"temperature": -0.1})
        assert exc_info.value.field == "temperature"

    def test_temperature_string_raises(self):
        """temperature='warm' raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"temperature": "warm"})
        assert exc_info.value.field == "temperature"

    def test_temperature_bool_raises(self):
        """temperature=True raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"temperature": True})
        assert exc_info.value.field == "temperature"

    def test_temperature_boundary_zero(self):
        """temperature=0.0 is valid."""
        req = parse_completion_request({"temperature": 0.0})
        assert req.temperature == 0.0

    def test_temperature_boundary_two(self):
        """temperature=2.0 is valid."""
        req = parse_completion_request({"temperature": 2.0})
        assert req.temperature == 2.0

    def test_temperature_int_accepted(self):
        """temperature as int (e.g. 1) is coerced to float."""
        req = parse_completion_request({"temperature": 1})
        assert req.temperature == 1.0

    def test_stop_list_of_strings_accepted(self):
        """stop as a list of strings is valid."""
        req = parse_completion_request({"stop": ["a", "b"]})
        assert req.stop == ["a", "b"]

    def test_stop_list_with_non_string_raises(self):
        """stop list containing a non-string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"stop": [1, 2, 3]})
        assert exc_info.value.field == "stop"

    def test_stop_dict_raises(self):
        """stop as a dict raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"stop": {"key": "val"}})
        assert exc_info.value.field == "stop"

    def test_stop_int_raises(self):
        """stop as an int raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"stop": 42})
        assert exc_info.value.field == "stop"

    def test_prompt_non_string_raises(self):
        """prompt as a non-string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"prompt": 123})
        assert exc_info.value.field == "prompt"

    def test_prompt_list_raises(self):
        """prompt as a list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"prompt": ["hello"]})
        assert exc_info.value.field == "prompt"

    def test_stream_non_bool_raises(self):
        """stream='yes' raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request({"stream": "yes"})
        assert exc_info.value.field == "stream"

    def test_body_is_list_raises(self):
        """JSON array body raises ValidationError (not a dict)."""
        with pytest.raises(ValidationError) as exc_info:
            parse_completion_request([])
        assert "JSON object" in exc_info.value.message

    def test_body_is_string_raises(self):
        """JSON string body raises ValidationError."""
        with pytest.raises(ValidationError):
            parse_completion_request("hello")

    def test_body_is_none_raises(self):
        """None body raises ValidationError."""
        with pytest.raises(ValidationError):
            parse_completion_request(None)


# ---------------------------------------------------------------------------
# parse_chat_completion_request — unit tests
# ---------------------------------------------------------------------------


class TestParseChatCompletionRequest:
    """Unit-level tests for the /v1/chat/completions request parser."""

    VALID_MESSAGES = [{"role": "user", "content": "Hello"}]

    def test_valid_minimal_request(self):
        """Minimal valid chat body parses successfully."""
        req = parse_chat_completion_request({"messages": self.VALID_MESSAGES})
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"
        assert req.messages[0].content == "Hello"
        assert req.max_tokens == 256
        assert req.temperature == 0.0
        assert req.stream is False
        assert req.stop is None

    def test_valid_multi_turn(self):
        """Multi-turn messages parse correctly."""
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        req = parse_chat_completion_request({"messages": msgs})
        assert len(req.messages) == 3

    def test_messages_missing_raises(self):
        """Absent 'messages' key raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request({"max_tokens": 10})
        assert exc_info.value.field == "messages"

    def test_messages_empty_list_raises(self):
        """Empty messages list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request({"messages": []})
        assert exc_info.value.field == "messages"

    def test_messages_not_a_list_raises(self):
        """messages as a dict raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request({"messages": {"role": "user"}})
        assert exc_info.value.field == "messages"

    def test_messages_list_with_non_dict_raises(self):
        """A message that is not a dict raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request({"messages": ["hello"]})
        assert exc_info.value.field == "messages"

    def test_message_missing_role_raises(self):
        """Message without 'role' raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request({"messages": [{"content": "Hi"}]})
        assert exc_info.value.field == "messages"
        assert "role" in exc_info.value.message

    def test_message_non_string_role_raises(self):
        """Message with role=123 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request(
                {"messages": [{"role": 123, "content": "Hi"}]}
            )
        assert exc_info.value.field == "messages"
        assert "role" in exc_info.value.message

    def test_message_missing_content_raises(self):
        """Message without 'content' raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request({"messages": [{"role": "user"}]})
        assert exc_info.value.field == "messages"
        assert "content" in exc_info.value.message

    def test_message_non_string_content_raises(self):
        """Message with content=[1,2] raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request(
                {"messages": [{"role": "user", "content": [1, 2]}]}
            )
        assert exc_info.value.field == "messages"
        assert "content" in exc_info.value.message

    def test_max_tokens_exceeds_limit_raises(self):
        """max_tokens over limit raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request(
                {"messages": self.VALID_MESSAGES, "max_tokens": 99999}
            )
        assert exc_info.value.field == "max_tokens"

    def test_max_tokens_string_raises(self):
        """max_tokens as string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request(
                {"messages": self.VALID_MESSAGES, "max_tokens": "lots"}
            )
        assert exc_info.value.field == "max_tokens"

    def test_temperature_string_raises(self):
        """temperature='hot' raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request(
                {"messages": self.VALID_MESSAGES, "temperature": "hot"}
            )
        assert exc_info.value.field == "temperature"

    def test_stop_list_non_strings_raises(self):
        """stop list with non-string elements raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            parse_chat_completion_request(
                {"messages": self.VALID_MESSAGES, "stop": [1, 2]}
            )
        assert exc_info.value.field == "stop"

    def test_body_is_array_raises(self):
        """JSON array body raises ValidationError."""
        with pytest.raises(ValidationError):
            parse_chat_completion_request([])


# ---------------------------------------------------------------------------
# ValidationError helper
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_to_dict_with_field(self):
        err = ValidationError("bad value", field="max_tokens")
        d = err.to_dict()
        assert d["detail"] == "bad value"
        assert d["field"] == "max_tokens"

    def test_to_dict_without_field(self):
        err = ValidationError("body must be object")
        d = err.to_dict()
        assert "detail" in d
        assert "field" not in d

    def test_str_is_message(self):
        err = ValidationError("something went wrong")
        assert str(err) == "something went wrong"


# ---------------------------------------------------------------------------
# Integration tests via TestClient
# ---------------------------------------------------------------------------


class TestCompletionEndpointValidation:
    """/v1/completions returns 400 for invalid inputs, not 500."""

    def test_max_tokens_too_large_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={
            "prompt": "Hello",
            "max_tokens": 999999,
        })
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data
        assert "max_tokens" in data["detail"]

    def test_max_tokens_zero_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 0})
        assert resp.status_code == 400

    def test_max_tokens_negative_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"prompt": "x", "max_tokens": -5})
        assert resp.status_code == 400

    def test_max_tokens_string_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"max_tokens": "big"})
        assert resp.status_code == 400
        assert resp.json()["field"] == "max_tokens"

    def test_temperature_string_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"temperature": "warm"})
        assert resp.status_code == 400
        assert resp.json()["field"] == "temperature"

    def test_temperature_out_of_range_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"temperature": 9999})
        assert resp.status_code == 400

    def test_stop_list_non_string_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"stop": [1, 2, 3]})
        assert resp.status_code == 400
        assert resp.json()["field"] == "stop"

    def test_stop_dict_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"stop": {"bad": "value"}})
        assert resp.status_code == 400

    def test_body_array_returns_400(self, proxy_config):
        """A JSON array body (not an object) returns 400."""
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            content=b"[]",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_prompt_non_string_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"prompt": 42})
        assert resp.status_code == 400
        assert resp.json()["field"] == "prompt"

    def test_stream_non_bool_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"stream": "yes"})
        assert resp.status_code == 400

    def test_invalid_json_returns_400(self, proxy_config):
        """Malformed JSON body returns 400, not 500."""
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            content=b"{not valid json}",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_valid_request_reaches_backend(self, proxy_config):
        """Valid request passes validation (may 500 because backend is down)."""
        app = create_app(proxy_config)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/completions", json={
            "prompt": "Hello",
            "max_tokens": 10,
        })
        # Should NOT be 400 (validation error) — backend down gives 500
        assert resp.status_code != 400

    def test_custom_max_tokens_limit_enforced(self, low_limit_config):
        """max_tokens_limit from config is enforced."""
        app = create_app(low_limit_config)
        client = TestClient(app)
        resp = client.post("/v1/completions", json={"max_tokens": 101})
        assert resp.status_code == 400
        assert "100" in resp.json()["detail"]

    def test_custom_max_tokens_limit_accepts_at_limit(self, low_limit_config):
        """max_tokens == limit is accepted (may 500 due to backend down)."""
        app = create_app(low_limit_config)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/completions", json={"max_tokens": 100})
        assert resp.status_code != 400


class TestChatCompletionEndpointValidation:
    """/v1/chat/completions returns 400 for invalid inputs."""

    VALID_BODY = {"messages": [{"role": "user", "content": "Hi"}]}

    def test_messages_missing_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={"max_tokens": 10})
        assert resp.status_code == 400
        assert "messages" in resp.json()["detail"]

    def test_messages_empty_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 400

    def test_messages_not_list_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={"messages": "hello"})
        assert resp.status_code == 400

    def test_message_missing_role_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"content": "Hi"}]},
        )
        assert resp.status_code == 400
        assert "role" in resp.json()["detail"]

    def test_message_non_string_role_returns_400(self, proxy_config):
        """role=123 triggers 400."""
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": 123, "content": "Hi"}]},
        )
        assert resp.status_code == 400

    def test_message_missing_content_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user"}]},
        )
        assert resp.status_code == 400
        assert "content" in resp.json()["detail"]

    def test_message_non_string_content_returns_400(self, proxy_config):
        """content=[1,2] triggers 400."""
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": [1, 2]}]},
        )
        assert resp.status_code == 400

    def test_max_tokens_too_large_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={**self.VALID_BODY, "max_tokens": 999999},
        )
        assert resp.status_code == 400

    def test_temperature_string_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={**self.VALID_BODY, "temperature": "hot"},
        )
        assert resp.status_code == 400

    def test_stop_list_non_string_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={**self.VALID_BODY, "stop": [1, 2]},
        )
        assert resp.status_code == 400

    def test_invalid_json_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            content=b"{bad json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_body_array_returns_400(self, proxy_config):
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            content=b"[]",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_valid_request_reaches_backend(self, proxy_config):
        """Valid chat request passes validation (may 500 — backend down)."""
        app = create_app(proxy_config)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/chat/completions", json=self.VALID_BODY)
        assert resp.status_code != 400


# ---------------------------------------------------------------------------
# Body size limit tests
# ---------------------------------------------------------------------------


class TestBodySizeLimit:
    """Oversized payloads are rejected with 413 before buffering."""

    def test_content_length_over_limit_returns_413(self, proxy_config):
        """A Content-Length header exceeding max_body_size returns 413."""
        app = create_app(proxy_config)
        client = TestClient(app)
        # max_body_size defaults to 10MB; send a fake huge Content-Length
        huge = proxy_config.max_body_size + 1
        resp = client.post(
            "/v1/completions",
            json={"prompt": "x"},
            headers={"Content-Length": str(huge)},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_content_length_exactly_at_limit_passes(self, proxy_config):
        """Content-Length exactly equal to max_body_size is allowed."""
        app = create_app(proxy_config)
        client = TestClient(app, raise_server_exceptions=False)
        exact = proxy_config.max_body_size
        # This won't actually have a 10MB body in the test, but the header
        # check itself should not reject it.
        # We send a small body with a spoofed Content-Length == limit.
        resp = client.post(
            "/v1/completions",
            json={"prompt": "x"},
            headers={"Content-Length": str(exact)},
        )
        # Should NOT be 413 (the content-length matches the limit exactly)
        assert resp.status_code != 413

    def test_custom_max_body_size_enforced(self, low_limit_config):
        """Custom max_body_size (512 bytes) is respected."""
        app = create_app(low_limit_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={"prompt": "x"},
            headers={"Content-Length": "513"},
        )
        assert resp.status_code == 413

    def test_custom_max_body_size_accepts_at_limit(self, low_limit_config):
        """Content-Length == custom limit is not rejected."""
        app = create_app(low_limit_config)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/completions",
            json={"prompt": "x"},
            headers={"Content-Length": "512"},
        )
        assert resp.status_code != 413

    def test_missing_content_length_not_rejected(self, proxy_config):
        """Absent Content-Length header is not rejected (we only check the header)."""
        app = create_app(proxy_config)
        client = TestClient(app, raise_server_exceptions=False)
        # TestClient automatically sends Content-Length for json= payloads;
        # use a raw send without it to simulate omission.
        resp = client.post(
            "/v1/completions",
            content=b'{"prompt":"x","max_tokens":5}',
            # Do not set Content-Type here so TestClient won't add Content-Length
        )
        # Should not be 413
        assert resp.status_code != 413

    def test_chat_completion_over_limit_returns_413(self, proxy_config):
        """413 also works on /v1/chat/completions."""
        app = create_app(proxy_config)
        client = TestClient(app)
        huge = proxy_config.max_body_size + 1
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Content-Length": str(huge)},
        )
        assert resp.status_code == 413

    def test_invalid_content_length_returns_400(self, proxy_config):
        """A non-numeric Content-Length header returns 400."""
        app = create_app(proxy_config)
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={"prompt": "x"},
            headers={"Content-Length": "not-a-number"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# ProxyConfig limit propagation
# ---------------------------------------------------------------------------


class TestProxyConfigLimits:
    """max_tokens_limit and max_body_size are stored on ProxyConfig."""

    def test_default_max_tokens_limit(self):
        config = ProxyConfig(
            draft=ServerEndpoint(url="http://d:1", model_name="d"),
            target=ServerEndpoint(url="http://t:2", model_name="t"),
        )
        assert config.max_tokens_limit == 16384

    def test_default_max_body_size(self):
        config = ProxyConfig(
            draft=ServerEndpoint(url="http://d:1", model_name="d"),
            target=ServerEndpoint(url="http://t:2", model_name="t"),
        )
        assert config.max_body_size == 10 * 1024 * 1024

    def test_custom_limits_stored(self):
        config = ProxyConfig(
            draft=ServerEndpoint(url="http://d:1", model_name="d"),
            target=ServerEndpoint(url="http://t:2", model_name="t"),
            max_tokens_limit=500,
            max_body_size=1024,
        )
        assert config.max_tokens_limit == 500
        assert config.max_body_size == 1024

    def test_env_max_tokens_limit(self, monkeypatch):
        """TIGHTWAD_MAX_TOKENS_LIMIT sets max_tokens_limit."""
        from tightwad.config import load_proxy_from_env

        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://draft:8081")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://target:8080")
        monkeypatch.setenv("TIGHTWAD_MAX_TOKENS_LIMIT", "8192")

        proxy = load_proxy_from_env()
        assert proxy is not None
        assert proxy.max_tokens_limit == 8192

    def test_env_max_body_size(self, monkeypatch):
        """TIGHTWAD_MAX_BODY_SIZE sets max_body_size."""
        from tightwad.config import load_proxy_from_env

        monkeypatch.setenv("TIGHTWAD_DRAFT_URL", "http://draft:8081")
        monkeypatch.setenv("TIGHTWAD_TARGET_URL", "http://target:8080")
        monkeypatch.setenv("TIGHTWAD_MAX_BODY_SIZE", "5242880")

        proxy = load_proxy_from_env()
        assert proxy is not None
        assert proxy.max_body_size == 5 * 1024 * 1024

    def test_yaml_max_tokens_limit(self, tmp_path):
        """proxy.max_tokens_limit in cluster.yaml is loaded."""
        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "hip",
                "gpus": [{"name": "XTX", "vram_gb": 24}],
            },
            "models": {"test": {"path": "/test.gguf", "default": True}},
            "proxy": {
                "max_tokens_limit": 4096,
                "draft": {"url": "http://192.168.1.1:8081", "model_name": "small"},
                "target": {"url": "http://192.168.1.2:8080", "model_name": "big"},
            },
        }
        p = tmp_path / "cluster.yaml"
        p.write_text(yaml.dump(cfg))
        config = load_config(p)
        assert config.proxy is not None
        assert config.proxy.max_tokens_limit == 4096

    def test_yaml_max_body_size(self, tmp_path):
        """proxy.max_body_size in cluster.yaml is loaded."""
        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "hip",
                "gpus": [{"name": "XTX", "vram_gb": 24}],
            },
            "models": {"test": {"path": "/test.gguf", "default": True}},
            "proxy": {
                "max_body_size": 2097152,  # 2 MB
                "draft": {"url": "http://192.168.1.1:8081", "model_name": "small"},
                "target": {"url": "http://192.168.1.2:8080", "model_name": "big"},
            },
        }
        p = tmp_path / "cluster.yaml"
        p.write_text(yaml.dump(cfg))
        config = load_config(p)
        assert config.proxy is not None
        assert config.proxy.max_body_size == 2 * 1024 * 1024

    def test_yaml_defaults_when_limits_omitted(self, tmp_path):
        """When limits are absent from YAML, defaults apply."""
        cfg = {
            "coordinator": {
                "host": "0.0.0.0",
                "port": 8080,
                "backend": "hip",
                "gpus": [{"name": "XTX", "vram_gb": 24}],
            },
            "models": {"test": {"path": "/test.gguf", "default": True}},
            "proxy": {
                "draft": {"url": "http://192.168.1.1:8081", "model_name": "small"},
                "target": {"url": "http://192.168.1.2:8080", "model_name": "big"},
            },
        }
        p = tmp_path / "cluster.yaml"
        p.write_text(yaml.dump(cfg))
        config = load_config(p)
        assert config.proxy is not None
        assert config.proxy.max_tokens_limit == 16384
        assert config.proxy.max_body_size == 10 * 1024 * 1024
