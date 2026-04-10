"""Input validation for API request bodies.

Provides type-checked, bounds-validated dataclasses for the
``/v1/completions`` and ``/v1/chat/completions`` endpoints, plus helpers
for enforcing request body size limits.

Audit refs: CQ-1 (input validation), CQ-5 (body size limit) — issue #8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when a request body fails validation.

    Attributes
    ----------
    message:
        Human-readable description of the validation failure.
    field:
        The request field that failed validation, or ``None`` for
        top-level / structural errors.
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict for error responses."""
        d: dict = {"detail": self.message}
        if self.field is not None:
            d["field"] = self.field
        return d


# ---------------------------------------------------------------------------
# Validated request models
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str
    content: str


@dataclass
class CompletionRequest:
    """Validated body for ``POST /v1/completions``.

    Parameters
    ----------
    prompt:
        The prompt string.  Must be a ``str``.
    max_tokens:
        Maximum tokens to generate.  Must be a positive integer no greater
        than *max_tokens_limit*.
    temperature:
        Sampling temperature in ``[0.0, 2.0]``.
    stream:
        Whether to stream the response via SSE.
    stop:
        One or more stop sequences.  When a list, every element must be a
        ``str``.
    """

    prompt: str
    max_tokens: int
    temperature: float
    stream: bool
    stop: list[str] | None


@dataclass
class ChatCompletionRequest:
    """Validated body for ``POST /v1/chat/completions``.

    Parameters
    ----------
    messages:
        Non-empty list of chat messages.
    max_tokens:
        Maximum tokens to generate.  Must be a positive integer no greater
        than *max_tokens_limit*.
    temperature:
        Sampling temperature in ``[0.0, 2.0]``.
    stream:
        Whether to stream the response via SSE.
    stop:
        One or more stop sequences.  When a list, every element must be a
        ``str``.
    """

    messages: list[ChatMessage]
    max_tokens: int
    temperature: float
    stream: bool
    stop: list[str] | None


# ---------------------------------------------------------------------------
# Parsing / validation helpers
# ---------------------------------------------------------------------------


def _require_dict(body: object) -> dict:
    """Assert that the top-level JSON value is a dict.

    A valid JSON *array* (``[]``) would make ``.get()`` raise
    ``AttributeError``. We catch this early and return a clean 400.
    """
    if not isinstance(body, dict):
        raise ValidationError(
            f"Request body must be a JSON object, got {type(body).__name__}"
        )
    return body


def _validate_max_tokens(
    value: object,
    *,
    default: int = 256,
    limit: int = 16384,
) -> int:
    """Return a validated *max_tokens* integer.

    Raises ``ValidationError`` for non-integers, values ≤ 0, or values
    exceeding *limit*.
    """
    if value is None:
        return default

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            f"'max_tokens' must be an integer, got {type(value).__name__}",
            field="max_tokens",
        )
    if value <= 0:
        raise ValidationError(
            f"'max_tokens' must be a positive integer, got {value}",
            field="max_tokens",
        )
    if value > limit:
        raise ValidationError(
            f"'max_tokens' must not exceed {limit}, got {value}",
            field="max_tokens",
        )
    return value


def _validate_temperature(value: object, *, default: float = 0.0) -> float:
    """Return a validated *temperature* float.

    Accepts both ``int`` and ``float`` literals.  Rejects strings and
    values outside ``[0.0, 2.0]``.
    """
    if value is None:
        return default

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"'temperature' must be a number, got {type(value).__name__}",
            field="temperature",
        )
    fval = float(value)
    if fval < 0.0 or fval > 2.0:
        raise ValidationError(
            f"'temperature' must be between 0.0 and 2.0, got {fval}",
            field="temperature",
        )
    return fval


def _validate_stop(value: object) -> list[str] | None:
    """Return a validated *stop* sequence list, or ``None``.

    Accepts:
    - ``None`` → ``None``
    - a ``str`` → ``[str]``
    - a list of ``str`` → the list as-is

    Raises ``ValidationError`` for any other type or for lists that
    contain non-string elements.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        for i, item in enumerate(value):
            if not isinstance(item, str):
                raise ValidationError(
                    f"'stop[{i}]' must be a string, got {type(item).__name__}",
                    field="stop",
                )
        return value  # type: ignore[return-value]

    raise ValidationError(
        f"'stop' must be a string or list of strings, got {type(value).__name__}",
        field="stop",
    )


def _validate_stream(value: object, *, default: bool = False) -> bool:
    """Return a validated *stream* bool."""
    if value is None:
        return default

    if not isinstance(value, bool):
        raise ValidationError(
            f"'stream' must be a boolean, got {type(value).__name__}",
            field="stream",
        )
    return value


def _validate_prompt(value: object) -> str:
    """Return a validated *prompt* string.

    An absent prompt defaults to an empty string (matching OpenAI behaviour),
    but a present prompt that is not a string raises ``ValidationError``.
    """
    if value is None:
        return ""

    if not isinstance(value, str):
        raise ValidationError(
            f"'prompt' must be a string, got {type(value).__name__}",
            field="prompt",
        )
    return value


def _validate_messages(value: object) -> list[ChatMessage]:
    """Return a validated non-empty list of :class:`ChatMessage` objects.

    Each element must be a dict with at least a ``role`` key (str) and
    a ``content`` key (str).  Missing or wrong-typed fields raise
    ``ValidationError``.
    """
    if value is None:
        raise ValidationError(
            "'messages' is required for chat completions",
            field="messages",
        )

    if not isinstance(value, list):
        raise ValidationError(
            f"'messages' must be a list, got {type(value).__name__}",
            field="messages",
        )

    if len(value) == 0:
        raise ValidationError(
            "'messages' must not be empty",
            field="messages",
        )

    result: list[ChatMessage] = []
    for i, msg in enumerate(value):
        if not isinstance(msg, dict):
            raise ValidationError(
                f"'messages[{i}]' must be an object, got {type(msg).__name__}",
                field="messages",
            )

        role = msg.get("role")
        if role is None:
            raise ValidationError(
                f"'messages[{i}].role' is required",
                field="messages",
            )
        if not isinstance(role, str):
            raise ValidationError(
                f"'messages[{i}].role' must be a string, got {type(role).__name__}",
                field="messages",
            )

        content = msg.get("content")
        if content is None:
            raise ValidationError(
                f"'messages[{i}].content' is required",
                field="messages",
            )
        if not isinstance(content, str):
            raise ValidationError(
                f"'messages[{i}].content' must be a string, "
                f"got {type(content).__name__}",
                field="messages",
            )

        result.append(ChatMessage(role=role, content=content))

    return result


# ---------------------------------------------------------------------------
# Public parse functions
# ---------------------------------------------------------------------------


def parse_completion_request(
    body: object,
    *,
    max_tokens_limit: int = 16384,
) -> CompletionRequest:
    """Parse and validate a ``/v1/completions`` request body.

    Parameters
    ----------
    body:
        The decoded JSON value (as returned by ``await request.json()``).
    max_tokens_limit:
        Hard upper bound on ``max_tokens``.

    Returns
    -------
    CompletionRequest
        A fully validated request object.

    Raises
    ------
    ValidationError
        On any validation failure.
    """
    d = _require_dict(body)

    prompt = _validate_prompt(d.get("prompt"))
    max_tokens = _validate_max_tokens(
        d.get("max_tokens"), default=256, limit=max_tokens_limit
    )
    temperature = _validate_temperature(d.get("temperature"), default=0.0)
    stream = _validate_stream(d.get("stream"), default=False)
    stop = _validate_stop(d.get("stop"))

    return CompletionRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        stop=stop,
    )


def parse_chat_completion_request(
    body: object,
    *,
    max_tokens_limit: int = 16384,
) -> ChatCompletionRequest:
    """Parse and validate a ``/v1/chat/completions`` request body.

    Parameters
    ----------
    body:
        The decoded JSON value (as returned by ``await request.json()``).
    max_tokens_limit:
        Hard upper bound on ``max_tokens``.

    Returns
    -------
    ChatCompletionRequest
        A fully validated request object.

    Raises
    ------
    ValidationError
        On any validation failure.
    """
    d = _require_dict(body)

    messages = _validate_messages(d.get("messages"))
    max_tokens = _validate_max_tokens(
        d.get("max_tokens"), default=256, limit=max_tokens_limit
    )
    temperature = _validate_temperature(d.get("temperature"), default=0.0)
    stream = _validate_stream(d.get("stream"), default=False)
    stop = _validate_stop(d.get("stop"))

    return ChatCompletionRequest(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        stop=stop,
    )
