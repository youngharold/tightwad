"""Chat template rendering for /v1/chat/completions.

Maps model families to their chat templates so the proxy formats prompts
correctly for any model — not just Qwen3.  Mismatched templates silently
destroy speculation acceptance rates because the target model sees garbage
tokens.

Templates are intentionally simple string formatting (no Jinja2 dependency).
Each template defines how to wrap system/user/assistant messages into the
raw prompt that the model expects.

Auto-detection order:
1. Explicit ``proxy.chat_template`` in cluster.yaml (e.g. ``"llama3"``)
2. ``TIGHTWAD_CHAT_TEMPLATE`` environment variable
3. Detected from target model family (via :mod:`tightwad.family`)
4. Fallback to ChatML (Qwen-style ``<|im_start|>`` format)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("tightwad.chat_templates")


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------


@dataclass
class ChatTemplate:
    """A chat template that can render OpenAI-format messages to a raw prompt."""

    name: str
    # Stop tokens the model uses to signal end of assistant turn
    stop_tokens: list[str]

    def render(self, messages: list[dict]) -> str:
        """Render a list of OpenAI chat messages to a raw prompt string."""
        raise NotImplementedError


class ChatMLTemplate(ChatTemplate):
    """ChatML format used by Qwen, Yi, and other models.

    ```
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Hello<|im_end|>
    <|im_start|>assistant
    ```
    """

    def __init__(self, default_system: str = "You are a helpful assistant."):
        super().__init__(name="chatml", stop_tokens=["<|im_end|>"])
        self.default_system = default_system

    def render(self, messages: list[dict]) -> str:
        system = self.default_system
        parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                system = content
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            + "".join(parts)
            + "<|im_start|>assistant\n"
        )


class Llama3Template(ChatTemplate):
    """Llama 3 / 3.1 / 3.2 / 3.3 chat format.

    ```
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

    Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    ```
    """

    def __init__(self):
        super().__init__(
            name="llama3",
            stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        )

    def render(self, messages: list[dict]) -> str:
        system = "You are a helpful assistant."
        parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                system = content
            else:
                parts.append(
                    f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    f"{content}<|eot_id|>"
                )

        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            + "".join(parts)
            + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )


class MistralTemplate(ChatTemplate):
    """Mistral / Mixtral instruction format.

    ```
    [INST] You are a helpful assistant.

    Hello [/INST]
    ```
    """

    def __init__(self):
        super().__init__(name="mistral", stop_tokens=["</s>"])

    def render(self, messages: list[dict]) -> str:
        system = ""
        conversation: list[dict] = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg.get("content", "")
            else:
                conversation.append(msg)

        parts: list[str] = []
        for i, msg in enumerate(conversation):
            content = msg.get("content", "")
            if msg["role"] == "user":
                if i == 0 and system:
                    parts.append(f"[INST] {system}\n\n{content} [/INST]")
                else:
                    parts.append(f"[INST] {content} [/INST]")
            elif msg["role"] == "assistant":
                parts.append(f" {content}</s>")

        return "".join(parts)


class GemmaTemplate(ChatTemplate):
    """Gemma 2 / 3 chat format.

    ```
    <start_of_turn>user
    Hello<end_of_turn>
    <start_of_turn>model
    ```
    """

    def __init__(self):
        super().__init__(name="gemma", stop_tokens=["<end_of_turn>"])

    def render(self, messages: list[dict]) -> str:
        parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                # Gemma doesn't have a native system role — prepend to first user
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
            elif role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")

        return "".join(parts) + "<start_of_turn>model\n"


class PhiTemplate(ChatTemplate):
    """Phi-3 / Phi-4 chat format.

    ```
    <|system|>
    You are a helpful assistant.<|end|>
    <|user|>
    Hello<|end|>
    <|assistant|>
    ```
    """

    def __init__(self):
        super().__init__(name="phi", stop_tokens=["<|end|>"])

    def render(self, messages: list[dict]) -> str:
        parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}<|end|>\n")

        return "".join(parts) + "<|assistant|>\n"


class DeepSeekTemplate(ChatTemplate):
    """DeepSeek V2/V3 chat format (uses ChatML-like with <|begin_of_sentence|>).

    Falls back to ChatML since DeepSeek uses a compatible template.
    """

    def __init__(self):
        super().__init__(
            name="deepseek",
            stop_tokens=["<|end_of_sentence|>", "<|im_end|>"],
        )
        self._inner = ChatMLTemplate()

    def render(self, messages: list[dict]) -> str:
        return self._inner.render(messages)


class CommandRTemplate(ChatTemplate):
    """Cohere Command-R format.

    ```
    <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are helpful<|END_OF_TURN_TOKEN|>
    <|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello<|END_OF_TURN_TOKEN|>
    <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
    ```
    """

    def __init__(self):
        super().__init__(name="command-r", stop_tokens=["<|END_OF_TURN_TOKEN|>"])

    def render(self, messages: list[dict]) -> str:
        role_map = {"system": "SYSTEM_TOKEN", "user": "USER_TOKEN", "assistant": "CHATBOT_TOKEN"}
        parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            token = role_map.get(role, "USER_TOKEN")
            parts.append(
                f"<|START_OF_TURN_TOKEN|><|{token}|>{content}<|END_OF_TURN_TOKEN|>\n"
            )

        return "".join(parts) + "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# Template instances keyed by name
_TEMPLATES: dict[str, ChatTemplate] = {
    "chatml": ChatMLTemplate(),
    "qwen": ChatMLTemplate(),
    "llama3": Llama3Template(),
    "llama": Llama3Template(),
    "mistral": MistralTemplate(),
    "mixtral": MistralTemplate(),
    "gemma": GemmaTemplate(),
    "phi": PhiTemplate(),
    "deepseek": DeepSeekTemplate(),
    "command-r": CommandRTemplate(),
}

# Map model families (from family.py) to template names
_FAMILY_TO_TEMPLATE: dict[str, str] = {
    "llama": "llama3",
    "qwen": "chatml",
    "mistral": "mistral",
    "gemma": "gemma",
    "phi": "phi",
    "deepseek": "deepseek",
    "command-r": "command-r",
    "chatglm": "chatml",
    "internlm": "chatml",
    "minicpm": "chatml",
    "starcoder": "chatml",
    "falcon": "chatml",
}

# Default when nothing else matches
_DEFAULT_TEMPLATE = "chatml"


def get_template(name: str) -> ChatTemplate | None:
    """Look up a chat template by name."""
    return _TEMPLATES.get(name.lower().strip())


def get_template_for_family(family: str) -> ChatTemplate:
    """Get the appropriate chat template for a model family."""
    template_name = _FAMILY_TO_TEMPLATE.get(family.lower(), _DEFAULT_TEMPLATE)
    return _TEMPLATES[template_name]


def get_default_template() -> ChatTemplate:
    """Return the default ChatML template."""
    return _TEMPLATES[_DEFAULT_TEMPLATE]


def list_templates() -> list[str]:
    """Return all available template names."""
    return sorted(set(_TEMPLATES.keys()))


def apply_chat_template(
    messages: list[dict],
    template: ChatTemplate | None = None,
) -> tuple[str, list[str]]:
    """Render messages using the given template (or default ChatML).

    Returns (prompt_string, stop_tokens).
    """
    if template is None:
        template = get_default_template()
    return template.render(messages), template.stop_tokens


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


async def detect_chat_template(
    target_url: str,
    target_model: str,
    target_backend: str,
) -> ChatTemplate | None:
    """Auto-detect the appropriate chat template from the target model.

    Uses :mod:`tightwad.family` to detect the model architecture, then
    maps it to the corresponding template.
    """
    try:
        from .family import detect_family

        family_info = await detect_family(target_url, target_model, target_backend)
        if family_info:
            template = get_template_for_family(family_info.family)
            logger.info(
                "Auto-detected chat template: %s (family: %s, arch: %s)",
                template.name, family_info.family, family_info.arch,
            )
            return template
    except Exception as exc:
        logger.debug("Chat template auto-detection failed: %s", exc)

    return None
