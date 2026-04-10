"""Tests for chat template rendering and auto-detection."""

import pytest

from tightwad.chat_templates import (
    ChatMLTemplate,
    CommandRTemplate,
    GemmaTemplate,
    Llama3Template,
    MistralTemplate,
    PhiTemplate,
    apply_chat_template,
    get_template,
    get_template_for_family,
    get_default_template,
    list_templates,
)


SIMPLE_MESSAGES = [
    {"role": "user", "content": "Hello"},
]

SYSTEM_MESSAGES = [
    {"role": "system", "content": "You are a pirate."},
    {"role": "user", "content": "Hello"},
]

MULTI_TURN = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "How are you?"},
]


# ---------------------------------------------------------------------------
# ChatML / Qwen
# ---------------------------------------------------------------------------


class TestChatMLTemplate:
    def test_simple_message(self):
        t = ChatMLTemplate()
        result = t.render(SIMPLE_MESSAGES)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user\nHello<|im_end|>" in result
        assert result.endswith("<|im_start|>assistant\n")

    def test_system_message(self):
        t = ChatMLTemplate()
        result = t.render(SYSTEM_MESSAGES)
        assert "You are a pirate." in result
        assert "You are a helpful assistant." not in result

    def test_multi_turn(self):
        t = ChatMLTemplate()
        result = t.render(MULTI_TURN)
        assert "Be brief." in result
        assert "<|im_start|>assistant\nHello!<|im_end|>" in result
        assert result.endswith("<|im_start|>assistant\n")

    def test_stop_tokens(self):
        t = ChatMLTemplate()
        assert "<|im_end|>" in t.stop_tokens


# ---------------------------------------------------------------------------
# Llama 3
# ---------------------------------------------------------------------------


class TestLlama3Template:
    def test_simple_message(self):
        t = Llama3Template()
        result = t.render(SIMPLE_MESSAGES)
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "Hello<|eot_id|>" in result
        assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_system_message(self):
        t = Llama3Template()
        result = t.render(SYSTEM_MESSAGES)
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "You are a pirate." in result

    def test_stop_tokens(self):
        t = Llama3Template()
        assert "<|eot_id|>" in t.stop_tokens

    def test_multi_turn(self):
        t = Llama3Template()
        result = t.render(MULTI_TURN)
        assert "Hello!<|eot_id|>" in result


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------


class TestMistralTemplate:
    def test_simple_message(self):
        t = MistralTemplate()
        result = t.render(SIMPLE_MESSAGES)
        assert "[INST]" in result
        assert "Hello" in result
        assert "[/INST]" in result

    def test_system_prepended_to_first_user(self):
        t = MistralTemplate()
        result = t.render(SYSTEM_MESSAGES)
        assert "You are a pirate." in result
        # System and first user should be in the same [INST] block
        assert "[INST] You are a pirate.\n\nHello [/INST]" in result

    def test_stop_tokens(self):
        t = MistralTemplate()
        assert "</s>" in t.stop_tokens


# ---------------------------------------------------------------------------
# Gemma
# ---------------------------------------------------------------------------


class TestGemmaTemplate:
    def test_simple_message(self):
        t = GemmaTemplate()
        result = t.render(SIMPLE_MESSAGES)
        assert "<start_of_turn>user" in result
        assert "Hello<end_of_turn>" in result
        assert result.endswith("<start_of_turn>model\n")

    def test_system_as_user_turn(self):
        t = GemmaTemplate()
        result = t.render(SYSTEM_MESSAGES)
        # Gemma doesn't have native system role
        assert "<start_of_turn>user\nYou are a pirate.<end_of_turn>" in result

    def test_stop_tokens(self):
        t = GemmaTemplate()
        assert "<end_of_turn>" in t.stop_tokens


# ---------------------------------------------------------------------------
# Phi
# ---------------------------------------------------------------------------


class TestPhiTemplate:
    def test_simple_message(self):
        t = PhiTemplate()
        result = t.render(SIMPLE_MESSAGES)
        assert "<|user|>" in result
        assert "Hello<|end|>" in result
        assert result.endswith("<|assistant|>\n")

    def test_system_message(self):
        t = PhiTemplate()
        result = t.render(SYSTEM_MESSAGES)
        assert "<|system|>\nYou are a pirate.<|end|>" in result

    def test_stop_tokens(self):
        t = PhiTemplate()
        assert "<|end|>" in t.stop_tokens


# ---------------------------------------------------------------------------
# Command-R
# ---------------------------------------------------------------------------


class TestCommandRTemplate:
    def test_simple_message(self):
        t = CommandRTemplate()
        result = t.render(SIMPLE_MESSAGES)
        assert "<|USER_TOKEN|>Hello" in result
        assert result.endswith("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")

    def test_stop_tokens(self):
        t = CommandRTemplate()
        assert "<|END_OF_TURN_TOKEN|>" in t.stop_tokens


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_get_template_by_name(self):
        assert get_template("chatml") is not None
        assert get_template("llama3") is not None
        assert get_template("mistral") is not None
        assert get_template("gemma") is not None
        assert get_template("phi") is not None

    def test_get_template_case_insensitive(self):
        assert get_template("ChatML") is not None
        assert get_template("LLAMA3") is not None

    def test_get_template_unknown(self):
        assert get_template("nonexistent") is None

    def test_get_template_for_family(self):
        t = get_template_for_family("llama")
        assert t.name == "llama3"

        t = get_template_for_family("qwen")
        assert t.name == "chatml"

        t = get_template_for_family("mistral")
        assert t.name == "mistral"

    def test_get_template_for_unknown_family_returns_default(self):
        t = get_template_for_family("some_unknown_family")
        assert t.name == "chatml"

    def test_default_template(self):
        t = get_default_template()
        assert t.name == "chatml"

    def test_list_templates(self):
        names = list_templates()
        assert "chatml" in names
        assert "llama3" in names
        assert "mistral" in names

    def test_qwen_alias(self):
        assert get_template("qwen") is not None
        assert get_template("qwen").name == "chatml"

    def test_llama_alias(self):
        assert get_template("llama") is not None
        assert get_template("llama").name == "llama3"


# ---------------------------------------------------------------------------
# apply_chat_template
# ---------------------------------------------------------------------------


class TestApplyChatTemplate:
    def test_returns_tuple(self):
        prompt, stop = apply_chat_template(SIMPLE_MESSAGES)
        assert isinstance(prompt, str)
        assert isinstance(stop, list)
        assert len(stop) > 0

    def test_default_uses_chatml(self):
        prompt, stop = apply_chat_template(SIMPLE_MESSAGES)
        assert "<|im_start|>" in prompt
        assert "<|im_end|>" in stop

    def test_with_explicit_template(self):
        t = Llama3Template()
        prompt, stop = apply_chat_template(SIMPLE_MESSAGES, template=t)
        assert "<|begin_of_text|>" in prompt
        assert "<|eot_id|>" in stop
