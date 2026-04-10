"""Model family detection and compatibility validation.

Detects the architecture family of draft and target models to warn users
when they pair incompatible models for speculative decoding.  Mismatched
families cause catastrophically low acceptance rates (e.g. 1.6% instead
of 70%+) with no obvious error — just 10x slower inference.

Detection sources (tried in order):
1. Ollama ``/api/show`` — returns ``model_info`` with architecture keys
2. llama-server ``/v1/models`` or ``/props`` — limited metadata
3. GGUF file ``general.architecture`` — via :mod:`tightwad.gguf_reader`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger("tightwad.family")


# ---------------------------------------------------------------------------
# Architecture family mapping
# ---------------------------------------------------------------------------

# Maps raw architecture strings (from GGUF / Ollama) to canonical families.
# Models within the same family are compatible for speculative decoding.
# Models in different families will have near-zero acceptance rates.
_ARCH_TO_FAMILY: dict[str, str] = {
    # Llama family (includes Llama 2, 3, 3.1, 3.2, 3.3, Code Llama, etc.)
    "llama": "llama",
    # Qwen family
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen2_moe": "qwen",
    "qwen2moe": "qwen",
    "qwen3": "qwen",
    "qwen3moe": "qwen",
    # Mistral / Mixtral
    "mistral": "mistral",
    "mixtral": "mistral",
    # Gemma
    "gemma": "gemma",
    "gemma2": "gemma",
    "gemma3": "gemma",
    # Phi
    "phi2": "phi",
    "phi3": "phi",
    "phi4": "phi",
    # Starcoder
    "starcoder": "starcoder",
    "starcoder2": "starcoder",
    # Command-R
    "command_r": "command-r",
    # DeepSeek
    "deepseek": "deepseek",
    "deepseek2": "deepseek",
    # GPT-NeoX / Pythia
    "gpt_neox": "gpt_neox",
    # InternLM
    "internlm": "internlm",
    "internlm2": "internlm",
    # ChatGLM
    "chatglm": "chatglm",
    "glm4": "chatglm",
    # MiniCPM
    "minicpm": "minicpm",
    # Falcon
    "falcon": "falcon",
}


def _normalize_arch(raw: str) -> str:
    """Normalize an architecture string for lookup."""
    return raw.lower().strip().replace("-", "_")


def arch_to_family(arch: str) -> str:
    """Map an architecture string to its canonical family name.

    Returns the architecture itself (lowercased) if no mapping exists,
    so unknown architectures are treated as their own family.
    """
    normalized = _normalize_arch(arch)
    return _ARCH_TO_FAMILY.get(normalized, normalized)


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass
class ModelFamily:
    """Detected model family information."""

    arch: str  # raw architecture string (e.g. "llama", "qwen2")
    family: str  # canonical family (e.g. "llama", "qwen")
    model_name: str  # display name from config
    source: str  # where we got the info: "ollama", "llamacpp", "gguf"


@dataclass
class FamilyCheckResult:
    """Result of a draft/target family compatibility check."""

    compatible: bool
    draft: ModelFamily | None
    target: ModelFamily | None
    message: str


# ---------------------------------------------------------------------------
# Detection: Ollama
# ---------------------------------------------------------------------------


# Keys in Ollama's model_info that indicate architecture
_OLLAMA_ARCH_KEY = "general.architecture"


async def detect_ollama_family(
    url: str, model_name: str, timeout: float = 10.0
) -> ModelFamily | None:
    """Detect model family via Ollama's /api/show endpoint."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{url}/api/show",
                json={"name": model_name},
            )
        if resp.status_code != 200:
            logger.debug(
                "Ollama /api/show returned %d for %s", resp.status_code, model_name
            )
            return None

        data = resp.json()
        model_info = data.get("model_info", {})

        # Primary: general.architecture key
        arch = model_info.get(_OLLAMA_ARCH_KEY)
        if arch:
            return ModelFamily(
                arch=arch,
                family=arch_to_family(arch),
                model_name=model_name,
                source="ollama",
            )

        # Fallback: scan model_info keys for "{arch}." prefixed entries
        for key in model_info:
            if "." in key and key != _OLLAMA_ARCH_KEY:
                prefix = key.split(".")[0]
                if prefix in _ARCH_TO_FAMILY:
                    return ModelFamily(
                        arch=prefix,
                        family=arch_to_family(prefix),
                        model_name=model_name,
                        source="ollama",
                    )

    except Exception as exc:
        logger.debug("Failed to detect family via Ollama for %s: %s", model_name, exc)

    return None


# ---------------------------------------------------------------------------
# Detection: llama-server
# ---------------------------------------------------------------------------


async def detect_llamacpp_family(
    url: str, model_name: str, timeout: float = 10.0
) -> ModelFamily | None:
    """Detect model family via llama-server's /props or /v1/models endpoint.

    llama-server doesn't expose architecture directly, but /props may return
    the model path which we can parse for family hints, or the loaded model
    info via /v1/models.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Try /props first (llama.cpp specific)
            try:
                resp = await client.get(f"{url}/props")
                if resp.status_code == 200:
                    data = resp.json()
                    # Some builds include model metadata
                    arch = data.get("general.architecture")
                    if arch:
                        return ModelFamily(
                            arch=arch,
                            family=arch_to_family(arch),
                            model_name=model_name,
                            source="llamacpp",
                        )
            except Exception:
                pass

    except Exception as exc:
        logger.debug(
            "Failed to detect family via llama-server for %s: %s", model_name, exc
        )

    return None


# ---------------------------------------------------------------------------
# Detection: GGUF file
# ---------------------------------------------------------------------------


def detect_gguf_family(path: str, model_name: str = "") -> ModelFamily | None:
    """Detect model family from a local GGUF file's metadata."""
    try:
        from .gguf_reader import read_header
    except ImportError:
        logger.debug("GGUF reader not available")
        return None

    try:
        header = read_header(path)
        arch = header.metadata.get("general.architecture")
        if arch:
            return ModelFamily(
                arch=str(arch),
                family=arch_to_family(str(arch)),
                model_name=model_name or path,
                source="gguf",
            )
    except Exception as exc:
        logger.debug("Failed to read GGUF family from %s: %s", path, exc)

    return None


# ---------------------------------------------------------------------------
# Unified detection
# ---------------------------------------------------------------------------


async def detect_family(
    url: str, model_name: str, backend: str
) -> ModelFamily | None:
    """Detect model family using the appropriate method for the backend."""
    if backend == "ollama":
        return await detect_ollama_family(url, model_name)
    elif backend == "llamacpp":
        return await detect_llamacpp_family(url, model_name)
    return None


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

# Known compatible cross-family pairings (if any exist in the future).
# Currently empty — only same-family pairs work for speculative decoding.
_COMPATIBLE_CROSS_FAMILY: set[frozenset[str]] = set()


def check_compatibility(
    draft: ModelFamily | None,
    target: ModelFamily | None,
) -> FamilyCheckResult:
    """Check if draft and target model families are compatible.

    Returns a FamilyCheckResult with compatibility status and a message.
    """
    if draft is None and target is None:
        return FamilyCheckResult(
            compatible=True,
            draft=None,
            target=None,
            message="Could not detect architecture for draft or target model. "
            "Skipping family compatibility check.",
        )

    if draft is None:
        return FamilyCheckResult(
            compatible=True,
            draft=None,
            target=target,
            message=f"Could not detect draft model architecture. "
            f"Target is {target.family} ({target.arch}). "
            f"Cannot verify compatibility.",
        )

    if target is None:
        return FamilyCheckResult(
            compatible=True,
            draft=draft,
            target=None,
            message=f"Could not detect target model architecture. "
            f"Draft is {draft.family} ({draft.arch}). "
            f"Cannot verify compatibility.",
        )

    if draft.family == target.family:
        return FamilyCheckResult(
            compatible=True,
            draft=draft,
            target=target,
            message=f"Draft ({draft.model_name}) and target ({target.model_name}) "
            f"are both {draft.family} architecture — compatible.",
        )

    # Check known cross-family exceptions
    pair = frozenset([draft.family, target.family])
    if pair in _COMPATIBLE_CROSS_FAMILY:
        return FamilyCheckResult(
            compatible=True,
            draft=draft,
            target=target,
            message=f"Draft ({draft.family}) and target ({target.family}) "
            f"are a known compatible cross-family pairing.",
        )

    return FamilyCheckResult(
        compatible=False,
        draft=draft,
        target=target,
        message=(
            f"INCOMPATIBLE model families: "
            f"draft '{draft.model_name}' is {draft.family} ({draft.arch}) but "
            f"target '{target.model_name}' is {target.family} ({target.arch}). "
            f"Speculative decoding requires draft and target to share the same "
            f"architecture family. Mismatched families cause <5% acceptance rates "
            f"(10x slower than no speculation). "
            f"Use a {target.family}-family model as your draft."
        ),
    )


async def check_proxy_families(
    draft_url: str,
    draft_model: str,
    draft_backend: str,
    target_url: str,
    target_model: str,
    target_backend: str,
) -> FamilyCheckResult:
    """Detect families for both endpoints and check compatibility.

    This is the main entry point for proxy startup and doctor checks.
    """
    draft_family = await detect_family(draft_url, draft_model, draft_backend)
    target_family = await detect_family(target_url, target_model, target_backend)
    return check_compatibility(draft_family, target_family)
