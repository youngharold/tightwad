"""Model hub: resolve, download, and validate GGUF models from HuggingFace."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger("tightwad.model_hub")

DEFAULT_MODEL_DIR = Path.home() / ".tightwad" / "models"

# Curated model registry: short spec -> (hf_repo, filename)
# Format: "family:size-quant" -> ("org/repo", "filename.gguf")
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "llama3.3:70b-q4_k_m": (
        "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
    ),
    "llama3.3:70b-q4_k_xl": (
        "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "Llama-3.3-70B-Instruct-Q4_K_XL.gguf",
    ),
    "llama3.1:8b-q4_k_m": (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
    "llama3.1:8b-q8_0": (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    ),
    "qwen3:32b-q4_k_m": (
        "Qwen/Qwen3-32B-GGUF",
        "qwen3-32b-q4_k_m.gguf",
    ),
    "qwen3:8b-q4_k_m": (
        "Qwen/Qwen3-8B-GGUF",
        "qwen3-8b-q4_k_m.gguf",
    ),
    "qwen3:1.7b-q8_0": (
        "Qwen/Qwen3-1.7B-GGUF",
        "qwen3-1.7b-q8_0.gguf",
    ),
    "deepseek-r1:70b-q4_k_m": (
        "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
    ),
    "deepseek-r1:8b-q4_k_m": (
        "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
    ),
    "devstral:24b-q4_k_m": (
        "bartowski/Devstral-Small-2505-GGUF",
        "Devstral-Small-2505-Q4_K_M.gguf",
    ),
}


@dataclass
class ResolvedModel:
    """A resolved model ready for download."""
    hf_url: str
    filename: str
    spec: str


def resolve_model(spec: str) -> ResolvedModel:
    """Resolve a model spec to a download URL.

    Accepts:
    - Registry short names: "llama3.3:70b-q4_k_m"
    - Direct HuggingFace URLs: "https://huggingface.co/..."
    - HF repo/file format: "bartowski/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
    """
    # Direct URL
    if spec.startswith("http://") or spec.startswith("https://"):
        filename = spec.rsplit("/", 1)[-1]
        return ResolvedModel(hf_url=spec, filename=filename, spec=spec)

    # Registry lookup
    if spec.lower() in MODEL_REGISTRY:
        repo, filename = MODEL_REGISTRY[spec.lower()]
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        return ResolvedModel(hf_url=url, filename=filename, spec=spec)

    # HF repo/file format: "org/repo/filename.gguf"
    parts = spec.split("/")
    if len(parts) >= 3 and parts[-1].endswith(".gguf"):
        repo = "/".join(parts[:-1])
        filename = parts[-1]
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        return ResolvedModel(hf_url=url, filename=filename, spec=spec)

    available = ", ".join(sorted(MODEL_REGISTRY.keys()))
    raise ValueError(
        f"Unknown model spec: {spec!r}\n\n"
        f"Available models:\n  {available}\n\n"
        "Or provide a direct URL or HF repo path:\n"
        "  tightwad pull https://huggingface.co/.../model.gguf\n"
        "  tightwad pull org/repo/filename.gguf"
    )


def list_models() -> list[tuple[str, str, str]]:
    """List all available models in the registry.

    Returns list of (spec, repo, filename).
    """
    return [
        (spec, repo, filename)
        for spec, (repo, filename) in sorted(MODEL_REGISTRY.items())
    ]


def download_model(
    url: str,
    dest_dir: Path | None = None,
    filename: str | None = None,
    resume: bool = True,
    progress_callback=None,
) -> Path:
    """Download a model file with resume support.

    Parameters
    ----------
    url:
        Direct download URL.
    dest_dir:
        Download directory (default: ~/.tightwad/models/).
    filename:
        Override filename (default: from URL).
    resume:
        If True, resume partial downloads using Range header.
    progress_callback:
        Optional callable(bytes_downloaded, total_bytes) for progress updates.

    Returns
    -------
    Path to the downloaded file.
    """
    if dest_dir is None:
        dest_dir = DEFAULT_MODEL_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.rsplit("/", 1)[-1]
    dest_path = dest_dir / filename

    # Check for existing partial download
    existing_size = 0
    if resume and dest_path.exists():
        existing_size = dest_path.stat().st_size

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        logger.info("Resuming download from %d bytes", existing_size)

    with httpx.stream("GET", url, headers=headers, timeout=30.0, follow_redirects=True) as response:
        if response.status_code == 416:
            # Range not satisfiable — file is complete
            logger.info("File already complete: %s", dest_path)
            return dest_path

        response.raise_for_status()

        # Get total size
        total_size = existing_size
        content_range = response.headers.get("content-range")
        content_length = response.headers.get("content-length")
        if content_range and "/" in content_range:
            total_size = int(content_range.rsplit("/", 1)[1])
        elif content_length:
            total_size = existing_size + int(content_length)

        mode = "ab" if existing_size > 0 and response.status_code == 206 else "wb"
        bytes_downloaded = existing_size if mode == "ab" else 0

        with open(dest_path, mode) as f:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                bytes_downloaded += len(chunk)
                if progress_callback:
                    progress_callback(bytes_downloaded, total_size)

    return dest_path


def validate_download(path: Path) -> bool:
    """Validate a downloaded file is a valid GGUF.

    Returns True if the file has a valid GGUF magic number.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == b"GGUF"
    except (OSError, IOError):
        return False
