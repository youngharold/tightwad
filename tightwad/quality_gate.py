"""Quality gate: CPU fleet generates, GPU verifies.

A fleet of cheap agents (CPUs, small GPUs) generate full responses using
small models.  A single powerful GPU reviews each response and either
approves it, corrects minor issues, or rejects it entirely (triggering
regeneration on the GPU).

This mode optimizes for **cost**, not latency — 60-80% of responses pass
verification unchanged, meaning only 20-40% of tokens need GPU compute.

The proxy is OpenAI-compatible: clients see a single endpoint and get
verified responses transparently.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger("tightwad.quality_gate")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Verdict(Enum):
    APPROVE = "approve"
    CORRECT = "correct"
    REJECT = "reject"
    ERROR = "error"  # verification itself failed


@dataclass
class VerificationResult:
    verdict: Verdict
    original_response: str
    corrected_response: str | None = None  # set when verdict is CORRECT
    verification_ms: float = 0.0
    agent_url: str = ""


@dataclass
class AgentEndpoint:
    url: str
    model_name: str
    backend: str = "ollama"


@dataclass
class QualityGateConfig:
    verifier_url: str
    verifier_model: str
    verifier_backend: str = "llamacpp"
    agents: list[AgentEndpoint] = field(default_factory=list)
    routing: str = "round_robin"  # round_robin, random
    verification_prompt: str = ""
    max_retries: int = 1
    cache_identical: bool = True
    host: str = "0.0.0.0"
    port: int = 8088
    auth_token: str | None = None
    #: When the verifier returns an unparseable verdict, default to REJECT
    #: (regenerate on the strong model). Flip to True only if availability
    #: matters more than correctness for the workload.
    fail_open: bool = False


@dataclass
class GateStats:
    total_requests: int = 0
    approved: int = 0
    corrected: int = 0
    rejected: int = 0
    errors: int = 0
    total_agent_ms: float = 0.0
    total_verify_ms: float = 0.0
    cache_hits: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def approve_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.approved / self.total_requests

    @property
    def gpu_usage_rate(self) -> float:
        """Fraction of requests that needed GPU compute (correct + reject)."""
        if self.total_requests == 0:
            return 0.0
        return (self.corrected + self.rejected) / self.total_requests

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.start_time


# ---------------------------------------------------------------------------
# Default verification prompt
# ---------------------------------------------------------------------------

DEFAULT_VERIFICATION_PROMPT = """Review the following AI assistant response for accuracy, completeness, and quality.

Original user request:
{prompt}

AI response to review:
{response}

Respond with exactly one of these three formats:
APPROVE
CORRECT: <your corrected version of the full response>
REJECT"""


# ---------------------------------------------------------------------------
# Verdict parser
# ---------------------------------------------------------------------------


_DIRECTIVE_RE = re.compile(
    r"""^\s*(?:VERDICT\s*:\s*)?  # optional VERDICT: prefix
        (APPROVE|REJECT|CORRECT)  # verdict keyword
        (?:\s*[:\-]?\s*(.*))?     # optional correction body for CORRECT
        \s*$""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def parse_verdict(text: str, fail_open: bool = False) -> tuple[Verdict, str | None]:
    """Parse the verifier's response into a verdict and optional correction.

    Strict directive matching: the verdict must come from the first non-empty
    line, optionally prefixed with ``VERDICT:``. We do this so prose like
    "I cannot approve this" doesn't get parsed as APPROVE just because it
    contains the word.

    Returns (Verdict, corrected_text_or_None).

    When the verifier output cannot be parsed, default to REJECT (regenerate
    on the strong model). Pass ``fail_open=True`` to keep the legacy
    availability-first behavior of returning APPROVE on parse failure.
    """
    text = text.strip()
    if not text:
        return _ambiguous_default(fail_open, "empty verifier output")

    # Use the first non-empty line as the directive line; everything after it
    # is treated as the optional correction body (only used when verdict==CORRECT).
    lines = text.splitlines()
    directive_line = ""
    body = ""
    for i, line in enumerate(lines):
        if line.strip():
            directive_line = line
            body = "\n".join(lines[i + 1:]).strip()
            break

    m = _DIRECTIVE_RE.match(directive_line)
    if not m:
        return _ambiguous_default(fail_open, f"unparseable directive line: {directive_line!r}")

    verdict_word = m.group(1).upper()
    inline_body = (m.group(2) or "").strip()

    if verdict_word == "APPROVE":
        return Verdict.APPROVE, None

    if verdict_word == "REJECT":
        return Verdict.REJECT, None

    # CORRECT: combine the inline correction with any trailing body lines.
    if inline_body and body:
        correction = inline_body + "\n" + body
    else:
        correction = inline_body or body
    if correction:
        return Verdict.CORRECT, correction
    # CORRECT with no payload is approval-with-no-edit.
    return Verdict.APPROVE, None


def _ambiguous_default(fail_open: bool, reason: str) -> tuple[Verdict, str | None]:
    if fail_open:
        logger.warning(
            "%s; fail_open=True so defaulting to APPROVE", reason,
        )
        return Verdict.APPROVE, None
    logger.warning(
        "%s; defaulting to REJECT (set fail_open=True to invert)", reason,
    )
    return Verdict.REJECT, None


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


class ResponseCache:
    """LRU cache for verified prompt+response pairs."""

    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, Verdict] = OrderedDict()
        self._max_size = max_size

    def _key(self, prompt: str, response: str) -> str:
        h = hashlib.sha256(f"{prompt}\x00{response}".encode()).hexdigest()[:16]
        return h

    def get(self, prompt: str, response: str) -> Verdict | None:
        key = self._key(prompt, response)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, prompt: str, response: str, verdict: Verdict) -> None:
        key = self._key(prompt, response)
        self._cache[key] = verdict
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Quality Gate Proxy
# ---------------------------------------------------------------------------


class QualityGateProxy:
    """Orchestrates agent generation → GPU verification → response delivery."""

    def __init__(self, config: QualityGateConfig):
        self.config = config
        self.stats = GateStats()
        self._cache = ResponseCache() if config.cache_identical else None
        self._agent_idx = 0  # round-robin counter
        self._verification_prompt = config.verification_prompt or DEFAULT_VERIFICATION_PROMPT

        # Persistent HTTP clients
        self.verifier_client = httpx.AsyncClient(
            base_url=config.verifier_url, timeout=120.0, http2=True,
        )
        self.agent_clients: list[tuple[AgentEndpoint, httpx.AsyncClient]] = []
        for agent in config.agents:
            client = httpx.AsyncClient(
                base_url=agent.url, timeout=60.0, http2=True,
            )
            self.agent_clients.append((agent, client))

    async def close(self):
        await self.verifier_client.aclose()
        for _, client in self.agent_clients:
            await client.aclose()

    def _pick_agent(self) -> tuple[AgentEndpoint, httpx.AsyncClient]:
        """Select next agent via configured routing strategy."""
        if not self.agent_clients:
            raise RuntimeError("No agents configured")
        if self.config.routing == "random":
            import random
            return random.choice(self.agent_clients)
        # round_robin (default)
        idx = self._agent_idx % len(self.agent_clients)
        self._agent_idx += 1
        return self.agent_clients[idx]

    async def _generate_from_agent(
        self, agent: AgentEndpoint, client: httpx.AsyncClient,
        prompt: str, max_tokens: int, temperature: float,
    ) -> str:
        """Get a full response from an agent's small model."""
        if agent.backend == "ollama":
            body = {
                "model": agent.model_name,
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            }
            resp = await client.post(f"{agent.url}/api/generate", json=body)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "") or data.get("thinking", "")
        else:
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            resp = await client.post("/v1/completions", json=body)
            resp.raise_for_status()
            return resp.json()["choices"][0].get("text", "")

    async def _verify(self, prompt: str, response: str) -> VerificationResult:
        """Send the response to the GPU verifier for review."""
        verification_prompt = self._verification_prompt.format(
            prompt=prompt, response=response,
        )

        t0 = time.monotonic()
        if self.config.verifier_backend == "ollama":
            body = {
                "model": self.config.verifier_model,
                "prompt": verification_prompt,
                "raw": True,
                "stream": False,
                "options": {"num_predict": 4096, "temperature": 0.0},
            }
            resp = await self.verifier_client.post(
                f"{self.config.verifier_url}/api/generate", json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            verifier_text = data.get("response", "")
        else:
            body = {
                "prompt": verification_prompt,
                "max_tokens": 4096,
                "temperature": 0.0,
                "stream": False,
            }
            resp = await self.verifier_client.post("/v1/completions", json=body)
            resp.raise_for_status()
            verifier_text = resp.json()["choices"][0].get("text", "")

        verify_ms = (time.monotonic() - t0) * 1000
        verdict, corrected = parse_verdict(verifier_text, fail_open=self.config.fail_open)

        return VerificationResult(
            verdict=verdict,
            original_response=response,
            corrected_response=corrected,
            verification_ms=verify_ms,
        )

    async def _generate_from_verifier(
        self, prompt: str, max_tokens: int, temperature: float,
    ) -> str:
        """Generate directly on the GPU (used after REJECT)."""
        if self.config.verifier_backend == "ollama":
            body = {
                "model": self.config.verifier_model,
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            }
            resp = await self.verifier_client.post(
                f"{self.config.verifier_url}/api/generate", json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "") or data.get("thinking", "")
        else:
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            resp = await self.verifier_client.post("/v1/completions", json=body)
            resp.raise_for_status()
            return resp.json()["choices"][0].get("text", "")

    async def handle_request(
        self, prompt: str, max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Full quality gate pipeline: agent → verify → return."""
        self.stats.total_requests += 1

        # 1. Generate from a CPU agent
        agent, client = self._pick_agent()
        t0 = time.monotonic()
        agent_response = await self._generate_from_agent(
            agent, client, prompt, max_tokens, temperature,
        )
        agent_ms = (time.monotonic() - t0) * 1000
        self.stats.total_agent_ms += agent_ms

        if not agent_response:
            # Agent returned nothing — fall through to verifier
            self.stats.rejected += 1
            return await self._generate_from_verifier(prompt, max_tokens, temperature)

        # 2. Check cache
        if self._cache:
            cached = self._cache.get(prompt, agent_response)
            if cached == Verdict.APPROVE:
                self.stats.cache_hits += 1
                self.stats.approved += 1
                return agent_response

        # 3. Verify with GPU
        result = await self._verify(prompt, agent_response)
        self.stats.total_verify_ms += result.verification_ms

        if self._cache and result.verdict in (Verdict.APPROVE, Verdict.CORRECT):
            self._cache.put(prompt, agent_response, result.verdict)

        if result.verdict == Verdict.APPROVE:
            self.stats.approved += 1
            return agent_response

        if result.verdict == Verdict.CORRECT and result.corrected_response:
            self.stats.corrected += 1
            return result.corrected_response

        if result.verdict == Verdict.REJECT:
            self.stats.rejected += 1
            # Retry on agent once before falling back to GPU
            for _ in range(self.config.max_retries):
                agent_response2 = await self._generate_from_agent(
                    agent, client, prompt, max_tokens, temperature,
                )
                if agent_response2:
                    result2 = await self._verify(prompt, agent_response2)
                    self.stats.total_verify_ms += result2.verification_ms
                    if result2.verdict == Verdict.APPROVE:
                        self.stats.approved += 1
                        return agent_response2
                    if result2.verdict == Verdict.CORRECT and result2.corrected_response:
                        self.stats.corrected += 1
                        return result2.corrected_response

            # All retries exhausted — generate on GPU directly
            return await self._generate_from_verifier(prompt, max_tokens, temperature)

        # Error or unknown verdict — fail open with agent response
        self.stats.errors += 1
        return agent_response
