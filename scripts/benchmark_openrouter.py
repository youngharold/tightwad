#!/usr/bin/env python3
"""Benchmark speculative decoding: local Llama 3.1 8B draft vs OpenRouter 405B target.

Multi-round spec decoding with whitespace-normalized text-match verification.
Same methodology as benchmark_proxy.py.

Usage:
    # Start local draft server first:
    llama-server -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
        --port 8081 --host 127.0.0.1 -ngl 999 --ctx-size 4096

    # Then run benchmark:
    OPENROUTER_API_KEY=sk-or-... python scripts/benchmark_openrouter.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import httpx

# --- Config ---

DRAFT_URL = os.environ.get("DRAFT_URL", "http://127.0.0.1:8081")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TARGET_MODEL = os.environ.get("TARGET_MODEL", "meta-llama/llama-3.1-405b-instruct")
DRAFT_MODEL = "Meta-Llama-3.1-8B-Instruct"
MAX_TOKENS = 256
MAX_DRAFT_TOKENS = 32
MAX_ROUNDS = 30
DEBUG = os.environ.get("BENCH_DEBUG", "")

PROMPTS = [
    {"name": "reasoning",   "messages": [{"role": "user", "content": "What is 17 * 24? Show your work step by step."}]},
    {"name": "reasoning",   "messages": [{"role": "user", "content": "If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire trip?"}]},
    {"name": "code",        "messages": [{"role": "user", "content": "Write a Python function that checks if a number is prime. Include a docstring."}]},
    {"name": "code",        "messages": [{"role": "user", "content": "Write a Python function to reverse a linked list iteratively. Include type hints."}]},
    {"name": "factual",     "messages": [{"role": "user", "content": "What is the capital of France and what is its population?"}]},
    {"name": "factual",     "messages": [{"role": "user", "content": "Explain the three laws of thermodynamics in simple terms."}]},
    {"name": "factual",     "messages": [{"role": "user", "content": "What are the key differences between TCP and UDP?"}]},
    {"name": "list",        "messages": [{"role": "user", "content": "List the 10 largest countries by area."}]},
    {"name": "list",        "messages": [{"role": "user", "content": "Name 5 programming languages and one strength of each."}]},
    {"name": "creative",    "messages": [{"role": "user", "content": "Write a haiku about artificial intelligence."}]},
    {"name": "creative",    "messages": [{"role": "user", "content": "Write a one-paragraph story about a robot learning to cook."}]},
    {"name": "reasoning",   "messages": [{"role": "user", "content": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?"}]},
]


def normalize(s: str) -> str:
    """Normalize whitespace for fair text-match comparison."""
    return re.sub(r'\s+', ' ', s).strip()


def apply_llama_template(messages: list[dict], partial_response: str = "") -> str:
    """Convert chat messages to Llama 3.1 instruct format with optional partial response."""
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        parts.append(f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    if partial_response:
        parts.append(partial_response)
    return "".join(parts)


def draft_generate(client: httpx.Client, prompt: str) -> str:
    """Generate from local llama-server draft model."""
    resp = client.post(
        f"{DRAFT_URL}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": MAX_DRAFT_TOKENS,
            "temperature": 0.0,
            "stream": False,
            "stop": ["<|eot_id|>", "<|end_of_text|>"],
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0].get("text", "")


def target_generate(client: httpx.Client, messages: list[dict], partial: str = "") -> str:
    """Generate from OpenRouter target, continuing from partial response."""
    msgs = list(messages)
    if partial:
        msgs.append({"role": "assistant", "content": partial})
        msgs.append({"role": "user", "content": "Continue from exactly where you left off. Do not repeat anything."})

    resp = client.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/youngharold/tightwad",
            "X-Title": "Tightwad Benchmark",
        },
        json={
            "model": TARGET_MODEL,
            "messages": msgs,
            "max_tokens": MAX_DRAFT_TOKENS,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def baseline_generate(client: httpx.Client, messages: list[dict]) -> tuple[str, float]:
    """Full generation from target only (baseline). Returns (text, elapsed)."""
    t0 = time.monotonic()
    resp = client.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/youngharold/tightwad",
            "X-Title": "Tightwad Benchmark",
        },
        json={
            "model": TARGET_MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=120.0,
    )
    elapsed = time.monotonic() - t0
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return text, elapsed


def spec_round(client: httpx.Client, messages: list[dict], generated: str) -> dict:
    """One speculative decoding round. Returns round stats."""
    # Draft phase: local llama-server
    prompt = apply_llama_template(messages, generated)
    draft_text = draft_generate(client, prompt)

    if not draft_text:
        return {"draft_text": "", "target_text": "", "match": 0, "drafted": 0, "done": True}

    draft_done = False
    for stop in ["<|eot_id|>", "<|end_of_text|>"]:
        if stop in draft_text:
            draft_text = draft_text[:draft_text.index(stop)]
            draft_done = True

    if not draft_text:
        return {"draft_text": "", "target_text": "", "match": 0, "drafted": 0, "done": True}

    # Target phase: OpenRouter (same context)
    target_text = target_generate(client, messages, generated)

    if not target_text:
        return {"draft_text": draft_text, "target_text": "", "match": 0, "drafted": len(normalize(draft_text)), "done": True}

    # Normalize and compare
    d_norm = normalize(draft_text)
    t_norm = normalize(target_text)

    match_len = 0
    for i in range(min(len(d_norm), len(t_norm))):
        if d_norm[i] == t_norm[i]:
            match_len = i + 1
        else:
            break

    return {
        "draft_text": draft_text,
        "target_text": target_text,
        "match": match_len,
        "drafted": len(d_norm),
        "done": draft_done or not target_text,
    }


def check_draft_server() -> bool:
    try:
        return httpx.get(f"{DRAFT_URL}/health", timeout=5.0).status_code == 200
    except Exception:
        return False


def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Usage: OPENROUTER_API_KEY=sk-or-... python scripts/benchmark_openrouter.py")
        sys.exit(1)

    if not check_draft_server():
        print(f"ERROR: Draft server not running at {DRAFT_URL}")
        print(f"Start with: llama-server -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port 8081 -ngl 999 --jinja")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"Tightwad Speculative Decoding Benchmark (Multi-Round)")
    print(f"{'='*70}")
    print(f"Draft:  {DRAFT_MODEL} (local, {DRAFT_URL})")
    print(f"Target: {TARGET_MODEL} (OpenRouter)")
    print(f"Max tokens: {MAX_TOKENS} | Draft tokens/round: {MAX_DRAFT_TOKENS}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"{'='*70}\n")

    client = httpx.Client()
    results = []

    for i, prompt_info in enumerate(PROMPTS):
        category = prompt_info["name"]
        messages = prompt_info["messages"]
        user_msg = messages[-1]["content"][:60]

        print(f"[{i+1}/{len(PROMPTS)}] {category}: {user_msg}...")

        # Baseline
        print(f"  baseline...", end="", flush=True)
        try:
            base_text, base_time = baseline_generate(client, messages)
            base_words = len(base_text.split())
            print(f" {base_time:.1f}s ({base_words} words)", flush=True)
        except Exception as e:
            print(f" FAILED: {e}")
            continue

        # Multi-round speculative decoding
        print(f"  spec...", end="", flush=True)
        generated = ""
        total_rounds = 0
        total_drafted = 0
        total_accepted = 0
        spec_start = time.monotonic()

        for _ in range(MAX_ROUNDS):
            try:
                rnd = spec_round(client, messages, generated)
            except Exception as e:
                print(f" round error: {e}", flush=True)
                break

            total_rounds += 1
            total_drafted += rnd["drafted"]
            total_accepted += rnd["match"]

            if DEBUG:
                print(f"\n    [round {total_rounds}] match={rnd['match']}/{rnd['drafted']} "
                      f"draft={rnd['draft_text'][:50]!r} target={rnd['target_text'][:50]!r}")

            # Accept target output
            generated += rnd["target_text"]
            word_count = len(generated.split())

            if rnd["done"] or word_count >= MAX_TOKENS:
                break

        spec_time = time.monotonic() - spec_start
        acceptance = total_accepted / total_drafted if total_drafted > 0 else 0
        spec_words = len(generated.split())

        print(f" {spec_time:.1f}s ({acceptance*100:.0f}% accept, {total_rounds} rounds, {spec_words} words)")

        results.append({
            "category": category,
            "prompt": user_msg,
            "baseline_time": base_time,
            "baseline_words": base_words,
            "spec_time": spec_time,
            "spec_words": spec_words,
            "rounds": total_rounds,
            "drafted_chars": total_drafted,
            "accepted_chars": total_accepted,
            "acceptance_pct": round(acceptance * 100, 1),
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY — Multi-Round Speculative Decoding")
    print(f"{'='*70}")
    print(f"Draft: {DRAFT_MODEL} | Target: {TARGET_MODEL}")
    print(f"Whitespace-normalized text-match verification")
    print(f"{'='*70}\n")

    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    print(f"{'Category':<15} {'Acceptance':>10} {'Rounds':>8} {'Avg Spec(s)':>12} {'Avg Base(s)':>12}")
    print(f"{'-'*15} {'-'*10} {'-'*8} {'-'*12} {'-'*12}")

    total_drafted = 0
    total_accepted = 0

    for cat, rs in sorted(categories.items()):
        avg_acc = sum(r["acceptance_pct"] for r in rs) / len(rs)
        avg_spec = sum(r["spec_time"] for r in rs) / len(rs)
        avg_base = sum(r["baseline_time"] for r in rs) / len(rs)
        n = len(rs)

        for r in rs:
            total_drafted += r["drafted_chars"]
            total_accepted += r["accepted_chars"]

        print(f"{cat:<15} {avg_acc:>9.1f}% {n:>8} {avg_spec:>11.1f}s {avg_base:>11.1f}s")

    overall = total_accepted / total_drafted * 100 if total_drafted > 0 else 0
    avg_spec_all = sum(r["spec_time"] for r in results) / len(results)
    avg_base_all = sum(r["baseline_time"] for r in results) / len(results)

    print(f"{'-'*15} {'-'*10} {'-'*8} {'-'*12} {'-'*12}")
    print(f"{'OVERALL':<15} {overall:>9.1f}% {len(results):>8} {avg_spec_all:>11.1f}s {avg_base_all:>11.1f}s")

    # Save
    os.makedirs("benchmarks", exist_ok=True)
    outfile = "benchmarks/benchmark_llama_openrouter.json"
    with open(outfile, "w") as f:
        json.dump({
            "config": {
                "draft_model": DRAFT_MODEL,
                "target_model": TARGET_MODEL,
                "max_tokens": MAX_TOKENS,
                "max_draft_tokens": MAX_DRAFT_TOKENS,
                "method": "multi-round spec decoding, whitespace-normalized text-match",
            },
            "results": results,
            "summary": {
                "overall_acceptance_pct": round(overall, 1),
                "total_prompts": len(results),
                "categories": {
                    cat: {
                        "acceptance_pct": round(sum(r["acceptance_pct"] for r in rs) / len(rs), 1),
                        "rounds": len(rs),
                    }
                    for cat, rs in categories.items()
                },
            },
        }, f, indent=2)
    print(f"\nRaw data saved to {outfile}")


if __name__ == "__main__":
    main()
