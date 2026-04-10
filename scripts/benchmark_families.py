#!/usr/bin/env python3
"""Benchmark speculative decoding across model families via OpenRouter.

Multi-round spec decoding with whitespace-normalized text-match verification.
Runs multiple draft→target configs sequentially, each with a local llama-server
draft and an OpenRouter cloud target.

Usage:
    # Start the draft server for the first config you want to test, then run:
    OPENROUTER_API_KEY=sk-or-... python scripts/benchmark_families.py

    # Or specify a single config:
    OPENROUTER_API_KEY=sk-or-... python scripts/benchmark_families.py --config qwen3-235b

    # List available configs:
    python scripts/benchmark_families.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import httpx

# --- Shared Config ---

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
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


# --- Model Family Configs ---
# Each config: (name, draft_model_gguf, draft_url, target_model, template_fn, stop_tokens)

def llama_template(messages: list[dict], partial: str = "") -> str:
    """Llama 3.1 instruct chat template."""
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        parts.append(f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    if partial:
        parts.append(partial)
    return "".join(parts)


def qwen3_template(messages: list[dict], partial: str = "") -> str:
    """Qwen3 chat template (ChatML-style) with /no_think to suppress reasoning."""
    parts = ["<|im_start|>system\nYou are a helpful assistant. /no_think<|im_end|>"]
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    if partial:
        parts.append(partial)
    return "\n".join(parts)


CONFIGS = {
    # --- Qwen3 family ---
    "qwen3-1.7b-235b": {
        "name": "Qwen3-1.7B → Qwen3-235B-A22B (OpenRouter)",
        "draft_model": "Qwen3-1.7B",
        "draft_gguf": "~/models/Qwen3-1.7B-Q8_0.gguf",
        "draft_url": "http://127.0.0.1:8081",
        "draft_backend": "llamacpp",
        "target_model": "qwen/qwen3-235b-a22b",
        "template": qwen3_template,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "system_message": "You are a helpful assistant. /no_think",
    },
    "qwen3-1.7b-397b": {
        "name": "Qwen3-1.7B → Qwen3.5-397B-A17B (OpenRouter)",
        "draft_model": "Qwen3-1.7B",
        "draft_gguf": "~/models/Qwen3-1.7B-Q8_0.gguf",
        "draft_url": "http://127.0.0.1:8081",
        "draft_backend": "llamacpp",
        "target_model": "qwen/qwen3.5-397b-a17b",
        "template": qwen3_template,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "system_message": "You are a helpful assistant. /no_think",
    },
    "qwen3-8b-235b": {
        "name": "Qwen3-8B → Qwen3-235B-A22B (OpenRouter)",
        "draft_model": "qwen3:8b",
        "draft_gguf": "(Ollama on 2070)",
        "draft_url": "http://192.168.1.101:11434",
        "draft_backend": "ollama",
        "target_model": "qwen/qwen3-235b-a22b",
        "template": qwen3_template,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "system_message": "You are a helpful assistant. /no_think",
    },
    "qwen3-8b-397b": {
        "name": "Qwen3-8B → Qwen3.5-397B-A17B (OpenRouter)",
        "draft_model": "qwen3:8b",
        "draft_gguf": "(Ollama on 2070)",
        "draft_url": "http://192.168.1.101:11434",
        "draft_backend": "ollama",
        "target_model": "qwen/qwen3.5-397b-a17b",
        "template": qwen3_template,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "system_message": "You are a helpful assistant. /no_think",
    },
    # --- Llama 3.1 family ---
    "llama-70b": {
        "name": "Llama-3.1-8B → Llama-3.1-70B (OpenRouter)",
        "draft_model": "Meta-Llama-3.1-8B-Instruct",
        "draft_gguf": "~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "draft_url": "http://127.0.0.1:8081",
        "draft_backend": "llamacpp",
        "target_model": "meta-llama/llama-3.1-70b-instruct",
        "template": llama_template,
        "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
    },
    "llama-405b": {
        "name": "Llama-3.1-8B → Llama-3.1-405B (OpenRouter)",
        "draft_model": "Meta-Llama-3.1-8B-Instruct",
        "draft_gguf": "~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "draft_url": "http://127.0.0.1:8081",
        "draft_backend": "llamacpp",
        "target_model": "meta-llama/llama-3.1-405b-instruct",
        "template": llama_template,
        "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
    },
    # --- DeepSeek R1 family ---
    # R1 distills are Qwen-architecture but trained to mimic R1 output
    "deepseek-r1-8b": {
        "name": "DeepSeek-R1-Distill-8B → DeepSeek-R1 (OpenRouter)",
        "draft_model": "deepseek-r1:8b",
        "draft_gguf": "(Ollama on M2 Mac)",
        "draft_url": "http://192.168.1.102:11434",
        "draft_backend": "ollama",
        "target_model": "deepseek/deepseek-r1-0528:free",
        "template": qwen3_template,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "system_message": None,  # R1 uses thinking by design
    },
    "deepseek-r1-14b": {
        "name": "DeepSeek-R1-Distill-14B → DeepSeek-R1 (OpenRouter)",
        "draft_model": "deepseek-r1:14b",
        "draft_gguf": "(Ollama on 2070)",
        "draft_url": "http://192.168.1.101:11434",
        "draft_backend": "ollama",
        "target_model": "deepseek/deepseek-r1-0528:free",
        "template": qwen3_template,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "system_message": None,  # R1 uses thinking by design
    },
}


# --- Core Functions ---

def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def strip_thinking(s: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 thinking mode responses."""
    return re.sub(r'<think>.*?</think>\s*', '', s, flags=re.DOTALL).strip()


def draft_generate(client: httpx.Client, cfg: dict, prompt: str) -> str:
    resp = client.post(
        f"{cfg['draft_url']}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": MAX_DRAFT_TOKENS,
            "temperature": 0.0,
            "stream": False,
            "stop": cfg["stop_tokens"],
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0].get("text", "")


def target_generate(client: httpx.Client, cfg: dict, messages: list[dict], partial: str = "") -> str:
    msgs = []
    if cfg.get("system_message"):
        msgs.append({"role": "system", "content": cfg["system_message"]})
    msgs.extend(messages)
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
            "model": cfg["target_model"],
            "messages": msgs,
            "max_tokens": MAX_DRAFT_TOKENS,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return strip_thinking(text)


def baseline_generate(client: httpx.Client, cfg: dict, messages: list[dict]) -> tuple[str, float]:
    t0 = time.monotonic()
    msgs = []
    if cfg.get("system_message"):
        msgs.append({"role": "system", "content": cfg["system_message"]})
    msgs.extend(messages)
    resp = client.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/youngharold/tightwad",
            "X-Title": "Tightwad Benchmark",
        },
        json={
            "model": cfg["target_model"],
            "messages": msgs,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=120.0,
    )
    elapsed = time.monotonic() - t0
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return strip_thinking(text), elapsed


def spec_round(client: httpx.Client, cfg: dict, messages: list[dict], generated: str) -> dict:
    prompt = cfg["template"](messages, generated)
    draft_text = strip_thinking(draft_generate(client, cfg, prompt))

    if not draft_text:
        return {"draft_text": "", "target_text": "", "match": 0, "drafted": 0, "done": True}

    draft_done = False
    for stop in cfg["stop_tokens"]:
        if stop in draft_text:
            draft_text = draft_text[:draft_text.index(stop)]
            draft_done = True

    if not draft_text:
        return {"draft_text": "", "target_text": "", "match": 0, "drafted": 0, "done": True}

    target_text = target_generate(client, cfg, messages, generated)

    if not target_text:
        return {"draft_text": draft_text, "target_text": "", "match": 0, "drafted": len(normalize(draft_text)), "done": True}

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


def check_draft_server(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=5.0).status_code == 200
    except Exception:
        return False


def run_benchmark(cfg: dict) -> list[dict]:
    """Run all prompts for one config. Returns list of result dicts."""
    print(f"\n{'='*70}")
    print(f"{cfg['name']}")
    print(f"{'='*70}")
    print(f"Draft:  {cfg['draft_model']} (local, {cfg['draft_url']})")
    print(f"Target: {cfg['target_model']} (OpenRouter)")
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
            base_text, base_time = baseline_generate(client, cfg, messages)
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
                rnd = spec_round(client, cfg, messages, generated)
            except Exception as e:
                print(f" round error: {e}", flush=True)
                break

            total_rounds += 1
            total_drafted += rnd["drafted"]
            total_accepted += rnd["match"]

            if DEBUG:
                print(f"\n    [round {total_rounds}] match={rnd['match']}/{rnd['drafted']} "
                      f"draft={rnd['draft_text'][:50]!r} target={rnd['target_text'][:50]!r}")

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

    client.close()
    return results


def print_summary(cfg: dict, results: list[dict]):
    """Print and save results for one config."""
    if not results:
        print("  No results to summarize.")
        return

    print(f"\n{'='*70}")
    print(f"RESULTS — {cfg['name']}")
    print(f"{'='*70}")
    print(f"Draft: {cfg['draft_model']} | Target: {cfg['target_model']}")
    print(f"Whitespace-normalized text-match verification")
    print(f"{'='*70}\n")

    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    print(f"{'Category':<15} {'Acceptance':>10} {'Prompts':>8} {'Avg Spec(s)':>12} {'Avg Base(s)':>12}")
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
    safe_name = cfg["target_model"].replace("/", "_")
    outfile = f"benchmarks/benchmark_{safe_name}.json"
    with open(outfile, "w") as f:
        json.dump({
            "config": {
                "draft_model": cfg["draft_model"],
                "target_model": cfg["target_model"],
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
                        "count": len(rs),
                    }
                    for cat, rs in categories.items()
                },
            },
        }, f, indent=2)
    print(f"\nRaw data saved to {outfile}")
    return overall


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding across model families")
    parser.add_argument("--config", "-c", help="Run a specific config (e.g. qwen3-235b, llama-70b)")
    parser.add_argument("--list", "-l", action="store_true", help="List available configs")
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for key, cfg in CONFIGS.items():
            print(f"  {key:<20} {cfg['name']}")
            print(f"  {'':20} Draft GGUF: {cfg['draft_gguf']}")
        sys.exit(0)

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Usage: OPENROUTER_API_KEY=sk-or-... python scripts/benchmark_families.py")
        sys.exit(1)

    # Select configs to run
    if args.config:
        if args.config not in CONFIGS:
            print(f"ERROR: Unknown config '{args.config}'. Use --list to see options.")
            sys.exit(1)
        configs_to_run = {args.config: CONFIGS[args.config]}
    else:
        configs_to_run = CONFIGS

    # Run each config
    all_summaries = []
    for key, cfg in configs_to_run.items():
        if not check_draft_server(cfg["draft_url"]):
            gguf = cfg["draft_gguf"]
            print(f"\nDraft server not running at {cfg['draft_url']} for config '{key}'")
            print(f"Start with: llama-server -m {gguf} --port 8081 -ngl 999 --ctx-size 4096")
            print(f"Skipping {key}...\n")
            continue

        results = run_benchmark(cfg)
        overall = print_summary(cfg, results)
        all_summaries.append({"config": key, "name": cfg["name"], "overall_pct": overall})

    # Final cross-config summary
    if len(all_summaries) > 1:
        print(f"\n\n{'='*70}")
        print("CROSS-FAMILY SUMMARY")
        print(f"{'='*70}")
        print(f"{'Config':<50} {'Acceptance':>10}")
        print(f"{'-'*50} {'-'*10}")
        for s in all_summaries:
            print(f"{s['name']:<50} {s['overall_pct']:>9.1f}%")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
