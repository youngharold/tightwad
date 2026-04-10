#!/usr/bin/env python3
"""Benchmark speculative decoding acceptance rates across configurations.

Tests draft→target model pairs with text-match verification.
Supports Ollama, llama-server, and OpenAI-compatible APIs (OpenRouter, etc).
"""

import asyncio
import json
import os
import time

import httpx

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

PROMPTS = [
    {
        "name": "factual",
        "messages": [{"role": "user", "content": "What are the three laws of thermodynamics? Explain each in one sentence."}],
    },
    {
        "name": "code",
        "messages": [{"role": "user", "content": "Write a Python function that checks if a string is a palindrome. Include a docstring."}],
    },
    {
        "name": "creative",
        "messages": [{"role": "user", "content": "Write a haiku about GPU inference."}],
    },
    {
        "name": "reasoning",
        "messages": [{"role": "user", "content": "If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire trip?"}],
    },
    {
        "name": "list",
        "messages": [{"role": "user", "content": "List 5 practical uses for speculative decoding in production AI systems."}],
    },
]


def ollama_text(data: dict) -> str:
    """Extract text from Ollama response, handling thinking mode."""
    return data.get("response", "") or data.get("thinking", "")


async def generate_ollama(url: str, model: str, prompt: str, n: int) -> tuple[str, dict]:
    """Generate via Ollama raw API. Returns (text, raw_response)."""
    body = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {"num_predict": n, "temperature": 0},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{url}/api/generate", json=body)
    resp.raise_for_status()
    data = resp.json()
    text = ollama_text(data)
    return text, data


async def generate_openai(url: str, model: str, messages: list, n: int,
                          api_key: str | None = None) -> tuple[str, dict]:
    """Generate via OpenAI-compatible chat API. Returns (text, raw_response)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": n,
        "temperature": 0,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{url}/v1/chat/completions", json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return text, data


async def baseline_generate(target: dict, prompt: dict, max_tokens: int = 256) -> dict:
    """Generate full response from target only (baseline)."""
    start = time.monotonic()
    user_content = prompt["messages"][0]["content"]

    if target["backend"] == "ollama":
        raw_prompt = (
            f"<|im_start|>system\nYou are a helpful assistant. /no_think<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        text, data = await generate_ollama(target["url"], target["model"], raw_prompt, max_tokens)
        if "<|im_end|>" in text:
            text = text[:text.index("<|im_end|>")]
        eval_count = data.get("eval_count", len(text.split()))
        eval_dur = data.get("eval_duration", 0) / 1e9
        tok_s = eval_count / eval_dur if eval_dur > 0 else 0
    else:
        msgs = [{"role": "system", "content": "You are a helpful assistant."}] + prompt["messages"]
        text, data = await generate_openai(
            target["url"], target["model"], msgs, max_tokens,
            api_key=target.get("api_key"),
        )
        usage = data.get("usage", {})
        eval_count = usage.get("completion_tokens", len(text.split()))
        tok_s = 0

    elapsed = time.monotonic() - start
    return {
        "text": text.strip(),
        "tokens": eval_count,
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(tok_s, 1) if tok_s else round(eval_count / elapsed, 1),
    }


async def spec_decoding(draft: dict, target: dict, prompt: dict,
                        max_tokens: int = 256, max_draft_tokens: int = 8) -> dict:
    """Run speculative decoding: draft on local, verify against target."""
    user_content = prompt["messages"][0]["content"]
    base_prompt = (
        f"<|im_start|>system\nYou are a helpful assistant. /no_think<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    generated = ""
    total_rounds = 0
    total_drafted = 0
    total_accepted = 0
    start = time.monotonic()

    for _ in range(50):
        current_prompt = base_prompt + generated

        # --- Draft phase (always Ollama for local GPU) ---
        if draft["backend"] == "ollama":
            draft_text, _ = await generate_ollama(
                draft["url"], draft["model"], current_prompt, max_draft_tokens
            )
        else:
            # llamacpp
            body = {"prompt": current_prompt, "max_tokens": max_draft_tokens,
                    "temperature": 0, "stream": False}
            async with httpx.AsyncClient(timeout=30.0) as c:
                resp = await c.post(f"{draft['url']}/v1/completions", json=body)
            resp.raise_for_status()
            draft_text = resp.json()["choices"][0].get("text", "")

        if not draft_text:
            break
        draft_done = False
        if "<|im_end|>" in draft_text:
            draft_text = draft_text[:draft_text.index("<|im_end|>")]
            draft_done = True
        if not draft_text:
            break

        # --- Verify phase ---
        if target["backend"] == "ollama":
            target_text, _ = await generate_ollama(
                target["url"], target["model"], current_prompt, max_draft_tokens
            )
        else:
            # OpenAI/OpenRouter chat API — reconstruct conversation
            msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
            ] + prompt["messages"]
            if generated:
                # Send partial assistant response, ask for continuation
                msgs.append({"role": "assistant", "content": generated})
                msgs.append({"role": "user", "content": "Continue from exactly where you left off. Do not repeat anything."})
            target_text, _ = await generate_openai(
                target["url"], target["model"], msgs, max_draft_tokens,
                api_key=target.get("api_key"),
            )

        target_done = False
        if "<|im_end|>" in target_text:
            target_text = target_text[:target_text.index("<|im_end|>")]
            target_done = True

        # Text-match verification (normalize whitespace for fair cross-backend comparison)
        def normalize(s: str) -> str:
            import re
            return re.sub(r'\s+', ' ', s).strip()

        draft_norm = normalize(draft_text)
        target_norm = normalize(target_text)
        match_len = 0
        for i in range(min(len(draft_norm), len(target_norm))):
            if draft_norm[i] == target_norm[i]:
                match_len = i + 1
            else:
                break

        if os.environ.get("BENCH_DEBUG"):
            print(f"\n    [round {total_rounds+1}] draft={draft_text[:60]!r} target={target_text[:60]!r} match={match_len}")

        total_rounds += 1
        total_drafted += len(draft_norm)
        total_accepted += match_len

        # Accept target output
        generated += target_text
        word_count = len(generated.split())

        if target_done or draft_done or word_count >= max_tokens or not target_text:
            break

    elapsed = time.monotonic() - start
    acceptance = total_accepted / total_drafted if total_drafted > 0 else 0

    return {
        "text": generated.strip(),
        "rounds": total_rounds,
        "drafted_chars": total_drafted,
        "accepted_chars": total_accepted,
        "acceptance_rate": round(acceptance, 3),
        "elapsed_s": round(elapsed, 2),
        "word_count": len(generated.split()),
    }


async def run_config(name: str, draft: dict, target: dict) -> dict:
    """Run all prompts for one config."""
    print(f"\n{'='*70}")
    print(f"Config: {name}")
    print(f"  Draft:  {draft['model']} @ {draft['url']} ({draft['backend']})")
    print(f"  Target: {target['model']} @ {target['url']} ({target['backend']})")
    print(f"{'='*70}")

    results = []
    for prompt in PROMPTS:
        print(f"\n  [{prompt['name']}] ", end="", flush=True)

        # Baseline
        print("baseline...", end="", flush=True)
        try:
            base = await baseline_generate(target, prompt)
            print(f" {base['elapsed_s']}s ({base['tok_s']} tok/s)", end="", flush=True)
        except Exception as e:
            print(f" FAILED: {e}")
            results.append({"prompt": prompt["name"], "baseline": {"error": str(e)}, "speculative": {"error": "skipped"}})
            continue

        # Speculative
        print(" | spec...", end="", flush=True)
        try:
            spec = await spec_decoding(draft, target, prompt)
            print(f" {spec['elapsed_s']}s ({spec['acceptance_rate']*100:.0f}% accept, {spec['rounds']} rounds)")
        except Exception as e:
            print(f" FAILED: {e}")
            spec = {"error": str(e)}

        speedup = round(base["elapsed_s"] / spec["elapsed_s"], 2) if "elapsed_s" in spec and spec["elapsed_s"] > 0 else None
        results.append({"prompt": prompt["name"], "baseline": base, "speculative": spec, "speedup": speedup})

    ok = [r for r in results if "error" not in r.get("speculative", {})]
    if ok:
        avg_accept = sum(r["speculative"]["acceptance_rate"] for r in ok) / len(ok)
        avg_speed = sum(r["speedup"] for r in ok if r["speedup"]) / len(ok)
        avg_rounds = sum(r["speculative"]["rounds"] for r in ok) / len(ok)
    else:
        avg_accept = avg_speed = avg_rounds = 0

    summary = {
        "config": name,
        "draft": f"{draft['model']} ({draft['backend']})",
        "target": f"{target['model']} ({target['backend']})",
        "avg_acceptance_rate": round(avg_accept, 3),
        "avg_speedup": round(avg_speed, 2),
        "avg_rounds": round(avg_rounds, 1),
        "prompts": results,
    }
    print(f"\n  Summary: {avg_accept*100:.1f}% acceptance, {avg_speed:.2f}x speedup, {avg_rounds:.1f} avg rounds")
    return summary


async def main():
    draft_2070 = {
        "url": "http://192.168.1.101:11434",
        "model": "qwen3:8b",
        "backend": "ollama",
    }

    configs = [
        # Local same-family: Qwen3-8B (2070) → Qwen3-32B (Desktop/Ollama)
        (
            "Qwen3-8B → Qwen3-32B (local Ollama)",
            draft_2070,
            {"url": "http://192.168.1.100:11434", "model": "qwen3:32b", "backend": "ollama"},
        ),
        # Cloud same-family: Qwen3-8B (2070) → Qwen3.5-397B (OpenRouter)
        (
            "Qwen3-8B → Qwen3.5-397B (OpenRouter)",
            draft_2070,
            {"url": "https://openrouter.ai/api", "model": "qwen/qwen3.5-397b-a17b",
             "backend": "openai", "api_key": OPENROUTER_KEY},
        ),
        # Cloud cross-family: Qwen3-8B (2070) → Llama 3.3 70B (OpenRouter)
        (
            "Qwen3-8B → Llama 3.3 70B (OpenRouter, cross-family)",
            draft_2070,
            {"url": "https://openrouter.ai/api", "model": "meta-llama/llama-3.3-70b-instruct",
             "backend": "openai", "api_key": OPENROUTER_KEY},
        ),
    ]

    all_results = []
    for name, draft, target in configs:
        try:
            result = await run_config(name, draft, target)
            all_results.append(result)
        except Exception as e:
            print(f"\n  CONFIG FAILED: {e}")
            all_results.append({"config": name, "error": str(e)})

    # Write results
    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults written to benchmarks/benchmark_results.json")

    # Summary table
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)
    print(f"{'Config':<55} {'Accept%':>8} {'Speedup':>8} {'Rounds':>7}")
    print("-"*90)
    for r in all_results:
        if "error" not in r:
            print(f"{r['config']:<55} {r['avg_acceptance_rate']*100:>7.1f}% {r['avg_speedup']:>7.2f}x {r['avg_rounds']:>7.1f}")
        else:
            print(f"{r['config']:<55} {'FAILED':>8}")
    print("="*90)


if __name__ == "__main__":
    asyncio.run(main())
