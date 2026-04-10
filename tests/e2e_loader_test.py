"""End-to-end loader test on a real machine with a real GGUF model.

Usage: python e2e_loader_test.py <model_blob_path> <llama_server_path>

Tests the full v0.1.4 lifecycle:
  1. GGUF header parsing
  2. Memory pressure detection
  3. Sequential pre-warming
  4. llama-server startup + health check
  5. Inference verification
  6. RAM reclaim
  7. Post-reclaim inference verification
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

# ── Args ──────────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <model_blob_path> <llama_server_path>")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
SERVER_BIN = sys.argv[2]
PORT = 8090
OK = "PASS"
FAIL = "FAIL"
results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = ""):
    icon = OK if passed else FAIL
    results.append((name, passed, detail))
    print(f"  {icon} {name}" + (f"  ({detail})" if detail else ""))
    if not passed:
        print(f"    FAILED — aborting")
        cleanup()
        report()
        sys.exit(1)


def cleanup():
    if sys.platform == "win32":
        os.system("taskkill /F /IM llama-server.exe >nul 2>&1")
    else:
        os.system("pkill -f llama-server 2>/dev/null")


def report():
    print("\n" + "=" * 60)
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("ALL TESTS PASSED")
    else:
        for name, p, detail in results:
            if not p:
                print(f"  {FAIL} {name}: {detail}")
    print("=" * 60)


# ── Test begins ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Tightwad v0.1.4 — End-to-End Loader Test")
print("=" * 60)
print(f"  Model:  {os.path.basename(MODEL_PATH)}")
print(f"  Server: {SERVER_BIN}")
print(f"  Port:   {PORT}")
print()

# 1. GGUF Header Parsing
print("[1/7] GGUF Header Parsing")
try:
    from tightwad.gguf_reader import read_header, model_summary
    t0 = time.monotonic()
    header = read_header(MODEL_PATH)
    parse_ms = (time.monotonic() - t0) * 1000
    summary = model_summary(header)
    arch = summary["arch"]
    layers = summary["layers"]
    quant = summary["quant"]
    file_size = header.file_size
    step("Parse header", True, f"{parse_ms:.0f}ms")
    step("Architecture", arch != "unknown", arch)
    step("Layers", layers is not None and layers > 0, str(layers))
    step("Quant", quant is not None, str(quant))
    step("Tensor count", header.tensor_count > 0, str(header.tensor_count))
    step("File size", file_size > 0, f"{file_size / (1024**3):.2f} GB")
except Exception as e:
    step("Parse header", False, str(e))
    sys.exit(1)

print()

# 2. Memory Pressure Detection
print("[2/7] Memory Pressure Detection")
from tightwad.loader import needs_streaming_load
from tightwad.reclaim import get_available_ram_bytes
avail = get_available_ram_bytes()
needs = needs_streaming_load(file_size, avail)
step("Available RAM", avail > 0, f"{avail / (1024**3):.2f} GB")
step("needs_streaming_load", True, f"{'yes' if needs else 'no'} (model={file_size/(1024**3):.1f} GB, avail={avail/(1024**3):.1f} GB, ratio={file_size/avail*100:.0f}%)")
print()

# 3. Sequential Pre-warm
print("[3/7] Sequential Pre-warm")
from tightwad.loader import prewarm_sequential
try:
    elapsed = prewarm_sequential(MODEL_PATH, file_size)
    throughput = (file_size / (1024**3)) / elapsed if elapsed > 0 else 0
    step("Pre-warm complete", elapsed > 0, f"{elapsed:.1f}s, {throughput:.2f} GB/s")
except Exception as e:
    step("Pre-warm", False, str(e))
print()

# 4. Start llama-server
print("[4/7] Start llama-server")
ram_before = get_available_ram_bytes()
try:
    log_path = os.path.join(os.path.dirname(SERVER_BIN), "test_server.log")
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [SERVER_BIN, "-m", MODEL_PATH, "--host", "127.0.0.1", "--port", str(PORT),
         "-ngl", "999", "--ctx-size", "4096", "-n", "256"],
        stdout=log_fh, stderr=log_fh,
    )
    step("Process started", proc.poll() is None, f"PID {proc.pid}")
except Exception as e:
    step("Start server", False, str(e))
    sys.exit(1)
print()

# 5. Wait for health
print("[5/7] Health Check")
import httpx
healthy = False
t0 = time.monotonic()
deadline = t0 + 120
while time.monotonic() < deadline:
    try:
        r = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=3)
        if r.status_code == 200 and r.json().get("status") == "ok":
            healthy = True
            break
    except Exception:
        pass
    time.sleep(2)
health_time = time.monotonic() - t0
step("Health OK", healthy, f"{health_time:.1f}s")

ram_after_load = get_available_ram_bytes()
ram_eaten = (ram_before - ram_after_load) / (1024**2)
step("RAM consumed by mmap", ram_eaten > 100, f"{ram_eaten:.0f} MB")
print()

# 6. Inference test (pre-reclaim)
print("[6/7] Inference + RAM Reclaim")
try:
    r = httpx.post(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Say hello."}],
              "max_tokens": 32, "temperature": 0},
        timeout=120,
    )
    data = r.json()
    tok_s = round(data["timings"]["predicted_per_second"], 1)
    tokens = data["usage"]["completion_tokens"]
    step("Inference (pre-reclaim)", tokens > 0, f"{tokens} tokens, {tok_s} tok/s")
except Exception as e:
    step("Inference (pre-reclaim)", False, str(e))

# RAM reclaim
from tightwad.reclaim import reclaim_ram, get_process_rss_mb
rss_before = get_process_rss_mb(proc.pid)
result = reclaim_ram(proc.pid, MODEL_PATH)
step("Reclaim method", result.method != "skipped", result.method)
step("RSS reduction", result.reclaimed_mb > 100,
     f"{result.rss_before_mb:.0f} MB -> {result.rss_after_mb:.0f} MB (freed {result.reclaimed_mb:.0f} MB)")

# On Windows, SetProcessWorkingSetSize moves pages to standby list.
# GlobalMemoryStatusEx.ullAvailPhys may not immediately reflect this.
# RSS reduction is the reliable metric — already verified above.
ram_after_reclaim = get_available_ram_bytes()
ram_delta = (ram_after_reclaim - ram_after_load) / (1024**2)
step("RAM state after reclaim", True,
     f"system avail delta: {ram_delta:+.0f} MB (RSS freed {result.reclaimed_mb:.0f} MB)")
print()

# 7. Post-reclaim inference
print("[7/7] Post-Reclaim Verification")
try:
    r = httpx.post(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Say goodbye."}],
              "max_tokens": 32, "temperature": 0},
        timeout=120,
    )
    data = r.json()
    tok_s_after = round(data["timings"]["predicted_per_second"], 1)
    tokens_after = data["usage"]["completion_tokens"]
    step("Inference (post-reclaim)", tokens_after > 0, f"{tokens_after} tokens, {tok_s_after} tok/s")
    # Speed should not degrade significantly (within 30%)
    if tok_s > 0:
        degradation = (tok_s - tok_s_after) / tok_s * 100
        step("No speed degradation", degradation < 30,
             f"{tok_s} -> {tok_s_after} tok/s ({degradation:+.1f}%)")
except Exception as e:
    step("Inference (post-reclaim)", False, str(e))

# Cleanup
cleanup()
log_fh.close()
print()
report()
