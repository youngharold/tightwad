# llama.cpp patches for Tightwad

## Hot-expert routing log (profile-guided placement)

**Why:** `llama.h`'s public C API exposes no hook to observe MoE expert-routing
decisions per token. `ggml_backend_sched_eval_callback` fires at graph-op
granularity without surfacing selected expert IDs cleanly enough for profiling.
A 15-line patch that prints chosen expert IDs from inside
`llm_build_moe_ffn` is the pragmatic path.

**Status:** experimental. Not upstreamed. Maintain against the pinned llama.cpp
commit recorded in `scripts/install-coordinator.sh`.

## Applying

```bash
cd /path/to/llama.cpp
git checkout <pinned-commit>   # see install-coordinator.sh
git apply /path/to/tightwad/scripts/patches/llamacpp-moe-log.patch
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

Then set `LLAMA_LOG_MOE=1` in the env for `llama-server` / `rpc-server` and
run `tightwad moe profile` to capture.

## Patch shape

```diff
--- a/src/llama-model.cpp
+++ b/src/llama-model.cpp
@@ llm_build_moe_ffn(...) {
+    const bool log_moe = std::getenv("LLAMA_LOG_MOE") != nullptr;
+    if (log_moe) {
+        // Example format parsed by tightwad/moe_profile.py:
+        //   moe: layer=LAYER chosen=[E0,E1,E2,E3]
+        // Integration point depends on the pinned llama.cpp revision.
+    }
```

The actual patch must be regenerated against the specific llama.cpp SHA you
build from — the MoE codepath has moved between `src/llama.cpp`,
`src/llama-model.cpp`, and `src/llama-graph.cpp` over recent releases.

## Fallback when unpatched

`tightwad/moe_profile.py` handles stderr from an unpatched build by counting
`n_expert_used` slot events. The result populates `total_tokens` but leaves
`hits` empty, so `profile-guided` degrades to `balanced`. `tightwad doctor`
warns when `profile-guided` is configured against a worker whose stderr shows
no `moe: layer=` lines.
