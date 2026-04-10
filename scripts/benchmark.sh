#!/usr/bin/env bash
# Standardized benchmark for Tightwad cluster
# Usage: ./benchmark.sh [host:port]

set -euo pipefail

SERVER="${1:-127.0.0.1:8080}"

echo "=== Tightwad Benchmark ==="
echo "Server: $SERVER"
echo ""

# Check server health
if ! curl -sf "http://$SERVER/health" >/dev/null 2>&1; then
    echo "ERROR: Server not reachable at $SERVER"
    exit 1
fi

echo "Server healthy. Running benchmarks..."
echo ""

# Prompt processing (pp512)
echo "--- Prompt Processing (pp512) ---"
PROMPT=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 51)")
START=$(date +%s%N)
RESPONSE=$(curl -sf "http://$SERVER/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": 1, \"temperature\": 0}")
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))
echo "Time: ${ELAPSED}ms"
echo ""

# Text generation (tg128)
echo "--- Text Generation (tg128) ---"
START=$(date +%s%N)
RESPONSE=$(curl -sf "http://$SERVER/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Write a detailed essay about the history of computing:", "max_tokens": 128, "temperature": 0}')
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))

TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))")
if [ "$TOKENS" -gt 0 ] && [ "$ELAPSED" -gt 0 ]; then
    SPEED=$(python3 -c "print(f'{$TOKENS / ($ELAPSED / 1000):.1f}')")
    echo "Tokens: $TOKENS"
    echo "Time: ${ELAPSED}ms"
    echo "Speed: ${SPEED} tok/s"
else
    echo "Time: ${ELAPSED}ms"
    echo "Could not parse token count"
fi

echo ""
echo "=== Done ==="
