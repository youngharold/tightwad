#!/usr/bin/env bash
# Bootstrap rpc-server on a Windows machine (run from WSL or Git Bash)
# Usage: ./install-worker.sh [build_dir]

set -euo pipefail

BUILD_DIR="${1:-$HOME/llama.cpp}"
PORTS="${RPC_PORTS:-50052,50053}"

echo "=== Tightwad RPC Worker Setup ==="
echo "Build dir: $BUILD_DIR"
echo "RPC ports: $PORTS"

# Clone or update llama.cpp
if [ -d "$BUILD_DIR" ]; then
    echo "Updating llama.cpp..."
    cd "$BUILD_DIR" && git pull
else
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$BUILD_DIR"
    cd "$BUILD_DIR"
fi

# Build with CUDA + RPC
echo "Building with CUDA + RPC backends..."
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON
cmake --build build --config Release -j

echo ""
echo "=== Build complete ==="
echo "rpc-server binary: $BUILD_DIR/build/bin/rpc-server"
echo ""
echo "Start workers with:"
IFS=',' read -ra PORT_ARRAY <<< "$PORTS"
for port in "${PORT_ARRAY[@]}"; do
    echo "  $BUILD_DIR/build/bin/rpc-server -p $port"
done
echo ""
echo "Or install as Windows services with NSSM:"
for port in "${PORT_ARRAY[@]}"; do
    echo "  nssm install tightwad-rpc-$port $BUILD_DIR/build/bin/rpc-server.exe -p $port"
done
