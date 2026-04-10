#!/usr/bin/env bash
# Bootstrap llama-server coordinator on Ubuntu ROCm machine
# Usage: ./install-coordinator.sh [build_dir]

set -euo pipefail

BUILD_DIR="${1:-$HOME/llama.cpp}"
GPU_TARGET="${AMDGPU_TARGET:-gfx1100}"  # 7900 XTX

echo "=== Tightwad Coordinator Setup (ROCm) ==="
echo "Build dir: $BUILD_DIR"
echo "GPU target: $GPU_TARGET"

# Check ROCm
if ! command -v rocminfo &>/dev/null; then
    echo "ERROR: ROCm not found. Install ROCm first:"
    echo "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    exit 1
fi

echo "ROCm detected:"
rocminfo | grep -E "Name:|Marketing Name:" | head -4

# Clone or update llama.cpp
if [ -d "$BUILD_DIR" ]; then
    echo "Updating llama.cpp..."
    cd "$BUILD_DIR" && git pull
else
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$BUILD_DIR"
    cd "$BUILD_DIR"
fi

# Build with HIP + RPC
echo "Building with HIP (ROCm) + RPC backends..."
cmake -B build \
    -DGGML_HIP=ON \
    -DGGML_RPC=ON \
    -DAMDGPU_TARGETS="$GPU_TARGET" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(nproc)"

echo ""
echo "=== Build complete ==="
echo "llama-server binary: $BUILD_DIR/build/bin/llama-server"
echo ""
echo "Copy to /usr/local/bin:"
echo "  sudo cp $BUILD_DIR/build/bin/llama-server /usr/local/bin/"
echo ""
echo "Test with:"
echo "  llama-server -m /path/to/model.gguf -ngl 999 --rpc <worker-ip>:50052"
