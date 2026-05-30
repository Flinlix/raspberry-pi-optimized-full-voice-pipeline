#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

MODEL="${1:-base}"   # tiny | base | small — pass as first arg, default: base

echo "=== whisper.cpp install ==="

# --- Dependencies ---
echo "[1/4] Installing dependencies..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential git cmake \
    libsdl2-dev \
    curl

# --- Clone or update ---
if [ ! -d "whisper.cpp/.git" ]; then
    echo "[2/4] Cloning whisper.cpp..."
    git clone https://github.com/ggerganov/whisper.cpp whisper.cpp
else
    echo "[2/4] whisper.cpp already cloned, pulling latest..."
    git -C whisper.cpp pull --ff-only
fi

# --- Build ---
echo "[3/4] Building (NEON auto-detected, SDL2 enabled)..."
cmake -S whisper.cpp -B whisper.cpp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_SDL2=ON
cmake --build whisper.cpp/build -j4 \
    --target whisper-server \
    --target whisper-stream \
    --target whisper-cli

echo "Built binaries:"
ls whisper.cpp/build/bin/

# --- Model ---
echo "[4/4] Downloading model: ggml-${MODEL}.bin ..."
if [ -f "whisper.cpp/models/ggml-${MODEL}.bin" ]; then
    echo "  Model already exists, skipping download."
else
    bash whisper.cpp/models/download-ggml-model.sh "${MODEL}"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Run streaming transcription:"
echo "  ./whisper.cpp/build/bin/whisper-stream -m whisper.cpp/models/ggml-${MODEL}.bin -l de -t 4"
echo ""
echo "Run server (used by start.sh):"
echo "  ./whisper.cpp/build/bin/whisper-server -m whisper.cpp/models/ggml-${MODEL}.bin --host 127.0.0.1 --port 8081 -l de -t 4"
