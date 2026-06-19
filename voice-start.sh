#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Stop any existing instances first so we never collide on a port.
./voice-stop.sh > /dev/null 2>&1 || true
sleep 1

echo "Starting whisper-server..."
./whisper/whisper.cpp/build/bin/whisper-server \
  --model whisper/whisper.cpp/models/ggml-base.bin \
  --host 127.0.0.1 --port 8081 --language de --threads 4 \
  > /tmp/whisper.log 2>&1 &
echo $! > /tmp/whisper.pid

# Tear down whisper-server when the voice loop exits (incl. Ctrl-C).
trap './voice-stop.sh > /dev/null 2>&1 || true' EXIT

echo "Waiting for whisper-server..."
until curl -sf http://127.0.0.1:8081/ > /dev/null 2>&1; do
  if ! kill -0 "$(cat /tmp/whisper.pid)" 2>/dev/null; then
    echo "ERROR: whisper-server died on startup. Last log lines:"; tail -n 15 /tmp/whisper.log; exit 1
  fi
  sleep 1
done
echo "whisper-server ready."

# Run the hands-free voice loop in the foreground (Ctrl-C to stop).
LLAMA_SITE=$(./llama/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
PYTHONPATH="$PWD/llama:$LLAMA_SITE" ./piper/venv/bin/python webui/voice_loop.py
