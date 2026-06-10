#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Stop any existing instances first so we never collide on a port.
./stop.sh > /dev/null 2>&1 || true
# Belt-and-suspenders: clear stragglers not tracked by a pid file.
pkill -f "whisper.cpp/build/bin/whisper-server" 2>/dev/null || true
pkill -f "webui/app.py" 2>/dev/null || true
sleep 1

echo "Starting whisper-server..."
./whisper/whisper.cpp/build/bin/whisper-server \
  --model whisper/whisper.cpp/models/ggml-base.bin \
  --host 127.0.0.1 --port 8081 --language de --threads 4 \
  > /tmp/whisper.log 2>&1 &
echo $! > /tmp/whisper.pid

echo "Waiting for whisper-server..."
until curl -sf http://127.0.0.1:8081/ > /dev/null 2>&1; do
  if ! kill -0 "$(cat /tmp/whisper.pid)" 2>/dev/null; then
    echo "ERROR: whisper-server died on startup. Last log lines:"; tail -n 15 /tmp/whisper.log; exit 1
  fi
  sleep 1
done
echo "whisper-server ready."

echo "Starting web UI..."
LLAMA_SITE=$(./llama/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
PYTHONPATH="$PWD/llama:$LLAMA_SITE" ./piper/venv/bin/python webui/app.py > /tmp/webui.log 2>&1 &
echo $! > /tmp/webui.pid

# HTTPS if a cert is present (needed for microphone access from other devices).
if [ -f webui/certs/cert.pem ]; then SCHEME=https; else SCHEME=http; fi

echo "Waiting for web UI..."
until curl -skf ${SCHEME}://127.0.0.1:5000/voices > /dev/null 2>&1; do
  if ! kill -0 "$(cat /tmp/webui.pid)" 2>/dev/null; then
    echo "ERROR: web UI died on startup. Last log lines:"; tail -n 15 /tmp/webui.log; exit 1
  fi
  sleep 1
done

IP=$(hostname -I | awk '{print $1}')
echo "Ready at ${SCHEME}://${IP:-localhost}:5000"
