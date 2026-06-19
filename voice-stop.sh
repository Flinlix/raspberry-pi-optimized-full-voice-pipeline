#!/usr/bin/env bash
cd "$(dirname "$0")"

# Kill whisper-server by saved pid file if present.
if [ -f /tmp/whisper.pid ]; then
  pid=$(cat /tmp/whisper.pid)
  if kill "$pid" 2>/dev/null; then
    echo "Stopped whisper (pid $pid)"
  fi
  rm -f /tmp/whisper.pid
fi

# Fall back to pattern match to catch any untracked instances.
pkill -f "webui/voice_loop.py" 2>/dev/null && echo "Stopped stray voice loop" || true
pkill -f "whisper.cpp/build/bin/whisper-server" 2>/dev/null && echo "Stopped stray whisper-server" || true
