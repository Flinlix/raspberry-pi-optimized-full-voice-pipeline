#!/usr/bin/env bash
cd "$(dirname "$0")"

# Kill by saved pid file if present.
for name in webui llama whisper; do
  pid_file="/tmp/${name}.pid"
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    if kill "$pid" 2>/dev/null; then
      echo "Stopped $name (pid $pid)"
    fi
    rm -f "$pid_file"
  fi
done

# Fall back to pattern match to catch any untracked instances.
pkill -f "webui/app.py" 2>/dev/null && echo "Stopped stray web UI" || true
pkill -f "llama.cpp/build/bin/llama-server" 2>/dev/null && echo "Stopped stray llama-server" || true
pkill -f "whisper.cpp/build/bin/whisper-server" 2>/dev/null && echo "Stopped stray whisper-server" || true
