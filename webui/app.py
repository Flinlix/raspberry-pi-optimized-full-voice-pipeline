#!/usr/bin/env python3
"""Voice chat web UI.

Serves a chat front-end that streams tokens from an in-process llama.cpp model
(via the ``llama_chat`` package) and speaks the response with Piper. Text is
split into sentences as it streams so the first sentence is synthesized and
played while the LLM is still generating the rest -> short time-to-first-audio.

The ``llama_chat`` ChatWrapper keeps one KV cache alive across turns and prefills
only new tokens; the chat template is auto-detected from the model.

Run with the piper venv (with llama_chat importable, e.g. installed via
faster-llama-chat/install.sh or on PYTHONPATH):
    piper/venv/bin/python webui/app.py
"""

import base64
import io
import json
import re
import ssl
import threading
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from piper import PiperVoice

from llama_chat import ChatWrapper, Config, ContextOverflowError

# --- config ---------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 5000
WHISPER_URL = "http://127.0.0.1:8081/inference"
WHISPER_LANGUAGE = "de"  # 'auto' to detect, or e.g. 'de' / 'en'
VOICES_DIR = Path(__file__).resolve().parent.parent / "piper" / "voices"
WEB_DIR = Path(__file__).resolve().parent
CERT_DIR = WEB_DIR / "certs"  # if cert.pem + key.pem exist here, serve HTTPS
                              # (browsers only allow mic on HTTPS / localhost)
# Flush the first sentence early (low latency); let later ones grow a bit
# longer for more natural prosody (quality).
FIRST_FLUSH_MIN = 12
LATER_FLUSH_MIN = 30
SYSTEM_PROMPT = "Du bist ein Baum. Antworte stets in kurzen bis mittleren Sätzen auf Deutsch. Antworte nur in Fließtext. Antworte Faktenbasiert."  # e.g. "You are a helpful assistant. Always respond in complete sentences in German."
# --------------------------------------------------------------------------

_voice_cache: dict[str, PiperVoice] = {}
_voice_lock = threading.Lock()

# Monotonic generation counter. Each new /chat bumps it; an in-flight request
# whose id is no longer current aborts itself (stops synth + closes llama).
_gen_lock = threading.Lock()
_current_gen = 0


def begin_generation() -> int:
    global _current_gen
    with _gen_lock:
        _current_gen += 1
        return _current_gen


def is_current(gen_id: int) -> bool:
    with _gen_lock:
        return _current_gen == gen_id

# Sentence boundary: punctuation followed by space/end, or a newline.
_SENT_RE = re.compile(r"(.*?[.!?:;]+(?:\s|$)|.*?\n)", re.DOTALL)


def list_voices() -> list[str]:
    return sorted(p.stem for p in VOICES_DIR.glob("*.onnx"))


def get_voice(name: str) -> PiperVoice:
    with _voice_lock:
        if name not in _voice_cache:
            model = VOICES_DIR / f"{name}.onnx"
            if not model.exists():
                raise FileNotFoundError(name)
            _voice_cache[name] = PiperVoice.load(model)
        return _voice_cache[name]


def synth_wav_b64(voice: PiperVoice, text: str) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def split_sentences(buffer: str, *, first: bool) -> tuple[list[str], str]:
    """Pull complete sentences out of buffer; return (sentences, remainder)."""
    sentences = []
    pos = 0
    min_len = FIRST_FLUSH_MIN if first else LATER_FLUSH_MIN
    for m in _SENT_RE.finditer(buffer):
        chunk = m.group(0)
        if chunk.strip() and len(chunk.strip()) >= min_len:
            sentences.append(chunk.strip())
            pos = m.end()
            min_len = LATER_FLUSH_MIN  # only the very first uses the low bound
    return sentences, buffer[pos:]


# The persistent in-process chat wrapper and its config (created in main()).
chat: ChatWrapper | None = None
cfg: Config | None = None


def _has_history() -> bool:
    """True if the wrapper holds any non-system (evictable) message."""
    return any(m["role"] != "system" for m in chat.snapshot())


def cache_state() -> dict:
    """Current KV-cache layout for the inspector tab."""
    return {
        "messages": chat.snapshot(),
        "total": chat.total_tokens,
        "n_ctx": cfg.context_size,
        "threshold": cfg.threshold_tokens,
    }


def stream_llama(messages: list[dict]):
    """Yield reply text deltas for the latest user turn in ``messages``.

    The wrapper owns the conversation and its KV cache, so only the new user
    text is prefilled. The client still posts its full history each turn; we use
    it only to resync the wrapper when a conversation is (re)started:

    * a single message  -> a fresh conversation (page load / new chat): re-begin.
    * wrapper has no history but the client sent prior turns -> the server was
      restarted mid-conversation: replay the prior turns, then continue.
    """
    if not messages:
        return
    if len(messages) == 1:
        chat.begin(SYSTEM_PROMPT)
    elif not _has_history():
        prior = [(m["role"], m["content"]) for m in messages[:-1]]
        chat.begin(SYSTEM_PROMPT, prior)
    # The generator's return value is the Turn summary (incl. n_evicted).
    return (yield from chat.stream(messages[-1]["content"]))


def transcribe_wav(wav_bytes: bytes) -> str:
    """Forward WAV audio to whisper-server's /inference and return the text."""
    with wave.open(io.BytesIO(wav_bytes)) as wav_file:
        duration = wav_file.getnframes() / wav_file.getframerate()
    # Whisper pads every clip to a 30 s window and runs its encoder over all
    # 1500 context frames regardless of clip length. Shrinking audio_ctx to the
    # clip's share (50 frames/s, plus headroom against truncated words) cuts
    # encoder time roughly proportionally - the same trick whisper-stream uses.
    audio_ctx = min(1500, int(duration * 50) + 128)
    boundary = "----webuiboundary"
    parts = []
    for field, value in (("response_format", "json"), ("language", WHISPER_LANGUAGE),
                         ("audio_ctx", str(audio_ctx))):
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{field}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode())
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
    parts.append(b"Content-Type: audio/wav\r\n\r\n")
    parts.append(wav_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        WHISPER_URL, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj.get("text", "").strip()


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quieter logs
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = (WEB_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/voices":
            self._send(200, json.dumps(list_voices()).encode())
        elif self.path == "/cache":
            self._send(200, json.dumps(cache_state()).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def handle_transcribe(self):
        length = int(self.headers.get("Content-Length", 0))
        wav_bytes = self.rfile.read(length)
        try:
            text = transcribe_wav(wav_bytes)
            self._send(200, json.dumps({"text": text}).encode())
        except Exception as e:
            self._send(502, json.dumps({"error": str(e)}).encode())

    def do_POST(self):
        if self.path == "/transcribe":
            self.handle_transcribe()
            return
        if self.path != "/chat":
            self._send(404, b'{"error":"not found"}')
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        messages = req.get("messages", [])
        voice_name = req.get("voice", "")

        try:
            voice = get_voice(voice_name)
        except (FileNotFoundError, Exception) as e:
            self._send(400, json.dumps({"error": str(e)}).encode())
            return

        # A new request supersedes any in-flight one.
        gen_id = begin_generation()

        # SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        buffer = ""
        first = True
        idx = 0
        stream = stream_llama(messages)
        turn = None  # set to the Turn summary on normal (non-superseded) completion
        try:
            while is_current(gen_id):
                try:
                    delta = next(stream)
                except StopIteration as done:
                    turn = done.value
                    break
                self.wfile.write(sse("token", {"text": delta}))
                self.wfile.flush()
                buffer += delta
                sentences, buffer = split_sentences(buffer, first=first)
                for s in sentences:
                    if not is_current(gen_id):
                        break
                    first = False
                    b64 = synth_wav_b64(voice, s)
                    self.wfile.write(sse("audio", {"i": idx, "wav": b64}))
                    self.wfile.flush()
                    idx += 1
            # turn is set only if generation finished without being superseded.
            if turn is not None and is_current(gen_id):
                tail = buffer.strip()
                if tail:
                    b64 = synth_wav_b64(voice, tail)
                    self.wfile.write(sse("audio", {"i": idx, "wav": b64}))
                    self.wfile.flush()
                if turn.n_evicted:
                    self.wfile.write(sse("evicted", {"n": turn.n_evicted}))
                    self.wfile.flush()
                self.wfile.write(sse("done", {}))
                self.wfile.flush()
        except ContextOverflowError as e:
            try:
                self.wfile.write(sse("error", {"message": str(e)}))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            # Closing the generator tears down the llama connection, freeing
            # the generation slot for the superseding request.
            stream.close()


def main():
    global chat, cfg
    cfg = Config()
    print(f"Loading model: {cfg.model_path}")
    chat = ChatWrapper(cfg)
    chat.begin(SYSTEM_PROMPT)

    httpd = ThreadingHTTPServer((HOST, PORT), Handler)

    cert, key = CERT_DIR / "cert.pem", CERT_DIR / "key.pem"
    scheme = "http"
    if cert.exists() and key.exists():
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"

    print(f"Voices: {', '.join(list_voices())}")
    print(f"Serving on {scheme}://{HOST}:{PORT}")
    print(f"Context: {cfg.context_size} tokens, history budget {cfg.threshold_tokens} "
          f"({int(cfg.eviction_threshold * 100)}%)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
