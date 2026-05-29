#!/usr/bin/env python3
"""Voice chat web UI.

Serves a chat front-end that streams tokens from a local llama-server
(OpenAI-compatible API) and speaks the response with Piper. Text is split
into sentences as it streams so the first sentence is synthesized and played
while the LLM is still generating the rest -> short time-to-first-audio.

Run with the piper venv:
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

# --- config ---------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 5000
LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
WHISPER_URL = "http://127.0.0.1:8081/inference"
WHISPER_LANGUAGE = "auto"  # 'auto' to detect, or e.g. 'de' / 'en'
VOICES_DIR = Path(__file__).resolve().parent.parent / "piper" / "voices"
WEB_DIR = Path(__file__).resolve().parent
CERT_DIR = WEB_DIR / "certs"  # if cert.pem + key.pem exist here, serve HTTPS
                              # (browsers only allow mic on HTTPS / localhost)
# Flush the first sentence early (low latency); let later ones grow a bit
# longer for more natural prosody (quality).
FIRST_FLUSH_MIN = 12
LATER_FLUSH_MIN = 30
SYSTEM_PROMPT = "Du bist ein Sprachassistent. Antworte stets in vollständigen Sätzen auf Deutsch. Antworte nur in Fließtext."  # e.g. "You are a helpful assistant. Always respond in complete sentences in German."
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


def stream_llama(messages: list[dict]):
    """Yield text deltas from llama-server's streaming chat endpoint."""
    full_messages = []
    if SYSTEM_PROMPT:
        full_messages.append({"role": "system", "content": SYSTEM_PROMPT})
    full_messages.extend(messages)
    payload = json.dumps({
        "messages": full_messages,
        "stream": True,
        "temperature": 0.7,
    }).encode("utf-8")
    req = urllib.request.Request(
        LLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta


def transcribe_wav(wav_bytes: bytes) -> str:
    """Forward WAV audio to whisper-server's /inference and return the text."""
    boundary = "----webuiboundary"
    parts = []
    for field, value in (("response_format", "json"), ("language", WHISPER_LANGUAGE)):
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
        try:
            for delta in stream:
                if not is_current(gen_id):
                    break  # superseded by a newer request
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
            else:
                # loop finished without break: flush trailing text + done
                tail = buffer.strip()
                if tail and is_current(gen_id):
                    b64 = synth_wav_b64(voice, tail)
                    self.wfile.write(sse("audio", {"i": idx, "wav": b64}))
                    self.wfile.flush()
                if is_current(gen_id):
                    self.wfile.write(sse("done", {}))
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            # Closing the generator tears down the llama connection, freeing
            # the generation slot for the superseding request.
            stream.close()


def main():
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)

    cert, key = CERT_DIR / "cert.pem", CERT_DIR / "key.pem"
    scheme = "http"
    if cert.exists() and key.exists():
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"

    print(f"Voices: {', '.join(list_voices())}")
    print(f"Serving on {scheme}://{HOST}:{PORT}  (llama-server expected at {LLAMA_URL})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
