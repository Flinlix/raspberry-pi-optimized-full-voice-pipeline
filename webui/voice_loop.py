#!/usr/bin/env python3
"""Hands-free voice assistant on the Pi's ReSpeaker mic + speaker.

Same pipeline as ``webui/app.py`` (whisper STT -> llama_chat LLM -> Piper TTS),
but the browser's microphone and speaker are replaced by the local ReSpeaker
4-Mic Array: ``arecord`` captures from the array and ``aplay`` plays back through
the speaker on its 3.5 mm jack (both ALSA card 2 -> ``plughw:2,0``).

The loop is hands-free and turn-based: it listens, auto-stops when you stop
talking (simple RMS voice-activity detection), transcribes, streams a reply from
the model, and speaks each sentence as soon as it is produced. It never records
while speaking, so it does not transcribe its own output.

Requires whisper-server running on 127.0.0.1:8081 (see ``voice-start.sh``).

Run with the piper venv and ``llama_chat`` importable (same as app.py):
    PYTHONPATH=llama:<llama-site-packages> piper/venv/bin/python webui/voice_loop.py
"""

import io
import logging
import queue
import signal
import subprocess
import sys
import threading
import wave

import numpy as np

from llama_chat import ChatWrapper, Config, ContextOverflowError

# Piper logs a warning for every phoneme espeak produces that isn't in the
# voice's id map (e.g. the cedilla in loanwords like "Façade"); the phoneme is
# simply dropped and synthesis continues, so quiet these to keep the loop clean.
logging.getLogger("piper").setLevel(logging.ERROR)
# llama.cpp (via llama-cpp-python) emits model-load and runtime detail at INFO;
# keep only warnings and errors.
logging.getLogger("llama-cpp-python").setLevel(logging.WARNING)

# Reuse the web UI's pure helpers and tuned constants so there is a single
# source of truth for transcription, sentence splitting and voice loading.
from app import (
    SYSTEM_PROMPT,
    get_voice,
    split_sentences,
    transcribe_wav,
)

# --- config ---------------------------------------------------------------
ALSA_DEVICE = "plughw:CARD=ArrayUAC10,DEV=0"  # ReSpeaker array: mic + 3.5mm speaker.
                                    # Addressed by stable card NAME, not index,
                                    # since USB re-enumeration changes the number.
VOICE = "de_DE-kerstin-low"         # Piper voice (German female, low/fast)
SAMPLE_RATE = 16000                 # 16 kHz mono S16_LE -> what whisper expects

FRAME_MS = 30                       # VAD analysis frame size
SILENCE_HANG = 0.8                  # stop after this many seconds of silence
START_FRAMES = 3                    # consecutive loud frames needed to start
PREROLL_FRAMES = 5                  # frames kept before onset so we don't clip
MAX_UTTERANCE_S = 15.0              # safety cap on a single utterance
MIN_UTTERANCE_S = 0.3               # ignore blips shorter than this

CALIBRATE_S = 0.5                   # ambient-noise sampling at startup
NOISE_FACTOR = 3.0                  # speech threshold = noise_floor * this ...
MIN_RMS = 250.0                     # ... but never below this absolute RMS
# --------------------------------------------------------------------------

FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 2 bytes/sample, mono


def rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of a little-endian int16 PCM frame."""
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)))


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw 16 kHz/16-bit/mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm)
    return buf.getvalue()


def synth_wav_bytes(voice, text: str) -> bytes:
    """Synthesize ``text`` to WAV bytes (raw form of app.synth_wav_b64)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return buf.getvalue()


class Recorder:
    """Captures single utterances from the mic via RMS voice-activity detection.

    The ReSpeaker is a USB UAC1.0 device that cannot capture and play back at the
    same time, so arecord is opened only while listening and fully released
    before any playback - otherwise the device errors out ("No such device").
    """

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.threshold = self._calibrate()

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            ["arecord", "-q", "-D", ALSA_DEVICE, "-t", "raw",
             "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1"],
            stdout=subprocess.PIPE,
        )

    def _stop(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None

    def _calibrate(self) -> float:
        """Sample ambient noise to set the speech threshold."""
        self.proc = self._spawn()
        try:
            n = max(1, int(CALIBRATE_S * 1000 / FRAME_MS))
            floor = max(rms(self.proc.stdout.read(FRAME_BYTES)) for _ in range(n))
        finally:
            self._stop()
        thr = max(floor * NOISE_FACTOR, MIN_RMS)
        print(f"[vad] noise floor ~{floor:.0f}, speech threshold {thr:.0f}")
        return thr

    def listen(self) -> bytes:
        """Open the mic, block until an utterance is captured, release the mic."""
        self.proc = self._spawn()
        read = self.proc.stdout.read
        try:
            preroll: list[bytes] = []
            loud_run = 0
            # Wait for speech onset, keeping a short rolling pre-roll buffer.
            while True:
                frame = read(FRAME_BYTES)
                if not frame:
                    return b""  # arecord ended (shutdown)
                preroll.append(frame)
                if len(preroll) > PREROLL_FRAMES:
                    preroll.pop(0)
                loud_run = loud_run + 1 if rms(frame) >= self.threshold else 0
                if loud_run >= START_FRAMES:
                    break

            frames = list(preroll)
            silence_run = 0
            max_frames = int(MAX_UTTERANCE_S * 1000 / FRAME_MS)
            hang_frames = int(SILENCE_HANG * 1000 / FRAME_MS)
            while len(frames) < max_frames:
                frame = read(FRAME_BYTES)
                if not frame:
                    break
                frames.append(frame)
                silence_run = silence_run + 1 if rms(frame) < self.threshold else 0
                if silence_run >= hang_frames:
                    break
            return b"".join(frames)
        finally:
            self._stop()  # release the device before playback

    def close(self):
        self._stop()


class Player:
    """Plays WAV chunks through the speaker, in order, on a background thread."""

    def __init__(self):
        self.q: queue.Queue[bytes | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            wav = self.q.get()
            try:
                if wav is None:
                    return
                subprocess.run(
                    ["aplay", "-q", "-D", ALSA_DEVICE],
                    input=wav, stderr=subprocess.DEVNULL,
                )
            finally:
                self.q.task_done()

    def play(self, wav: bytes):
        self.q.put(wav)

    def drain(self):
        """Block until every queued chunk has finished playing."""
        self.q.join()

    def close(self):
        self.q.put(None)


def speak_reply(chat: ChatWrapper, voice, player: Player, text: str):
    """Stream the model's reply for ``text`` and speak it sentence by sentence."""
    buffer = ""
    first = True
    reply = ""
    stream = chat.stream(text)
    try:
        for delta in stream:
            reply += delta
            buffer += delta
            sentences, buffer = split_sentences(buffer, first=first)
            for s in sentences:
                first = False
                player.play(synth_wav_bytes(voice, s))
    finally:
        stream.close()
    tail = buffer.strip()
    if tail:
        player.play(synth_wav_bytes(voice, tail))
    print(f"  < {reply.strip()}")


def main():
    cfg = Config()
    print(f"Loading model: {cfg.model_path}")
    chat = ChatWrapper(cfg)
    chat.begin(SYSTEM_PROMPT)
    voice = get_voice(VOICE)

    player = Player()
    recorder = Recorder()

    # Ctrl-C: tear down arecord/aplay cleanly.
    def shutdown(*_):
        recorder.close()
        player.close()
        print("\nStopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print(f"Voice: {VOICE}  |  Device: {ALSA_DEVICE}")
    print("Listening - speak to the ReSpeaker. Ctrl-C to stop.")
    try:
        while True:
            pcm = recorder.listen()
            if len(pcm) < MIN_UTTERANCE_S * SAMPLE_RATE * 2:
                continue  # too short to be speech
            try:
                text = transcribe_wav(pcm_to_wav(pcm))
            except Exception as e:
                print(f"[stt] error: {e}")
                continue
            if not text:
                continue
            print(f"  > {text}")
            try:
                speak_reply(chat, voice, player, text)
            except ContextOverflowError as e:
                print(f"[llm] {e}")
            # Don't listen again until we've finished speaking (no self-hearing).
            player.drain()
    finally:
        recorder.close()
        player.close()


if __name__ == "__main__":
    main()
