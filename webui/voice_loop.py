#!/usr/bin/env python3
"""Push-to-talk voice assistant on the Pi's ReSpeaker mic + speaker.

Same pipeline as ``webui/app.py`` (whisper STT -> llama_chat LLM -> Piper TTS),
but the browser's microphone and speaker are replaced by the local ReSpeaker
4-Mic Array: ``arecord`` captures from the array and ``aplay`` plays back through
the speaker on its 3.5 mm jack (both ALSA card 2 -> ``plughw:2,0``).

Interaction is push-to-talk via a momentary button on a GPIO pin:

* Hold the button to record; release to transcribe and get a spoken reply.
* Pressing the button at any time interrupts whatever is happening - it stops
  the model's generation and the speaker instantly - and immediately starts
  recording the next utterance (so a held press both barges in and begins the
  new turn). A quick tap during a reply just stops it and returns to idle.

The LED ring mirrors the state: pulsing green while idle, pulsing blue while
the button is held (recording), spinning green while transcribing/thinking,
and steady green while the reply is read out.

It never records while not pressed, so it cannot transcribe its own output.

Requires whisper-server running on 127.0.0.1:8081 (see ``voice-start.sh``).

Run with the piper venv, ``llama_chat`` importable, and the system ``gpiozero``
on the path (same as ``voice-start.sh``):
    PYTHONPATH=faster-llama-chat:<llama-site>:/usr/lib/python3/dist-packages \
        piper/venv/bin/python webui/voice_loop.py
"""

import io
import logging
import queue
import signal
import subprocess
import sys
import threading
import time
import wave

import numpy as np
from gpiozero import Button, Device
from gpiozero.pins.lgpio import LGPIOFactory

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
from led_controller import LEDController

# --- config ---------------------------------------------------------------
ALSA_DEVICE = "plughw:CARD=ArrayUAC10,DEV=0"  # ReSpeaker array speaker (3.5mm jack).
                                    # plughw resamples Piper's 22.05 kHz WAV to the
                                    # device. Addressed by stable card NAME, not
                                    # index, since USB re-enumeration changes it.
CAPTURE_DEVICE = "hw:CARD=ArrayUAC10,DEV=0"  # capture the array's *native* streams
                                    # (no plug layer) so the 6 channels reach us
                                    # un-mixed; we then keep only the ASR channel.
VOICE = "de_DE-kerstin-low"         # Piper voice (German female, low/fast)
SAMPLE_RATE = 16000                 # 16 kHz S16_LE -> the array's native rate / what
                                    # whisper expects

# The XVF3000 array's 6-channel firmware exposes: channel 0 = DSP-processed audio
# (echo-cancel + beamform + noise-suppress + auto-gain) meant for ASR; channels
# 1-4 = the raw mic capsules; channel 5 = the playback loopback reference. Feeding
# whisper the processed channel alone is far cleaner than ALSA's downmix of all 6.
CAPTURE_CHANNELS = 6                # native channel count of the 6-ch firmware
ASR_CHANNEL = 0                     # the DSP-processed channel to transcribe

BUTTON_PIN = 17                     # BCM GPIO17 (pin 11); button to GND, pull-up
BOUNCE_S = 0.05                     # debounce window for the mechanical button

# Typed stdin lines are added to the conversation as context (no reply is
# generated) via ChatWrapper.inject. Each line is injected with INJECT_ROLE
# ("user", "assistant", or "system") and its content prefixed by INJECT_PREFIX.
INJECT_ROLE = "system"
INJECT_PREFIX = "Received sensor data: "

FRAME_MS = 30                       # mic read granularity (also release polling)
MAX_UTTERANCE_S = 15.0              # safety cap on a single utterance
MIN_UTTERANCE_S = 0.3               # ignore taps shorter than this (pure interrupt)

# Noise filter (spectral subtraction + high-pass) applied to the ASR channel
# before transcription. Channel 0 is already DSP-denoised, so this stays gentle.
NOISE_WIN = 512                     # STFT window (32 ms @ 16 kHz)
NOISE_HOP = 128                     # 75% overlap
NOISE_HIGHPASS_HZ = 100.0           # attenuate everything below this
NOISE_PCTL = 20.0                   # per-bin noise floor = this percentile over time
NOISE_ALPHA = 1.5                   # noise over-subtraction factor (higher = stronger)
NOISE_FLOOR = 0.12                  # keep at least this fraction of each bin (anti-artifact)
# --------------------------------------------------------------------------

# One read = one FRAME_MS slice across all captured channels (interleaved).
CAPTURE_FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * CAPTURE_CHANNELS * 2


def extract_asr_channel(raw: bytes) -> bytes:
    """Deinterleave the captured frames and return only the ASR channel as PCM."""
    samples = np.frombuffer(raw, dtype=np.int16)
    n = (len(samples) // CAPTURE_CHANNELS) * CAPTURE_CHANNELS  # drop partial frame
    return samples[:n].reshape(-1, CAPTURE_CHANNELS)[:, ASR_CHANNEL].tobytes()


def denoise(pcm: bytes) -> bytes:
    """Light spectral-subtraction denoise + high-pass on mono S16_LE PCM.

    Estimates a per-frequency noise floor from the quiet parts of the clip and
    subtracts it (with a spectral floor to avoid musical-noise artifacts), then
    drops everything below ``NOISE_HIGHPASS_HZ``. Operates on the whole utterance,
    so call it once on the full recording rather than per frame.
    """
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) < NOISE_WIN:
        return pcm

    win = np.hanning(NOISE_WIN).astype(np.float32)
    n_frames = 1 + (len(x) - NOISE_WIN) // NOISE_HOP
    idx = np.arange(NOISE_WIN)[None, :] + NOISE_HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * win                       # (n_frames, NOISE_WIN)
    spec = np.fft.rfft(frames, axis=1)
    mag, phase = np.abs(spec), np.angle(spec)

    noise = np.percentile(mag, NOISE_PCTL, axis=0)          # per-bin noise floor
    clean = np.maximum(mag - NOISE_ALPHA * noise[None, :], NOISE_FLOOR * mag)
    clean[:, np.fft.rfftfreq(NOISE_WIN, 1.0 / SAMPLE_RATE) < NOISE_HIGHPASS_HZ] = 0.0

    rec = np.fft.irfft(clean * np.exp(1j * phase), n=NOISE_WIN, axis=1).astype(np.float32) * win
    out = np.zeros(NOISE_HOP * (n_frames - 1) + NOISE_WIN, dtype=np.float32)
    wsum = np.zeros_like(out)
    w2 = win * win
    for i in range(n_frames):                   # overlap-add with window normalization
        s = i * NOISE_HOP
        out[s:s + NOISE_WIN] += rec[i]
        wsum[s:s + NOISE_WIN] += w2
    out = (out / np.maximum(wsum, 1e-6))[: len(x)]
    return np.clip(out * 32768.0, -32768, 32767).astype(np.int16).tobytes()


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
    """Captures one utterance from the mic for as long as the button is held.

    The ReSpeaker is a USB UAC1.0 device that cannot capture and play back at the
    same time, so arecord is opened only while recording and fully released
    before any playback - otherwise the device errors out ("No such device").
    """

    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            ["arecord", "-q", "-D", CAPTURE_DEVICE, "-t", "raw",
             "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", str(CAPTURE_CHANNELS)],
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

    def record_while_pressed(self, button: Button) -> tuple[bytes, float]:
        """Capture while the button is held.

        Returns:
            Mono PCM of the ASR channel, and the ``time.monotonic()`` instant of
            the button release (the reference point for response-delay timing).
        """
        self.proc = self._spawn()
        read = self.proc.stdout.read
        try:
            frames: list[bytes] = []
            max_frames = int(MAX_UTTERANCE_S * 1000 / FRAME_MS)
            # is_pressed is polled once per ~30 ms frame, so release stops us
            # within one frame.
            while button.is_pressed and len(frames) < max_frames:
                frame = read(CAPTURE_FRAME_BYTES)
                if not frame:
                    break  # arecord ended (shutdown)
                frames.append(frame)
            released_at = time.monotonic()
            return denoise(extract_asr_channel(b"".join(frames))), released_at
        finally:
            self._stop()  # release the device before playback

    def close(self):
        self._stop()


class Player:
    """Plays WAV chunks through the speaker, in order, on a background thread.

    Each chunk is played by a killable ``aplay`` subprocess so ``stop`` can halt
    playback instantly (barge-in) and discard anything still queued.
    """

    def __init__(self):
        self.q: queue.Queue[bytes | None] = queue.Queue()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            wav = self.q.get()
            if wav is None:
                self.q.task_done()
                return
            with self._lock:
                self._proc = subprocess.Popen(
                    ["aplay", "-q", "-D", ALSA_DEVICE],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
            proc = self._proc
            try:
                proc.communicate(input=wav)  # blocks until played or killed
            except (ValueError, OSError):
                pass  # killed mid-play by stop()
            with self._lock:
                self._proc = None
            self.q.task_done()

    def play(self, wav: bytes):
        self.q.put(wav)

    def stop(self):
        """Halt current playback and drop anything queued (barge-in)."""
        with self._lock:
            while True:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
                self.q.task_done()
            if self._proc is not None:
                self._proc.kill()

    def wait_done(self):
        """Block until everything queued has been played (or dropped by stop)."""
        self.q.join()

    def close(self):
        self.stop()
        self.q.put(None)


def speak_reply(chat: ChatWrapper, voice, player: Player, text: str,
                interrupted: threading.Event, leds: LEDController,
                released_at: float, denoise_s: float, stt_s: float):
    """Stream the model's reply for ``text`` and speak it sentence by sentence.

    Stops early if ``interrupted`` is set (the button was pressed): the stream is
    closed so the wrapper records exactly the tokens that reached the cache, and
    no further audio is queued.

    When the first audio chunk is queued, ``leds`` switches from the processing
    animation to a steady color for the read-out, and a ``[timing]`` table is
    printed breaking down TTFA - the total delay from the button release at
    ``released_at`` until the first audio chunk is handed to the player - into
    ``denoise_s`` (denoise + mic close), ``stt_s`` (whisper), the LLM's delay to
    its first token, the first sentence's TTS synthesis, and the unaccounted
    remainder (sentence splitting, WAV wrapping, ...).
    """
    buffer = ""
    first = True
    reply = ""
    llm_start = time.monotonic()
    first_token_s = 0.0
    stream = chat.stream(text)

    def play_first(sentence: str):
        """Synthesize + queue the first chunk, then print the timing table."""
        tts_start = time.monotonic()
        wav = synth_wav_bytes(voice, sentence)
        tts_s = time.monotonic() - tts_start
        player.play(wav)
        ttfa_s = time.monotonic() - released_at
        rows = [
            ("denoise + mic close", denoise_s),
            ("whisper", stt_s),
            ("llm first token", first_token_s),
            ("tts first sentence", tts_s),
        ]
        rows.append(("other", ttfa_s - sum(secs for _, secs in rows)))
        rows.append(("ttfa (total)", ttfa_s))
        print("[timing]")
        for name, secs in rows:
            print(f"  {name:<19} {secs:5.2f}s")

    try:
        for delta in stream:
            if interrupted.is_set():
                break
            if not reply:
                first_token_s = time.monotonic() - llm_start
            reply += delta
            buffer += delta
            sentences, buffer = split_sentences(buffer, first=first)
            for s in sentences:
                if interrupted.is_set():
                    break
                if first:
                    leds.stop_animation()  # thinking -> speaking
                    leds.ring_on(leds.color_green())
                    play_first(s)
                else:
                    player.play(synth_wav_bytes(voice, s))
                first = False
    finally:
        stream.close()
    if not interrupted.is_set():
        tail = buffer.strip()
        if tail:
            if first:
                leds.stop_animation()
                leds.ring_on(leds.color_green())
                play_first(tail)
            else:
                player.play(synth_wav_bytes(voice, tail))
    print(f"  < {reply.strip()}")


def stdin_inject_loop(chat: ChatWrapper) -> None:
    """Inject each typed stdin line as conversation context (no reply)."""
    for line in sys.stdin:                 # ends cleanly on EOF (e.g. no tty)
        text = line.strip()
        if not text:
            continue
        message = INJECT_PREFIX + text
        try:
            chat.inject(message, INJECT_ROLE)   # blocks if a turn is streaming
            print(f"[inject] {INJECT_ROLE}: {message}")
        except Exception as e:             # e.g. ValueError if it can't fit
            print(f"[inject] error: {e}")


def main():
    cfg = Config()
    print(f"Loading model: {cfg.model_path}")
    chat = ChatWrapper(cfg)
    chat.begin(SYSTEM_PROMPT)
    voice = get_voice(VOICE)

    player = Player()
    recorder = Recorder()
    leds = LEDController()
    interrupted = threading.Event()

    Device.pin_factory = LGPIOFactory()
    button = Button(BUTTON_PIN, pull_up=True, bounce_time=BOUNCE_S)

    # Any press barges in: stop the speaker and the model's generation. The same
    # press then starts the next recording (if it's a hold) back in the loop.
    def on_press():
        interrupted.set()
        player.stop()

    button.when_pressed = on_press

    # Ctrl-C: tear down arecord/aplay and release the GPIO line cleanly.
    def shutdown(*_):
        recorder.close()
        player.close()
        leds.cleanup()
        button.close()
        print("\nStopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # Read typed lines from the terminal and inject them as context (no reply).
    threading.Thread(target=stdin_inject_loop, args=(chat,), daemon=True).start()

    print(f"Voice: {VOICE}  |  Device: {ALSA_DEVICE}  |  Button: GPIO{BUTTON_PIN}")
    print("Push-to-talk: hold the button to talk, release for a reply.")
    print(f"Type a line to inject context as {INJECT_ROLE!r} (prefix {INJECT_PREFIX!r}).")
    print("Press the button any time to interrupt and start a new turn. Ctrl-C to stop.")
    try:
        while True:
            leds.pulse_ring(leds.color_green(), duration=float("inf"))   # idle
            # Idle until pressed; the 1 s timeout keeps Ctrl-C responsive.
            while not button.wait_for_press(timeout=1.0):
                pass
            interrupted.clear()      # consume the press that woke us
            player.stop()            # silence any leftover reply still playing
            leds.pulse_ring(leds.color_blue())    # recording (button held)
            pcm, released_at = recorder.record_while_pressed(button)
            denoise_s = time.monotonic() - released_at
            if len(pcm) < MIN_UTTERANCE_S * SAMPLE_RATE * 2:
                continue  # too short to be speech (e.g. a tap to just interrupt)
            leds.spin_ring(leds.color_green())    # transcribing/thinking
            stt_start = time.monotonic()
            try:
                text = transcribe_wav(pcm_to_wav(pcm))
            except Exception as e:
                print(f"[stt] error: {e}")
                continue
            stt_s = time.monotonic() - stt_start
            if not text:
                continue
            print(f"  > {text}")
            try:
                speak_reply(chat, voice, player, text, interrupted, leds,
                            released_at, denoise_s, stt_s)
            except ContextOverflowError as e:
                print(f"[llm] {e}")
            # Hold the steady color until the reply has been fully spoken (or
            # stopped by a barge-in); then loop back to the idle pulse.
            player.wait_done()
    finally:
        recorder.close()
        player.close()
        leds.cleanup()
        button.close()


if __name__ == "__main__":
    main()
