#!/usr/bin/env python3
"""Synthesize a sample line with every voice in piper/voices/.

Writes one WAV per voice into piper/output/. Picks the sample line by
language (en_* vs de_*) so each voice speaks text it can pronounce.
"""

import wave
from pathlib import Path

from piper import PiperVoice

VOICES_DIR = Path(__file__).parent.parent / "voices"
OUTPUT_DIR = Path(__file__).parent / "output"

# Sample line per language prefix; falls back to DEFAULT_LINE.
LINES = {
    "en": "Hello world, this is a test of the Piper text to speech voice.",
    "de": "Hallo Welt, dies ist ein Test der Piper Sprachausgabe.",
}
DEFAULT_LINE = LINES["en"]


def line_for(model_name: str) -> str:
    lang = model_name.split("_", 1)[0].lower()
    return LINES.get(lang, DEFAULT_LINE)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    models = sorted(VOICES_DIR.glob("*.onnx"))
    if not models:
        raise SystemExit(f"No .onnx voices found in {VOICES_DIR}")

    for model_path in models:
        name = model_path.stem  # e.g. de_DE-thorsten-high
        text = line_for(name)
        out_path = OUTPUT_DIR / f"{name}.wav"
        print(f"[{name}] -> {out_path.name}: {text!r}")

        voice = PiperVoice.load(model_path)
        with wave.open(str(out_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

    print(f"\nDone. {len(models)} file(s) written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
