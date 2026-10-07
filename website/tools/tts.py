"""Narration for the tutorial video: text -> 24 kHz WAV with Kokoro-82M (Apache-2.0 weights).

    ESPEAK_SHORT=/short/path/to/espeak-ng-data  python tts.py narration.json out_dir [voice] [speed]

narration.json: {"scene_id": "text", ...}. Text may use spoken spellings ("v l a serve") that
read better aloud; SUBTITLE maps them back for the on-screen subtitles. Each sentence is
synthesised separately so subtitles can be timed exactly. Writes out_dir/<scene_id>.wav and
out_dir/timing.json: {scene_id: {"duration": s, "sentences": [[subtitle, start, end], ...]}}.

espeak-ng (Kokoro's fallback for out-of-dictionary words) keeps its data path in a ~160-byte
buffer: a longer path is silently dropped for the library's compiled-in default, which does not
exist on this machine. ESPEAK_SHORT must therefore be a short path (a symlink to the
espeakng_loader package's espeak-ng-data); it is patched in before Kokoro configures espeak.
"""
import json
import os
import re
import sys
from pathlib import Path

import espeakng_loader

if os.environ.get("ESPEAK_SHORT"):
    os.environ["ESPEAK_DATA_PATH"] = os.environ["ESPEAK_SHORT"]   # espeak-ng reads this too
    espeakng_loader.get_data_path = lambda: os.environ["ESPEAK_SHORT"]

import numpy as np                 # noqa: E402
import soundfile as sf             # noqa: E402
from kokoro import KPipeline       # noqa: E402

SR = 24000
GAP = 0.15                          # silence between sentences [s]
SUBTITLE = {"v l a serve": "vlaserve", "K A A I R": "kaair", "Lee-bear-oh": "LIBERO",
            "p s": "ps", "vision language action": "vision-language-action"}


def subtitle(text: str) -> str:
    for spoken, written in SUBTITLE.items():
        text = text.replace(spoken, written)
    return text


def main():
    script = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    voice = sys.argv[3] if len(sys.argv) > 3 else "am_michael"
    speed = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0     # 1.2 = 20 % faster, same pitch
    pipe = KPipeline(lang_code="a")
    timing_file = out / "timing.json"
    timing = json.loads(timing_file.read_text()) if timing_file.exists() else {}
    for sid, text in script.items():
        parts, sents, t = [], [], 0.0
        for s in [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]:
            a = np.concatenate([x for _, _, x in pipe(s, voice=voice, speed=speed)])
            parts += [a, np.zeros(int(GAP * SR), np.float32)]
            sents.append([subtitle(s), round(t, 3), round(t + len(a) / SR, 3)])
            t += len(a) / SR + GAP
        audio = np.concatenate(parts)
        sf.write(out / f"{sid}.wav", audio, SR)
        timing[sid] = {"duration": round(len(audio) / SR, 3), "sentences": sents}
        print(f"{sid}: {timing[sid]['duration']:.1f} s, {len(sents)} sentences")
    timing_file.write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
