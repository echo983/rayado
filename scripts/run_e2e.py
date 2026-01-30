from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "docs" / "testfiles" / "test.webm"
OUT_DIR = ROOT / "out" / "test"


def main() -> int:
    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        print("DEEPGRAM_API_KEY is not set", file=sys.stderr)
        return 2

    if not TEST_FILE.exists():
        print(f"Missing test file: {TEST_FILE}", file=sys.stderr)
        return 3

    cmd = [
        sys.executable,
        "-m",
        "rayado",
        "transcribe",
        str(TEST_FILE),
        "--out",
        str(OUT_DIR),
        "--chunk-sec",
        "10",
        "--overlap-sec",
        "1.0",
        "--no-deepgram-diarize",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"Transcription failed with code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    gcl = OUT_DIR / "episode.gcl"
    txt = OUT_DIR / "transcript.txt"
    srt = OUT_DIR / "subtitles.srt"

    if not gcl.exists() or gcl.stat().st_size == 0:
        print(f"Missing or empty output: {gcl}", file=sys.stderr)
        return 4
    for path in (txt, srt):
        if not path.exists():
            print(f"Missing output: {path}", file=sys.stderr)
            return 4

    gcl_text = gcl.read_text(encoding="utf-8", errors="ignore")
    if "GCL_SPAN" not in gcl_text:
        print("Warning: no GCL_SPAN entries found", file=sys.stderr)

    print("E2E test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
