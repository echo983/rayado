from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "docs" / "testfiles" / "test.webm"
OUT_SRT = ROOT / "out" / "test.srt"


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
        "phase1",
        str(TEST_FILE),
        "--out",
        str(OUT_SRT),
        "--target-sec",
        "10",
        "--max-sec",
        "20",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"Transcription failed with code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    if not OUT_SRT.exists() or OUT_SRT.stat().st_size == 0:
        print(f"Missing or empty output: {OUT_SRT}", file=sys.stderr)
        return 4

    print("E2E test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
