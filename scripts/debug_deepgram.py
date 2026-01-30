from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "docs" / "testfiles" / "test.webm"


def main() -> int:
    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        print("DEEPGRAM_API_KEY is not set", file=sys.stderr)
        return 2

    if not TEST_FILE.exists():
        print(f"Missing test file: {TEST_FILE}", file=sys.stderr)
        return 3

    url = "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true&detect_language=es"
    out_dir = ROOT / "out" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "debug_segment.wav"
    json_path = out_dir / "debug_deepgram.json"

    windows = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0)]
    for start, end in windows:
        print(f"Testing window {start:.1f}-{end:.1f}s")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start}",
            "-to",
            f"{end}",
            "-i",
            str(TEST_FILE),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            str(wav_path),
        ]
        ffmpeg_proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ffmpeg_proc.returncode != 0:
            print(ffmpeg_proc.stderr.decode(errors="ignore"), file=sys.stderr)
            return 4

        curl_cmd = [
            "curl.exe",
            "-s",
            "-o",
            str(json_path),
            "-w",
            "%{http_code}",
            "--max-time",
            "30",
            "--connect-timeout",
            "10",
            "--request",
            "POST",
            "--header",
            f"Authorization: Token {api_key}",
            "--header",
            "Content-Type: audio/wav",
            "--data-binary",
            f"@{wav_path}",
            url,
        ]
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            continue

        status = result.stdout.strip()
        if status != "200":
            print(f"Deepgram HTTP {status}", file=sys.stderr)
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        alt = (payload.get("results", {}).get("channels") or [{}])[0].get("alternatives") or [{}]
        transcript = alt[0].get("transcript", "") if alt else ""
        if transcript:
            output = f"[{start:.1f}-{end:.1f}s] {transcript}\n"
            sys.stdout.buffer.write(output.encode("utf-8", errors="replace"))
            return 0

    print("No transcript returned in 0-20s window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
