from __future__ import annotations

import argparse
import os
import sys

from .pipeline import run_pipeline
from .utils import ensure_dir


def _default_out_dir(input_path: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join("out", base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rayado CLI transcription pipeline")
    parser.add_argument("input", help="Path to local media file")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory")
    parser.add_argument("--concurrency", type=int, default=64, help="Concurrency (reserved)")
    parser.add_argument("--chunk-sec", type=float, default=25.0, help="Chunk length in seconds")
    parser.add_argument("--overlap-sec", type=float, default=1.5, help="Overlap on each side in seconds")
    parser.add_argument("--retry", type=int, default=1, help="Retry count (max 1)")
    parser.add_argument("--asr-provider", default="deepgram", help="ASR provider (deepgram/mock/noop)")
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(".cache", "rayado"),
        help="Cache directory",
    )
    parser.add_argument("--vad", default="ffmpeg_silence", help="VAD backend (ffmpeg_silence/none)")
    parser.add_argument("--vad-threshold", type=float, default=-30.0, help="VAD noise threshold in dB")
    parser.add_argument("--vad-min-speech-sec", type=float, default=0.6, help="Min speech segment length")
    parser.add_argument("--vad-merge-gap-sec", type=float, default=0.3, help="Merge gap length")
    parser.add_argument("--vad-pad-sec", type=float, default=0.1, help="Pad speech segments by seconds")

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.retry > 1:
        print("Retry is capped at 1; using 1.", file=sys.stderr)
        args.retry = 1

    out_dir = args.out_dir or _default_out_dir(input_path)
    ensure_dir(out_dir)

    if args.concurrency != 128:
        print("Note: concurrency is reserved in MVP; execution is sequential.", file=sys.stderr)

    run_pipeline(
        input_path=input_path,
        out_dir=out_dir,
        cache_dir=args.cache_dir,
        provider=args.asr_provider,
        retry=args.retry,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        vad_name=args.vad,
        vad_threshold=args.vad_threshold,
        vad_min_speech_sec=args.vad_min_speech_sec,
        vad_merge_gap_sec=args.vad_merge_gap_sec,
        vad_pad_sec=args.vad_pad_sec,
    )


if __name__ == "__main__":
    main()
