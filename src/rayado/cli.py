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
    argv = sys.argv[1:]
    if argv and argv[0] == "transcribe":
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="Rayado CLI transcription pipeline")
    parser.add_argument("input", help="Path to local media file")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory")
    parser.add_argument("--concurrency", type=int, default=64, help="Concurrency (reserved)")
    parser.add_argument("--chunk-sec", type=float, default=25.0, help="Chunk length in seconds")
    parser.add_argument("--overlap-sec", type=float, default=1.5, help="Overlap on each side in seconds")
    parser.add_argument("--retry", type=int, default=1, help="Retry count (max 1)")
    parser.add_argument("--asr-provider", default="deepgram", help="ASR provider (deepgram/mock/noop)")
    parser.add_argument("--deepgram-model", default="nova-2", help="Deepgram model name")
    parser.add_argument("--deepgram-language", default="", help="Deepgram language hint (e.g. es)")
    parser.add_argument(
        "--deepgram-diarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Deepgram diarization",
    )
    parser.add_argument(
        "--deepgram-smart-format",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Deepgram smart formatting",
    )
    parser.add_argument(
        "--deepgram-punctuate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Deepgram punctuation",
    )
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

    args = parser.parse_args(argv)

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
        deepgram_model=args.deepgram_model,
        deepgram_language=args.deepgram_language,
        deepgram_diarize=args.deepgram_diarize,
        deepgram_smart_format=args.deepgram_smart_format,
        deepgram_punctuate=args.deepgram_punctuate,
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
