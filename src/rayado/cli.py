from __future__ import annotations

import argparse
import os
import sys

from .phase1 import run_phase1
from .phase2 import run_phase2
from .utils import ensure_dir


def _default_srt_path(input_path: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join("out", f"{base}.srt")


def _default_graph_path(srt_path: str) -> str:
    base = os.path.splitext(os.path.basename(srt_path))[0]
    return os.path.join("out", f"{base}.graph.txt")


def _default_clean_srt_path(srt_path: str) -> str:
    base = os.path.splitext(os.path.basename(srt_path))[0]
    return os.path.join("out", f"{base}.clean.srt")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "transcribe":
        argv = ["phase1", *argv[1:]]

    parser = argparse.ArgumentParser(description="Rayado CLI refactor pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("phase1", help="Transcription phase (audio -> SRT)")
    p1.add_argument("input", help="Path to local media file")
    p1.add_argument("--out", dest="out_srt", default=None, help="Output SRT path")
    p1.add_argument("--concurrency", type=int, default=64, help="Concurrency")
    p1.add_argument("--retry", type=int, default=1, help="Retry count (max 1)")
    p1.add_argument("--deepgram-model", default="nova-2", help="Deepgram model name")
    p1.add_argument(
        "--cache-dir",
        default=os.path.join(".cache", "rayado"),
        help="Cache directory",
    )
    p1.add_argument("--vad-threshold", type=float, default=-30.0, help="VAD noise threshold in dB")
    p1.add_argument("--vad-min-speech-sec", type=float, default=0.6, help="Min speech segment length")
    p1.add_argument("--vad-merge-gap-sec", type=float, default=0.3, help="Merge gap length")
    p1.add_argument("--vad-pad-sec", type=float, default=0.1, help="Pad speech segments by seconds")
    p1.add_argument("--vad-min-silence-sec", type=float, default=0.5, help="Min silence length")
    p1.add_argument("--target-sec", type=float, default=20.0, help="Target segment length")
    p1.add_argument("--max-sec", type=float, default=35.0, help="Max segment length")
    p1.add_argument(
        "--lid-cache-dir",
        default=os.path.join(".cache", "speechbrain", "lang-id-voxlingua107-ecapa"),
        help="SpeechBrain model cache directory",
    )
    p1.add_argument("--lid-device", default="cpu", help="LID device (cpu/cuda)")

    p2 = subparsers.add_parser("phase2", help="Logic modeling phase (SRT -> graph -> cleaned SRT)")
    p2.add_argument("srt", help="Input SRT path")
    p2.add_argument("--prompt", default=os.path.join("prompts", "SORAL.txt"), help="Prompt file path")
    p2.add_argument("--graph-in", default=None, help="External object graph file")
    p2.add_argument("--graph-out", default=None, help="Output object graph file")
    p2.add_argument("--out", dest="out_srt", default=None, help="Output cleaned SRT path")
    p2.add_argument("--model-graph", default="gpt-5.2", help="Model for object graph")
    p2.add_argument("--model-clean", default="gpt-5-mini", help="Model for SRT rebuild")
    p2.add_argument("--chunk-chars", type=int, default=12000, help="Approx chunk size in chars")
    p2.add_argument("--prompt-cache-retention", default="24h", help="Prompt cache retention")
    p2.add_argument("--retry", type=int, default=1, help="Retry count (max 1)")
    p2.add_argument("--start-chunk", type=int, default=1, help="Start chunk index (1-based)")
    p2.add_argument("--max-chunks", type=int, default=None, help="Max chunks to process")

    args = parser.parse_args(argv)

    if args.command == "phase1":
        input_path = args.input
        if not os.path.exists(input_path):
            print(f"Input not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        if args.retry > 1:
            print("Retry is capped at 1; using 1.", file=sys.stderr)
            args.retry = 1
        if args.concurrency < 1:
            print("Concurrency must be >= 1", file=sys.stderr)
            sys.exit(1)
        out_srt_path = args.out_srt or _default_srt_path(input_path)
        ensure_dir(os.path.dirname(out_srt_path))

        detected = run_phase1(
            input_path=input_path,
            out_srt_path=out_srt_path,
            cache_dir=args.cache_dir,
            concurrency=args.concurrency,
            retry=args.retry,
            deepgram_model=args.deepgram_model,
            vad_threshold=args.vad_threshold,
            vad_min_speech_sec=args.vad_min_speech_sec,
            vad_merge_gap_sec=args.vad_merge_gap_sec,
            vad_pad_sec=args.vad_pad_sec,
            vad_min_silence_sec=args.vad_min_silence_sec,
            target_sec=args.target_sec,
            max_sec=args.max_sec,
            lid_cache_dir=args.lid_cache_dir,
            lid_device=args.lid_device,
        )
        print(f"Language={detected} Output={out_srt_path}")

    if args.command == "phase2":
        srt_path = args.srt
        if not os.path.exists(srt_path):
            print(f"SRT not found: {srt_path}", file=sys.stderr)
            sys.exit(1)
        if args.retry > 1:
            print("Retry is capped at 1; using 1.", file=sys.stderr)
            args.retry = 1

        graph_out = args.graph_out or _default_graph_path(srt_path)
        out_clean = args.out_srt or _default_clean_srt_path(srt_path)
        ensure_dir(os.path.dirname(graph_out))
        ensure_dir(os.path.dirname(out_clean))

        run_phase2(
            srt_path=srt_path,
            prompt_path=args.prompt,
            graph_in_path=args.graph_in,
            graph_out_path=graph_out,
            cleaned_srt_path=out_clean,
            model_graph=args.model_graph,
            model_clean=args.model_clean,
            chunk_chars=args.chunk_chars,
            prompt_cache_retention=args.prompt_cache_retention,
            retry=args.retry,
            start_chunk=args.start_chunk,
            max_chunks=args.max_chunks,
        )
        print(f"Output={out_clean}")


if __name__ == "__main__":
    main()
