from __future__ import annotations

import argparse
import os

from rayado.ffmpeg_tools import extract_audio_file_segment, ffprobe_duration, silencedetect
from rayado.utils import ensure_dir
from rayado.vad import build_speech_segments


def _default_out_dir(input_path: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join("out", "vad_chunks", base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump VAD-based audio chunks to a directory")
    parser.add_argument("input", help="Path to local media file")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory")
    parser.add_argument("--prefix", default=None, help="Filename prefix (default: base name)")
    parser.add_argument("--vad-threshold", type=float, default=-30.0, help="VAD noise threshold in dB")
    parser.add_argument("--vad-min-speech-sec", type=float, default=0.6, help="Min speech segment length")
    parser.add_argument("--vad-merge-gap-sec", type=float, default=0.3, help="Merge gap length")
    parser.add_argument("--vad-pad-sec", type=float, default=0.1, help="Pad speech segments by seconds")
    parser.add_argument("--vad-min-silence-sec", type=float, default=0.5, help="Min silence length")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Output channel count")

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise SystemExit(f"Input not found: {input_path}")

    out_dir = args.out_dir or _default_out_dir(input_path)
    ensure_dir(out_dir)

    prefix = args.prefix or os.path.splitext(os.path.basename(input_path))[0]

    duration = ffprobe_duration(input_path)
    silences = silencedetect(
        input_path,
        noise_db=args.vad_threshold,
        min_silence=args.vad_min_silence_sec,
    )
    speech_segments = build_speech_segments(
        duration,
        silences,
        pad_sec=args.vad_pad_sec,
        min_speech_sec=args.vad_min_speech_sec,
        merge_gap_sec=args.vad_merge_gap_sec,
    )

    kept = 0
    for seg in speech_segments:
        start = seg.start
        end = seg.end
        name = f"{prefix}_{start:.3f}.wav"
        out_path = os.path.join(out_dir, name)
        extract_audio_file_segment(
            input_path,
            out_path,
            start=start,
            end=end,
            sample_rate=args.sample_rate,
            channels=args.channels,
        )
        kept += 1

    print(f"speech_segments={len(speech_segments)} kept={kept} out_dir={out_dir}")


if __name__ == "__main__":
    main()
