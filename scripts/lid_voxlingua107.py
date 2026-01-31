from __future__ import annotations

import argparse
import json
import os
import sys

from rayado.lid_voxlingua import detect_language_voxlingua


def _list_wavs(dir_path: str) -> list[str]:
    return [
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if name.lower().endswith(".wav")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Vote dominant language using VoxLingua107 (SpeechBrain)")
    parser.add_argument("chunk_dir", help="Directory of WAV chunks")
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(".cache", "speechbrain", "lang-id-voxlingua107-ecapa"),
        help="Model cache directory",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu/cuda)")
    args = parser.parse_args()

    if not os.path.isdir(args.chunk_dir):
        print(f"Chunk dir not found: {args.chunk_dir}", file=sys.stderr)
        sys.exit(1)

    wavs = _list_wavs(args.chunk_dir)
    if not wavs:
        print("No wav chunks found.", file=sys.stderr)
        sys.exit(2)

    winner, weights, samples = detect_language_voxlingua(
        wavs,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    output = {
        "sample_count": len(samples),
        "samples": [
            {
                "file": os.path.basename(sample.path),
                "language": sample.language,
                "score": sample.score,
                "label": sample.label,
            }
            for sample in samples
        ],
        "weights": weights,
        "winner": winner,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
