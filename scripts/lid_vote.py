from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from typing import Dict, List, Tuple


def _list_chunks(dir_path: str) -> List[str]:
    files = []
    for name in os.listdir(dir_path):
        if name.lower().endswith(".wav"):
            files.append(name)
    return files


def _extract_start_seconds(name: str) -> float | None:
    # Expect format: <prefix>_<start>.wav
    m = re.search(r"_([0-9]+(?:\.[0-9]+)?)\.wav$", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _pick_samples(files: List[str]) -> List[str]:
    if not files:
        return []
    indexed: List[Tuple[float, str]] = []
    for name in files:
        start = _extract_start_seconds(name)
        if start is None:
            start = float("inf")
        indexed.append((start, name))
    indexed.sort(key=lambda x: (x[0], x[1]))
    ordered = [name for _, name in indexed]
    n = len(ordered)
    positions = [0, int(n * 0.25), int(n * 0.50), int(n * 0.75), n - 1]
    chosen = []
    seen = set()
    for pos in positions:
        pos = max(0, min(n - 1, pos))
        name = ordered[pos]
        if name in seen:
            continue
        seen.add(name)
        chosen.append(name)
    return chosen


def _detect_language(path: str) -> Tuple[str, float]:
    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")

    with open(path, "rb") as f:
        audio_bytes = f.read()

    url = "https://api.deepgram.com/v1/listen?detect_language=true&model=nova-2"
    req = urllib.request.Request(
        url=url,
        data=audio_bytes,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    channel = (payload.get("results", {}).get("channels") or [{}])[0]
    lang = channel.get("detected_language") or ""
    conf = channel.get("language_confidence")
    try:
        conf_val = float(conf) if conf is not None else 0.0
    except (TypeError, ValueError):
        conf_val = 0.0
    return lang, conf_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Vote dominant language using Deepgram detect_language")
    parser.add_argument("chunk_dir", help="Directory of WAV chunks")
    args = parser.parse_args()

    chunk_dir = args.chunk_dir
    if not os.path.isdir(chunk_dir):
        print(f"Chunk dir not found: {chunk_dir}", file=sys.stderr)
        sys.exit(1)

    files = _list_chunks(chunk_dir)
    samples = _pick_samples(files)
    if not samples:
        print("No wav chunks found.", file=sys.stderr)
        sys.exit(2)

    votes: Dict[str, int] = {}
    weights: Dict[str, float] = {}
    details = []
    for name in samples:
        path = os.path.join(chunk_dir, name)
        lang, conf = _detect_language(path)
        votes[lang] = votes.get(lang, 0) + 1
        if conf >= 0.3:
            weights[lang] = weights.get(lang, 0.0) + conf
        details.append({"file": name, "language": lang, "confidence": conf})

    winner_weight = ""
    if weights:
        winner_weight = sorted(weights.items(), key=lambda x: (-x[1], x[0]))[0][0]
    winner_count = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0] if votes else ""
    winner = winner_weight or winner_count

    output = {
        "sample_count": len(samples),
        "samples": details,
        "votes": votes,
        "weights": weights,
        "winner": winner,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
