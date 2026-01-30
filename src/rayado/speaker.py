from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple


def summarize_speaker(words: List[dict]) -> Tuple[str, float]:
    speakers = [w.get("speaker") for w in words if w.get("speaker") is not None]
    if not speakers:
        return "", 0.0
    counts = Counter(speakers)
    speaker = counts.most_common(1)[0][0]
    confs = [float(w.get("speaker_confidence", 0.0)) for w in words if w.get("speaker") == speaker]
    conf = sum(confs) / len(confs) if confs else 0.0
    return str(speaker), conf


def build_speaker_blocks(
    *,
    chunk_id: str,
    span_id: str,
    words: List[dict],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    speaker, conf = summarize_speaker(words)
    if not speaker:
        return {}, {}

    spk_id = f"SPK_{chunk_id}_{speaker}"
    speaker_block = {
        "spk_id": spk_id,
        "label": f"Speaker_{speaker}",
        "conf": f"{conf:.3f}" if conf else "",
    }
    speaker_map_block = {
        "spk_id": spk_id,
        "sid": span_id,
        "conf": f"{conf:.3f}" if conf else "",
    }
    return speaker_block, speaker_map_block
