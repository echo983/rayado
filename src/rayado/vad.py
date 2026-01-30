from __future__ import annotations

from typing import List, Tuple

from .models import Segment


def build_speech_segments(
    duration: float,
    silences: List[Tuple[float, float]],
    *,
    pad_sec: float,
    min_speech_sec: float,
    merge_gap_sec: float,
) -> List[Segment]:
    if duration <= 0:
        return []

    silences_sorted = sorted(silences)
    speech: List[Segment] = []
    cursor = 0.0
    for start, end in silences_sorted:
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if start > cursor:
            speech.append(Segment(cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        speech.append(Segment(cursor, duration))

    if not speech:
        speech = [Segment(0.0, duration)]

    padded: List[Segment] = []
    for seg in speech:
        start = max(0.0, seg.start - pad_sec)
        end = min(duration, seg.end + pad_sec)
        if end > start:
            padded.append(Segment(start, end))

    merged: List[Segment] = []
    for seg in sorted(padded, key=lambda s: s.start):
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg.start - prev.end <= merge_gap_sec:
            merged[-1] = Segment(prev.start, max(prev.end, seg.end))
        else:
            merged.append(seg)

    filtered = [seg for seg in merged if (seg.end - seg.start) >= min_speech_sec]
    return filtered
