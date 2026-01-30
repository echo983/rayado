from __future__ import annotations

from typing import Iterable, List

from .models import Chunk, Segment


def generate_chunks(
    duration: float,
    *,
    chunk_sec: float,
    overlap_sec: float,
) -> List[Chunk]:
    step = chunk_sec - 2 * overlap_sec
    if step <= 0:
        raise ValueError("chunk_sec must be greater than 2 * overlap_sec")

    chunks: List[Chunk] = []
    idx = 1
    t0 = 0.0
    while t0 < duration:
        t1 = min(t0 + chunk_sec, duration)
        chunk_id = f"C{idx:05d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                t0=round(t0, 3),
                t1=round(t1, 3),
                overlap_left=overlap_sec,
                overlap_right=overlap_sec,
            )
        )
        idx += 1
        t0 += step
    return chunks


def chunk_has_speech(chunk: Chunk, speech_segments: Iterable[Segment]) -> bool:
    for seg in speech_segments:
        if seg.end <= chunk.t0:
            continue
        if seg.start >= chunk.t1:
            continue
        return True
    return False
