from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .models import Chunk, Span


def _tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [tok for tok in cleaned.split() if tok]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = sa & sb
    union = sa | sb
    return len(inter) / max(1, len(union))


def _time_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _overlap_window_left(chunk: Chunk) -> Tuple[float, float]:
    return max(0.0, chunk.t1 - chunk.overlap_right), chunk.t1


def _overlap_window_right(chunk: Chunk) -> Tuple[float, float]:
    return chunk.t0, chunk.t0 + chunk.overlap_left


def overlap_judge(
    chunks: List[Chunk],
    spans: List[Span],
    *,
    iou_min: float = 0.30,
    sim_min: float = 0.70,
) -> Tuple[List[Dict[str, str]], List[str]]:
    spans_by_chunk: Dict[str, List[Span]] = {}
    for span in spans:
        spans_by_chunk.setdefault(span.chunk_id, []).append(span)

    overlap_records: List[Dict[str, str]] = []
    suppressed: List[str] = []

    for left, right in zip(chunks, chunks[1:]):
        left_spans = spans_by_chunk.get(left.chunk_id, [])
        right_spans = spans_by_chunk.get(right.chunk_id, [])
        if not left_spans or not right_spans:
            continue

        l0, l1 = _overlap_window_left(left)
        r0, r1 = _overlap_window_right(right)

        for lspan in left_spans:
            if lspan.t1 < l0 or lspan.t0 > l1:
                continue
            for rspan in right_spans:
                if rspan.t1 < r0 or rspan.t0 > r1:
                    continue
                iou = _time_iou(lspan.t0, lspan.t1, rspan.t0, rspan.t1)
                if iou < iou_min:
                    continue
                sim = _jaccard(_tokenize(lspan.text_raw), _tokenize(rspan.text_raw))
                if sim < sim_min:
                    continue

                if lspan.asr_conf >= rspan.asr_conf:
                    decision = "keep_left"
                    suppress_sid = rspan.sid
                else:
                    decision = "keep_right"
                    suppress_sid = lspan.sid

                overlap_records.append(
                    {
                        "olp_id": f"OLP_{lspan.sid}_{rspan.sid}",
                        "left_chunk": left.chunk_id,
                        "right_chunk": right.chunk_id,
                        "left_sid": lspan.sid,
                        "right_sid": rspan.sid,
                        "decision": decision,
                        "method": "time_iou_then_text_jaccard",
                        "conf": f"{sim:.3f}",
                    }
                )
                suppressed.append(suppress_sid)

    return overlap_records, suppressed
