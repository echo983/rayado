from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Segment:
    start: float
    end: float


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    t0: float
    t1: float
    overlap_left: float
    overlap_right: float
    skip_reason: Optional[str] = None


@dataclass(frozen=True)
class Span:
    sid: str
    t0: float
    t1: float
    chunk_id: str
    text_raw: str
    asr_conf: float
