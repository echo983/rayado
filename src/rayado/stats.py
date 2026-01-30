from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class RunStats:
    started_at: float
    ended_at: float
    duration_sec: float
    input_path: str
    output_dir: str
    provider: str
    chunk_count: int
    chunk_skipped: int
    chunk_processed: int
    chunk_failed: int
    span_count: int
    suppressed_count: int


def write_run_log(path: str, stats: RunStats, extra: Dict[str, Any] | None = None) -> None:
    payload = asdict(stats)
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def now() -> float:
    return time.time()
