from __future__ import annotations

import re
from typing import List, Tuple

from .utils import run_cmd


def ffprobe_duration(input_path: str) -> float:
    proc = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            input_path,
        ],
        capture_stderr=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or "ffprobe failed")
    value = (proc.stdout or "").strip()
    if not value:
        raise RuntimeError("ffprobe returned empty duration")
    return float(value)


def silencedetect(
    input_path: str,
    *,
    noise_db: float = -30.0,
    min_silence: float = 0.5,
) -> List[Tuple[float, float]]:
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f",
        "null",
        "-",
    ]
    proc = run_cmd(cmd, capture_stderr=True)
    if proc.returncode != 0 and proc.stderr:
        # ffmpeg returns 1 when it finishes writing to null; treat stderr as output
        pass

    stderr = proc.stderr or ""
    silence_starts = [float(x) for x in re.findall(r"silence_start: ([0-9.]+)", stderr)]
    silence_ends = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", stderr)]

    silences: List[Tuple[float, float]] = []
    for start, end in zip(silence_starts, silence_ends):
        if end > start:
            silences.append((start, end))
    return silences
