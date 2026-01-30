from __future__ import annotations

import re
import subprocess
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


def extract_audio_segment(
    input_path: str,
    *,
    start: float,
    end: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    cmd = [
        "ffmpeg",
        "-ss",
        f"{start}",
        "-to",
        f"{end}",
        "-i",
        input_path,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "wav",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(proc.stderr.decode(errors="ignore") or "ffmpeg extract failed")
    return proc.stdout


def extract_audio_file_segment(
    input_path: str,
    output_path: str,
    *,
    start: float,
    end: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start}",
        "-to",
        f"{end}",
        "-i",
        input_path,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "wav",
        output_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore") or "ffmpeg extract file failed")
