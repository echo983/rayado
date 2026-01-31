from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class SrtBlock:
    start: float
    end: float
    text: str


def format_srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    if millis < 0:
        millis = 0
    hours = millis // 3_600_000
    millis -= hours * 3_600_000
    minutes = millis // 60_000
    millis -= minutes * 60_000
    secs = millis // 1000
    millis -= secs * 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _parse_srt_time(value: str) -> float:
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", value.strip())
    if not match:
        return 0.0
    hours, minutes, secs, millis = [int(x) for x in match.groups()]
    return hours * 3600 + minutes * 60 + secs + millis / 1000.0


def parse_srt_blocks(text: str) -> List[SrtBlock]:
    blocks: List[SrtBlock] = []
    for raw in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip("\r") for line in raw.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_idx = 1 if "-->" in lines[1] else 0 if "-->" in lines[0] else None
        if time_idx is None:
            continue
        time_line = lines[time_idx]
        parts = [p.strip() for p in time_line.split("-->")]
        if len(parts) != 2:
            continue
        start = _parse_srt_time(parts[0])
        end = _parse_srt_time(parts[1])
        text_lines = lines[time_idx + 1 :]
        content = "\n".join(text_lines).strip()
        blocks.append(SrtBlock(start=start, end=end, text=content))
    return blocks


def format_srt_blocks(blocks: Iterable[SrtBlock]) -> str:
    output: List[str] = []
    index = 1
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        output.append(str(index))
        output.append(f"{format_srt_time(block.start)} --> {format_srt_time(block.end)}")
        output.append(text)
        output.append("")
        index += 1
    return "\n".join(output)


def chunk_srt_blocks(blocks: List[SrtBlock], *, max_chars: int) -> List[List[SrtBlock]]:
    if max_chars <= 0:
        return [blocks]
    chunks: List[List[SrtBlock]] = []
    current: List[SrtBlock] = []
    current_len = 0
    for block in blocks:
        block_text = f"{format_srt_time(block.start)} --> {format_srt_time(block.end)}\n{block.text}\n\n"
        if current and current_len + len(block_text) > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(block)
        current_len += len(block_text)
    if current:
        chunks.append(current)
    return chunks
