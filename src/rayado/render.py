from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .models import Span


def _format_srt_time(seconds: float) -> str:
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


def _speaker_prefix(
    span: Span,
    *,
    speaker_by_sid: Optional[Dict[str, str]],
) -> str:
    if not speaker_by_sid:
        return ""
    label = speaker_by_sid.get(span.sid)
    if not label:
        return ""
    return f"{label}: "


def render_transcript(
    spans: Iterable[Span],
    *,
    speaker_by_sid: Optional[Dict[str, str]] = None,
) -> str:
    lines: List[str] = []
    for span in sorted(spans, key=lambda s: (s.t0, s.t1, s.sid)):
        text = span.text_raw.strip()
        if not text:
            continue
        prefix = _speaker_prefix(span, speaker_by_sid=speaker_by_sid)
        lines.append(f"{prefix}{text}")
    return "\n".join(lines) + ("\n" if lines else "")


def render_srt(
    spans: Iterable[Span],
    *,
    speaker_by_sid: Optional[Dict[str, str]] = None,
) -> str:
    output: List[str] = []
    index = 1
    for span in sorted(spans, key=lambda s: (s.t0, s.t1, s.sid)):
        text = span.text_raw.strip()
        if not text:
            continue
        start = span.t0
        end = max(span.t1, span.t0 + 0.001)
        prefix = _speaker_prefix(span, speaker_by_sid=speaker_by_sid)
        output.append(str(index))
        output.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        output.append(f"{prefix}{text}")
        output.append("")
        index += 1
    return "\n".join(output)
