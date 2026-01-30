from __future__ import annotations

import os
from typing import List, Optional

from . import __version__
from .asr import transcribe_chunk
from .cache import Cache
from .chunking import chunk_has_speech, generate_chunks
from .ffmpeg_tools import ffprobe_duration, silencedetect
from .gcl import append_block, ensure_header
from .models import Chunk, Span
from .render import render_srt, render_transcript
from .utils import ensure_dir, hash_file
from .vad import build_speech_segments


def run_pipeline(
    *,
    input_path: str,
    out_dir: str,
    cache_dir: str,
    provider: str,
    chunk_sec: float,
    overlap_sec: float,
    vad_name: str,
    vad_threshold: float,
    vad_min_speech_sec: float,
    vad_merge_gap_sec: float,
    vad_pad_sec: float,
) -> None:
    ensure_dir(out_dir)
    cache = Cache(os.path.join(cache_dir, "cache.sqlite"))

    input_hash = hash_file(input_path)
    duration = ffprobe_duration(input_path)

    if vad_name.lower() in {"none", "off", "disabled"}:
        speech_segments = build_speech_segments(
            duration,
            [],
            pad_sec=0.0,
            min_speech_sec=0.0,
            merge_gap_sec=0.0,
        )
    else:
        silences = silencedetect(input_path, noise_db=vad_threshold, min_silence=0.5)
        speech_segments = build_speech_segments(
            duration,
            silences,
            pad_sec=vad_pad_sec,
            min_speech_sec=vad_min_speech_sec,
            merge_gap_sec=vad_merge_gap_sec,
        )

    chunks = generate_chunks(duration, chunk_sec=chunk_sec, overlap_sec=overlap_sec)

    gcl_path = os.path.join(out_dir, "episode.gcl")
    ensure_header(gcl_path)

    spans: List[Span] = []
    span_id = 1
    for chunk in chunks:
        if not chunk_has_speech(chunk, speech_segments):
            skip_chunk = Chunk(
                chunk_id=chunk.chunk_id,
                t0=chunk.t0,
                t1=chunk.t1,
                overlap_left=chunk.overlap_left,
                overlap_right=chunk.overlap_right,
                skip_reason="non_speech",
            )
            append_block(
                gcl_path,
                "GCL_CHUNK",
                {
                    "chunk_id": skip_chunk.chunk_id,
                    "t0": f"{skip_chunk.t0}",
                    "t1": f"{skip_chunk.t1}",
                    "overlap_left": f"{skip_chunk.overlap_left}",
                    "overlap_right": f"{skip_chunk.overlap_right}",
                    "skip_reason": skip_chunk.skip_reason or "",
                },
            )
            continue

        append_block(
            gcl_path,
            "GCL_CHUNK",
            {
                "chunk_id": chunk.chunk_id,
                "t0": f"{chunk.t0}",
                "t1": f"{chunk.t1}",
                "overlap_left": f"{chunk.overlap_left}",
                "overlap_right": f"{chunk.overlap_right}",
            },
        )

        chunk_spans = transcribe_chunk(
            input_hash=input_hash,
            chunk=chunk,
            provider=provider,
            params={"version": __version__},
            cache=cache,
            span_start_id=span_id,
        )
        for span in chunk_spans:
            spans.append(span)
            span_id += 1
            append_block(
                gcl_path,
                "GCL_SPAN",
                {
                    "sid": span.sid,
                    "t0": f"{span.t0}",
                    "t1": f"{span.t1}",
                    "chunk_id": span.chunk_id,
                    "text_raw": span.text_raw,
                    "asr_conf": f"{span.asr_conf}",
                },
            )

    transcript = render_transcript(spans)
    srt = render_srt(spans)

    with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(transcript)
    with open(os.path.join(out_dir, "subtitles.srt"), "w", encoding="utf-8") as f:
        f.write(srt)
