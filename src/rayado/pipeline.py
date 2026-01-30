from __future__ import annotations

import os
from typing import Dict, List

from . import __version__
from .asr import transcribe_chunk
from .cache import Cache
from .chunking import chunk_has_speech, generate_chunks
from .entity import extract_entities
from .ffmpeg_tools import ffprobe_duration, silencedetect
from .gcl import append_block, ensure_header
from .models import Chunk, Span
from .overlap import overlap_judge
from .render import render_srt, render_transcript
from .speaker import build_speaker_blocks
from .utils import ensure_dir, hash_file
from .vad import build_speech_segments


def run_pipeline(
    *,
    input_path: str,
    out_dir: str,
    cache_dir: str,
    provider: str,
    retry: int,
    deepgram_model: str,
    deepgram_language: str,
    deepgram_detect_language: bool,
    deepgram_detect_language_set: list[str],
    deepgram_diarize: bool,
    deepgram_smart_format: bool,
    deepgram_punctuate: bool,
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
    speakers_written: set[str] = set()
    speaker_labels: Dict[str, str] = {}
    speaker_by_sid: Dict[str, str] = {}

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

        params = {"version": __version__}
        if provider == "deepgram":
            params.update(
                {
                    "model": deepgram_model,
                    "language": deepgram_language,
                    "detect_language": deepgram_detect_language,
                    "detect_language_set": deepgram_detect_language_set,
                    "diarize": deepgram_diarize,
                    "smart_format": deepgram_smart_format,
                    "punctuate": deepgram_punctuate,
                }
            )

        attempts = 0
        while True:
            try:
                chunk_spans, chunk_meta = transcribe_chunk(
                    input_path=input_path,
                    input_hash=input_hash,
                    chunk=chunk,
                    provider=provider,
                    params=params,
                    cache=cache,
                    span_start_id=span_id,
                )
                break
            except Exception:
                attempts += 1
                if attempts > retry:
                    raise

        if provider == "deepgram" and chunk_meta and chunk_spans:
            detected_lang = chunk_meta.get("detected_language") or ""
            lang_conf = chunk_meta.get("language_confidence")
            if detected_lang:
                append_block(
                    gcl_path,
                    "GCL_OVERRIDE",
                    {
                        "oid": f"LANG_{chunk.chunk_id}",
                        "sid": chunk_spans[0].sid,
                        "policy": "detected_language",
                        "norm_zh": "",
                        "conf": f"{lang_conf}" if lang_conf is not None else "",
                        "deps": detected_lang,
                    },
                )

            words = chunk_meta.get("words") or []
            speaker_block, speaker_map_block = build_speaker_blocks(
                chunk_id=chunk.chunk_id,
                span_id=chunk_spans[0].sid,
                words=words,
            )
            if speaker_block:
                spk_id = speaker_block.get("spk_id", "")
                label = speaker_block.get("label", "")
                if spk_id and spk_id not in speakers_written:
                    append_block(gcl_path, "GCL_SPEAKER", speaker_block)
                    speakers_written.add(spk_id)
                    if label:
                        speaker_labels[spk_id] = label
                if speaker_map_block:
                    append_block(gcl_path, "GCL_SPEAKER_MAP", speaker_map_block)
                    if label:
                        speaker_by_sid[speaker_map_block.get("sid", "")] = label

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

    overlap_records, suppressed = overlap_judge(chunks, spans)
    for record in overlap_records:
        append_block(gcl_path, "GCL_OVERLAP", record)
    for sid in suppressed:
        append_block(
            gcl_path,
            "GCL_OVERRIDE",
            {
                "oid": f"SUP_{sid}",
                "sid": sid,
                "policy": "suppress",
                "conf": "",
            },
        )

    suppressed_set = set(suppressed)
    spans_filtered = [span for span in spans if span.sid not in suppressed_set]

    entities, mentions = extract_entities(spans_filtered)
    for entity in entities:
        append_block(gcl_path, "GCL_ENTITY", entity)
    for mention in mentions:
        append_block(gcl_path, "GCL_MENTION", mention)

    transcript = render_transcript(spans_filtered, speaker_by_sid=speaker_by_sid)
    srt = render_srt(spans_filtered, speaker_by_sid=speaker_by_sid)

    with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(transcript)
    with open(os.path.join(out_dir, "subtitles.srt"), "w", encoding="utf-8") as f:
        f.write(srt)
