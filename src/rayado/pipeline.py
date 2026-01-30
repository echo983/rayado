from __future__ import annotations

import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from . import __version__
from .asr import transcribe_chunk
from .cache import Cache
from .chunking import generate_chunks
from .entity import extract_entities
from .ffmpeg_tools import extract_audio_file_segment, ffprobe_duration, silencedetect
from .gcl import append_block, ensure_header
from .models import Chunk, Span
from .overlap import overlap_judge
from .render import render_srt, render_transcript
from .speaker import build_speaker_blocks
from .stats import RunStats, now, write_run_log
from .utils import ensure_dir, hash_file
from .vad import build_speech_segments

ResultItem = Tuple[Chunk, Optional[str], List[Span], Dict[str, str], Dict[str, str], List[Dict[str, str]]]


def _process_chunk(
    *,
    chunk: Chunk,
    input_path: str,
    provider: str,
    params: dict,
    cache: Cache,
    span_start_id: int,
    retry: int,
    vad_enabled: bool,
    vad_threshold: float,
    vad_min_speech_sec: float,
    vad_merge_gap_sec: float,
    vad_pad_sec: float,
    tmp_dir: str,
) -> ResultItem:
    wav_path = os.path.join(tmp_dir, f"{chunk.chunk_id}.wav")
    extract_audio_file_segment(
        input_path,
        wav_path,
        start=chunk.t0,
        end=chunk.t1,
        sample_rate=16000,
        channels=1,
    )

    try:
        if vad_enabled:
            silences = silencedetect(wav_path, noise_db=vad_threshold, min_silence=0.5)
            speech_segments = build_speech_segments(
                chunk.t1 - chunk.t0,
                silences,
                pad_sec=vad_pad_sec,
                min_speech_sec=vad_min_speech_sec,
                merge_gap_sec=vad_merge_gap_sec,
            )
            if not speech_segments:
                return (chunk, "non_speech", [], {}, {}, [])

        input_hash = hash_file(wav_path)
        attempts = 0
        while True:
            try:
                spans, meta = transcribe_chunk(
                    input_path=wav_path,
                    input_hash=input_hash,
                    chunk=chunk,
                    provider=provider,
                    params=params,
                    cache=cache,
                    span_start_id=span_start_id,
                )
                break
            except Exception:
                attempts += 1
                if attempts > retry:
                    raise

        gcl_overrides: List[Dict[str, str]] = []
        speaker_block: Dict[str, str] = {}
        speaker_map_block: Dict[str, str] = {}

        if provider == "deepgram" and meta and spans:
            detected_lang = meta.get("detected_language") or ""
            lang_conf = meta.get("language_confidence")
            if detected_lang:
                gcl_overrides.append(
                    {
                        "oid": f"LANG_{chunk.chunk_id}",
                        "sid": spans[0].sid,
                        "policy": "detected_language",
                        "norm_zh": "",
                        "conf": f"{lang_conf}" if lang_conf is not None else "",
                        "deps": detected_lang,
                    }
                )

            words = meta.get("words") or []
            speaker_block, speaker_map_block = build_speaker_blocks(
                chunk_id=chunk.chunk_id,
                span_id=spans[0].sid,
                words=words,
            )

        return (chunk, None, spans, speaker_block, speaker_map_block, gcl_overrides)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def run_pipeline(
    *,
    input_path: str,
    out_dir: str,
    cache_dir: str,
    provider: str,
    retry: int,
    concurrency: int,
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
    started_at = now()
    step_timings: List[Dict[str, float | str | None]] = []
    current_step: Dict[str, float | str | None] | None = None

    def step_start(name: str) -> Dict[str, float | str | None]:
        entry = {"name": name, "started_at": now(), "ended_at": None, "duration_sec": None}
        step_timings.append(entry)
        return entry

    def step_end(entry: Dict[str, float | str | None]) -> None:
        ended = now()
        entry["ended_at"] = ended
        started = entry.get("started_at")
        if isinstance(started, (int, float)):
            entry["duration_sec"] = ended - started
    current_step = step_start("init")
    ensure_dir(out_dir)
    cache = Cache(os.path.join(cache_dir, "cache.sqlite"))
    step_end(current_step)

    current_step = step_start("probe_duration")
    duration = ffprobe_duration(input_path)
    step_end(current_step)

    gcl_path = os.path.join(out_dir, "episode.gcl")
    ensure_header(gcl_path)

    spans: List[Span] = []
    speakers_written: set[str] = set()
    speaker_by_sid: Dict[str, str] = {}

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

    current_step = step_start("chunk_plan")
    chunks = generate_chunks(duration, chunk_sec=chunk_sec, overlap_sec=overlap_sec)
    step_end(current_step)

    tmp_dir = os.path.join(out_dir, "_audio_chunks")
    ensure_dir(tmp_dir)

    results: List[ResultItem] = []
    results_lock = threading.Lock()
    errors: List[Exception] = []

    vad_enabled = vad_name.lower() not in {"none", "off", "disabled"}

    current_step = step_start("chunk_process")
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_map = {}
        for idx, chunk in enumerate(chunks, start=1):
            future = executor.submit(
                _process_chunk,
                chunk=chunk,
                input_path=input_path,
                provider=provider,
                params=params,
                cache=cache,
                span_start_id=idx,
                retry=retry,
                vad_enabled=vad_enabled,
                vad_threshold=vad_threshold,
                vad_min_speech_sec=vad_min_speech_sec,
                vad_merge_gap_sec=vad_merge_gap_sec,
                vad_pad_sec=vad_pad_sec,
                tmp_dir=tmp_dir,
            )
            future_map[future] = chunk

        for future in as_completed(future_map):
            try:
                item = future.result()
                with results_lock:
                    results.append(item)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                break
    step_end(current_step)

    if errors:
        ended_at = now()
        stats = RunStats(
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=ended_at - started_at,
            input_path=input_path,
            output_dir=out_dir,
            provider=provider,
            chunk_count=len(chunks),
            chunk_skipped=0,
            chunk_processed=0,
            chunk_failed=1,
            span_count=len(spans),
            suppressed_count=0,
        )
        write_run_log(
            os.path.join(out_dir, "run.log"),
            stats,
            {"error": str(errors[0]), "steps": step_timings},
        )
        raise RuntimeError("Chunk failed") from errors[0]

    results.sort(key=lambda x: x[0].t0)

    chunk_skipped = 0
    chunk_processed = 0

    for chunk, skip_reason, chunk_spans, speaker_block, speaker_map_block, gcl_overrides in results:
        if skip_reason:
            append_block(
                gcl_path,
                "GCL_CHUNK",
                {
                    "chunk_id": chunk.chunk_id,
                    "t0": f"{chunk.t0}",
                    "t1": f"{chunk.t1}",
                    "overlap_left": f"{chunk.overlap_left}",
                    "overlap_right": f"{chunk.overlap_right}",
                    "skip_reason": skip_reason,
                },
            )
            chunk_skipped += 1
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
        chunk_processed += 1

        for override in gcl_overrides:
            append_block(gcl_path, "GCL_OVERRIDE", override)

        if speaker_block:
            spk_id = speaker_block.get("spk_id", "")
            label = speaker_block.get("label", "")
            if spk_id and spk_id not in speakers_written:
                append_block(gcl_path, "GCL_SPEAKER", speaker_block)
                speakers_written.add(spk_id)
            if speaker_map_block:
                append_block(gcl_path, "GCL_SPEAKER_MAP", speaker_map_block)
                if label:
                    speaker_by_sid[speaker_map_block.get("sid", "")] = label

        for span in chunk_spans:
            spans.append(span)
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

    current_step = step_start("overlap_judge")
    overlap_records, suppressed = overlap_judge(chunks, spans)
    step_end(current_step)
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

    current_step = step_start("entity_extract")
    entities, mentions = extract_entities(spans_filtered)
    step_end(current_step)
    for entity in entities:
        append_block(gcl_path, "GCL_ENTITY", entity)
    for mention in mentions:
        append_block(gcl_path, "GCL_MENTION", mention)

    current_step = step_start("render_outputs")
    transcript = render_transcript(spans_filtered, speaker_by_sid=speaker_by_sid)
    srt = render_srt(spans_filtered, speaker_by_sid=speaker_by_sid)
    step_end(current_step)

    current_step = step_start("write_outputs")
    with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(transcript)
    with open(os.path.join(out_dir, "subtitles.srt"), "w", encoding="utf-8") as f:
        f.write(srt)
    step_end(current_step)

    if os.path.isdir(tmp_dir):
        current_step = step_start("cleanup")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        step_end(current_step)

    ended_at = now()
    stats = RunStats(
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=ended_at - started_at,
        input_path=input_path,
        output_dir=out_dir,
        provider=provider,
        chunk_count=len(chunks),
        chunk_skipped=chunk_skipped,
        chunk_processed=chunk_processed,
        chunk_failed=0,
        span_count=len(spans_filtered),
        suppressed_count=len(suppressed),
    )
    write_run_log(os.path.join(out_dir, "run.log"), stats, {"steps": step_timings})
