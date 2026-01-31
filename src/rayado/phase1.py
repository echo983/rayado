from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .asr import transcribe_chunk
from .cache import Cache
from .ffmpeg_tools import extract_audio_file, extract_audio_file_segment, ffprobe_duration, silencedetect
from .lid_voxlingua import detect_language_voxlingua
from .models import Chunk, Segment, Span
from .render import render_srt
from .utils import ensure_dir, hash_file
from .vad import build_speech_segments

SENTENCE_PUNCT = {".", "?", "!", "。", "？", "！"}


def _join_token(text: str, token: str, *, language: str) -> str:
    if not text:
        return token
    if language in {"zh", "ja", "ko"}:
        return f"{text}{token}"
    if token and token[0] in {".", ",", "?", "!", ":", ";", "。", "，", "？", "！"}:
        return f"{text}{token}"
    return f"{text} {token}"


def _words_to_spans(
    *,
    words: List[Dict[str, object]],
    fallback_text: str,
    chunk: Chunk,
    language: str,
    span_start_id: int,
) -> List[Span]:
    spans: List[Span] = []
    if not words and fallback_text.strip():
        spans.append(
            Span(
                sid=f"S{span_start_id:05d}",
                t0=chunk.t0,
                t1=max(chunk.t1, chunk.t0 + 0.001),
                chunk_id=chunk.chunk_id,
                text_raw=fallback_text.strip(),
                asr_conf=0.0,
            )
        )
        return spans

    current_text = ""
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_words = 0
    span_id = span_start_id

    for word in words:
        raw = (word.get("punctuated_word") or word.get("word") or "").strip()
        if not raw:
            continue
        w_start = word.get("start")
        w_end = word.get("end")
        if w_start is None or w_end is None:
            continue
        abs_start = chunk.t0 + float(w_start)
        abs_end = chunk.t0 + float(w_end)

        if current_start is None:
            current_start = abs_start
        if current_end is not None and abs_start - current_end > 1.0:
            spans.append(
                Span(
                    sid=f"S{span_id:05d}",
                    t0=current_start,
                    t1=max(current_end, current_start + 0.001),
                    chunk_id=chunk.chunk_id,
                    text_raw=current_text.strip(),
                    asr_conf=0.0,
                )
            )
            span_id += 1
            current_text = ""
            current_start = abs_start
            current_end = None
            current_words = 0

        current_text = _join_token(current_text, raw, language=language)
        current_end = abs_end
        current_words += 1

        duration = (current_end - current_start) if current_start is not None else 0.0
        char_limit = 24 if language in {"zh", "ja", "ko"} else 42
        should_break = False
        if raw[-1:] in SENTENCE_PUNCT:
            should_break = True
        if duration >= 7.0:
            should_break = True
        if current_words >= 14:
            should_break = True
        if len(current_text) >= char_limit:
            should_break = True

        if should_break and current_start is not None and current_end is not None:
            spans.append(
                Span(
                    sid=f"S{span_id:05d}",
                    t0=current_start,
                    t1=max(current_end, current_start + 0.001),
                    chunk_id=chunk.chunk_id,
                    text_raw=current_text.strip(),
                    asr_conf=0.0,
                )
            )
            span_id += 1
            current_text = ""
            current_start = None
            current_end = None
            current_words = 0

    if current_text and current_start is not None and current_end is not None:
        spans.append(
            Span(
                sid=f"S{span_id:05d}",
                t0=current_start,
                t1=max(current_end, current_start + 0.001),
                chunk_id=chunk.chunk_id,
                text_raw=current_text.strip(),
                asr_conf=0.0,
            )
        )
    return spans


def _group_segments(segments: List[Segment], *, target_sec: float, max_sec: float) -> List[Segment]:
    if not segments:
        return []
    if target_sec <= 0:
        return segments
    max_sec = max_sec if max_sec > 0 else target_sec * 2
    grouped: List[Segment] = []
    cur_start = segments[0].start
    cur_end = segments[0].end
    for seg in segments[1:]:
        proposed_end = seg.end
        if (cur_end - cur_start) < target_sec and (proposed_end - cur_start) <= max_sec:
            cur_end = proposed_end
        else:
            grouped.append(Segment(cur_start, cur_end))
            cur_start = seg.start
            cur_end = seg.end
    grouped.append(Segment(cur_start, cur_end))
    return grouped


def _build_chunks(segments: List[Segment]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for idx, seg in enumerate(segments, start=1):
        chunks.append(
            Chunk(
                chunk_id=f"C{idx:05d}",
                t0=round(seg.start, 3),
                t1=round(seg.end, 3),
                overlap_left=0.0,
                overlap_right=0.0,
            )
        )
    return chunks


def _pick_sample_segments(segments: List[Segment]) -> List[Segment]:
    if not segments:
        return []
    n = len(segments)
    positions = [0, int(n * 0.25), int(n * 0.50), int(n * 0.75), n - 1]
    chosen: List[Segment] = []
    seen = set()
    for pos in positions:
        pos = max(0, min(n - 1, pos))
        seg = segments[pos]
        if seg in seen:
            continue
        seen.add(seg)
        chosen.append(seg)
    return chosen


def run_phase1(
    *,
    input_path: str,
    out_srt_path: str,
    cache_dir: str,
    concurrency: int,
    retry: int,
    deepgram_model: str,
    vad_threshold: float,
    vad_min_speech_sec: float,
    vad_merge_gap_sec: float,
    vad_pad_sec: float,
    vad_min_silence_sec: float,
    target_sec: float,
    max_sec: float,
    lid_cache_dir: str,
    lid_device: str,
    output_txt_only: bool = False,
) -> str:
    ensure_dir(os.path.dirname(out_srt_path))
    cache = Cache(os.path.join(cache_dir, "cache.sqlite"))

    work_dir = os.path.join(cache_dir, "work_audio")
    ensure_dir(work_dir)
    base = os.path.splitext(os.path.basename(input_path))[0]
    work_wav = os.path.join(work_dir, f"{base}.wav")

    extract_audio_file(input_path, work_wav, sample_rate=16000, channels=1)

    duration = ffprobe_duration(work_wav)
    silences = silencedetect(work_wav, noise_db=vad_threshold, min_silence=vad_min_silence_sec)
    speech_segments = build_speech_segments(
        duration,
        silences,
        pad_sec=vad_pad_sec,
        min_speech_sec=vad_min_speech_sec,
        merge_gap_sec=vad_merge_gap_sec,
    )
    grouped_segments = _group_segments(speech_segments, target_sec=target_sec, max_sec=max_sec)
    chunks = _build_chunks(grouped_segments)

    if not chunks:
        raise RuntimeError("No speech chunks found after VAD.")

    with tempfile.TemporaryDirectory(prefix="rayado_lid_") as lid_tmp:
        sample_wavs: List[str] = []
        for seg in _pick_sample_segments(grouped_segments):
            sample_path = os.path.join(lid_tmp, f"{base}_{seg.start:.3f}.wav")
            extract_audio_file_segment(
                work_wav,
                sample_path,
                start=seg.start,
                end=seg.end,
                sample_rate=16000,
                channels=1,
            )
            sample_wavs.append(sample_path)
        detected_language, _, _ = detect_language_voxlingua(
            sample_wavs,
            cache_dir=lid_cache_dir,
            device=lid_device,
        )
    if not detected_language:
        raise RuntimeError("Language detection failed to select a dominant language.")

    params = {
        "model": deepgram_model,
        "language": detected_language,
        "detect_language": False,
        "detect_language_set": [],
        "diarize": False,
        "smart_format": False,
        "punctuate": True,
    }

    spans: List[Span] = []
    span_id = 1

    def _process_chunk(chunk: Chunk) -> List[Span]:
        wav_path = os.path.join(chunk_tmp_dir, f"{chunk.chunk_id}.wav")
        extract_audio_file_segment(
            work_wav,
            wav_path,
            start=chunk.t0,
            end=chunk.t1,
            sample_rate=16000,
            channels=1,
        )
        input_hash = hash_file(wav_path)
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        attempts = 0
        span_seed = int(chunk.chunk_id[1:]) if chunk.chunk_id[1:].isdigit() else 0
        while True:
            try:
                chunk_spans, meta = transcribe_chunk(
                    input_path=wav_path,
                    input_hash=input_hash,
                    chunk=chunk,
                    provider="deepgram",
                    params=params,
                    cache=cache,
                    span_start_id=span_seed,
                    audio_bytes=audio_bytes,
                )
                break
            except Exception as exc:
                attempts += 1
                if "429" in str(exc):
                    time.sleep(5)
                if attempts > retry:
                    raise

        words = meta.get("words") if meta else []
        fallback_text = chunk_spans[0].text_raw if chunk_spans else ""
        return _words_to_spans(
            words=words or [],
            fallback_text=fallback_text,
            chunk=chunk,
            language=detected_language,
            span_start_id=0,
        )

    with tempfile.TemporaryDirectory(prefix="rayado_chunks_") as chunk_tmp_dir:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            future_map = {executor.submit(_process_chunk, chunk): chunk for chunk in chunks}
            for future in as_completed(future_map):
                chunk_spans = future.result()
                for span in chunk_spans:
                    spans.append(
                        Span(
                            sid=f"S{span_id:05d}",
                            t0=span.t0,
                            t1=span.t1,
                            chunk_id=span.chunk_id,
                            text_raw=span.text_raw,
                            asr_conf=span.asr_conf,
                        )
                    )
                    span_id += 1

    if output_txt_only:
        lines: List[str] = []
        for span in sorted(spans, key=lambda s: (s.t0, s.t1, s.sid)):
            text = span.text_raw.strip()
            if text:
                lines.append(text)
        with open(out_srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    else:
        srt = render_srt(spans)
        with open(out_srt_path, "w", encoding="utf-8") as f:
            f.write(srt)

    return detected_language
