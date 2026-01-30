from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Optional

from .cache import Cache
from .ffmpeg_tools import extract_audio_segment
from .models import Chunk, Span
from .utils import sha256_hex


def _make_cache_key(input_hash: str, chunk: Chunk, provider: str, params: dict) -> str:
    raw = f"{input_hash}:{chunk.chunk_id}:{provider}:{json.dumps(params, sort_keys=True)}".encode("utf-8")
    return sha256_hex(raw)


def transcribe_chunk(
    *,
    input_path: str,
    input_hash: str,
    chunk: Chunk,
    provider: str,
    params: dict,
    cache: Optional[Cache],
    span_start_id: int,
) -> List[Span]:
    request_body = {
        "provider": provider,
        "chunk_id": chunk.chunk_id,
        "t0": chunk.t0,
        "t1": chunk.t1,
        "params": params,
    }
    request_hash = sha256_hex(json.dumps(request_body, sort_keys=True).encode("utf-8"))
    cache_key = _make_cache_key(input_hash, chunk, provider, params)

    if cache:
        cached = cache.get(cache_key, request_hash)
        if cached is not None:
            return [
                Span(
                    sid=item["sid"],
                    t0=item["t0"],
                    t1=item["t1"],
                    chunk_id=item["chunk_id"],
                    text_raw=item["text_raw"],
                    asr_conf=item["asr_conf"],
                )
                for item in cached.get("spans", [])
            ]

    spans: List[Span] = []
    if provider == "mock":
        sid = f"S{span_start_id:05d}"
        spans.append(
            Span(
                sid=sid,
                t0=chunk.t0,
                t1=chunk.t1,
                chunk_id=chunk.chunk_id,
                text_raw=f"[mock] {chunk.chunk_id}",
                asr_conf=0.5,
            )
        )
    elif provider == "noop":
        spans = []
    elif provider == "deepgram":
        api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

        model = params.get("model", "nova-2")
        diarize = params.get("diarize", True)
        smart_format = params.get("smart_format", False)
        punctuate = params.get("punctuate", True)

        audio_bytes = extract_audio_segment(
            input_path,
            start=chunk.t0,
            end=chunk.t1,
            sample_rate=16000,
            channels=1,
        )
        query = (
            f"model={model}"
            f"&diarize={'true' if diarize else 'false'}"
            f"&smart_format={'true' if smart_format else 'false'}"
            f"&punctuate={'true' if punctuate else 'false'}"
        )
        url = f"https://api.deepgram.com/v1/listen?{query}"
        req = urllib.request.Request(
            url=url,
            data=audio_bytes,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "audio/wav",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - surface as runtime error
            raise RuntimeError(f"Deepgram request failed: {exc}") from exc

        channel = (payload.get("results", {}).get("channels") or [{}])[0]
        alt = (channel.get("alternatives") or [{}])[0]
        transcript = (alt.get("transcript") or "").strip()
        words = alt.get("words") or []
        confidence = float(alt.get("confidence") or 0.0)

        if transcript and words:
            start_time = chunk.t0 + float(words[0].get("start", 0.0))
            end_time = chunk.t0 + float(words[-1].get("end", 0.0))
        else:
            start_time = chunk.t0
            end_time = chunk.t1

        if transcript:
            sid = f"S{span_start_id:05d}"
            spans.append(
                Span(
                    sid=sid,
                    t0=round(start_time, 3),
                    t1=round(end_time, 3),
                    chunk_id=chunk.chunk_id,
                    text_raw=transcript,
                    asr_conf=confidence,
                )
            )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if cache is not None:
        cache.set(
            cache_key,
            request_hash,
            {
                "spans": [
                    {
                        "sid": span.sid,
                        "t0": span.t0,
                        "t1": span.t1,
                        "chunk_id": span.chunk_id,
                        "text_raw": span.text_raw,
                        "asr_conf": span.asr_conf,
                    }
                    for span in spans
                ]
            },
        )

    return spans
