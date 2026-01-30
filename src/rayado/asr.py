from __future__ import annotations

import json
from typing import List, Optional

from .cache import Cache
from .models import Chunk, Span
from .utils import sha256_hex


def _make_cache_key(input_hash: str, chunk: Chunk, provider: str, params: dict) -> str:
    raw = f"{input_hash}:{chunk.chunk_id}:{provider}:{json.dumps(params, sort_keys=True)}".encode("utf-8")
    return sha256_hex(raw)


def transcribe_chunk(
    *,
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
