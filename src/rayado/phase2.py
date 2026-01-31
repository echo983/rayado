from __future__ import annotations

import os
from typing import Iterable, List, Optional

from .srt_utils import SrtBlock, chunk_srt_blocks, format_srt_blocks, parse_srt_blocks


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _openai_client():
    from openai import OpenAI

    return OpenAI()


def _response_text(resp) -> str:
    if hasattr(resp, "output_text"):
        return resp.output_text or ""
    output = getattr(resp, "output", None)
    if not output:
        return ""
    parts: List[str] = []
    for item in output:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "message":
            continue
        contents = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
        for content in contents:
            content_type = content.get("type") if isinstance(content, dict) else getattr(content, "type", None)
            if content_type == "output_text":
                text = content.get("text") if isinstance(content, dict) else getattr(content, "text", "")
                parts.append(text or "")
    return "\n".join(parts).strip()


def _call_openai(
    *,
    model: str,
    input_payload,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = None,
    retry: int = 1,
) -> str:
    client = _openai_client()
    attempts = 0
    while True:
        try:
            payload = {
                "model": model,
                "input": input_payload,
            }
            if prompt_cache_key:
                payload["prompt_cache_key"] = prompt_cache_key
            if prompt_cache_retention:
                payload["prompt_cache_retention"] = prompt_cache_retention
            resp = client.responses.create(**payload)
            return _response_text(resp)
        except Exception:
            attempts += 1
            if attempts > retry:
                raise


def _chunk_prefix(index: int) -> str:
    return f"C{index:05d}"


def generate_object_graph(
    *,
    srt_path: str,
    prompt_path: str,
    model: str,
    out_graph_path: str,
    retry: int,
) -> str:
    prompt_text = _read_text(prompt_path)
    srt_text = _read_text(srt_path)
    input_payload = [
        {"role": "developer", "content": prompt_text},
        {"role": "user", "content": srt_text},
    ]
    graph_text = _call_openai(model=model, input_payload=input_payload, retry=retry)
    _write_text(out_graph_path, graph_text)
    return graph_text


def rebuild_srt(
    *,
    srt_path: str,
    graph_text: str,
    out_srt_path: str,
    model: str,
    chunk_chars: int,
    prompt_cache_retention: str,
    retry: int,
) -> None:
    blocks = parse_srt_blocks(_read_text(srt_path))
    srt_chunks = chunk_srt_blocks(blocks, max_chars=chunk_chars)

    base_prompt = (
        "You will clean and normalize SRT text for LLM consumption.\n"
        "Rules:\n"
        "- Output SRT only, no extra commentary.\n"
        "- Keep timestamps unchanged unless they are obviously invalid.\n"
        "- Preserve language (no translation).\n"
        "- Keep lines concise and readable.\n"
    )
    static_context = f"{base_prompt}\n[OBJECT_GRAPH]\n{graph_text}\n"

    rebuilt_blocks: List[SrtBlock] = []
    for idx, chunk in enumerate(srt_chunks, start=1):
        chunk_text = format_srt_blocks(chunk)
        input_payload = [
            {"role": "developer", "content": static_context},
            {"role": "user", "content": chunk_text},
        ]
        output = _call_openai(
            model=model,
            input_payload=input_payload,
            prompt_cache_key=_chunk_prefix(idx),
            prompt_cache_retention=prompt_cache_retention,
            retry=retry,
        )
        parsed = parse_srt_blocks(output)
        if not parsed:
            parsed = chunk
        rebuilt_blocks.extend(parsed)

    _write_text(out_srt_path, format_srt_blocks(rebuilt_blocks))


def run_phase2(
    *,
    srt_path: str,
    prompt_path: str,
    graph_in_path: Optional[str],
    graph_out_path: str,
    cleaned_srt_path: str,
    model_graph: str,
    model_clean: str,
    chunk_chars: int,
    prompt_cache_retention: str,
    retry: int,
) -> str:
    if graph_in_path:
        graph_text = _read_text(graph_in_path)
    else:
        graph_text = generate_object_graph(
            srt_path=srt_path,
            prompt_path=prompt_path,
            model=model_graph,
            out_graph_path=graph_out_path,
            retry=retry,
        )

    rebuild_srt(
        srt_path=srt_path,
        graph_text=graph_text,
        out_srt_path=cleaned_srt_path,
        model=model_clean,
        chunk_chars=chunk_chars,
        prompt_cache_retention=prompt_cache_retention,
        retry=retry,
    )
    return graph_text
