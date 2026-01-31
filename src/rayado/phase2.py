from __future__ import annotations

import os
from typing import List, Optional


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _openai_client():
    from openai import OpenAI

    return OpenAI(timeout=60)


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
    cache_models = {
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-codex",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1-chat-latest",
        "gpt-4.1",
    }

    def _supports_retention(name: str) -> bool:
        if name in cache_models:
            return True
        for base in ("gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1"):
            if name.startswith(base + "-"):
                suffix = name[len(base) + 1 :]
                return suffix.replace(".", "").isdigit()
        return False
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
            if prompt_cache_retention and _supports_retention(model):
                payload["prompt_cache_retention"] = prompt_cache_retention
            resp = client.responses.create(**payload)
            return _response_text(resp)
        except Exception:
            attempts += 1
            if attempts > retry:
                raise


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


def run_phase2(
    *,
    srt_path: str,
    prompt_path: str,
    graph_in_path: Optional[str],
    graph_out_path: str,
    model_graph: str,
    retry: int,
) -> str:
    base_graph = _read_text(graph_in_path) if graph_in_path else ""
    prompt_text = _read_text(prompt_path)
    srt_text = _read_text(srt_path)

    if base_graph:
        input_payload = [
            {"role": "developer", "content": prompt_text},
            {"role": "user", "content": f"[EXISTING_GRAPH]\n{base_graph}\n\n[NEW_SRT]\n{srt_text}"},
        ]
        append_text = _call_openai(model=model_graph, input_payload=input_payload, retry=retry)
        merged = base_graph.rstrip() + "\n\n" + append_text.strip() + "\n"
        _write_text(graph_out_path, merged)
        return merged

    graph_text = generate_object_graph(
        srt_path=srt_path,
        prompt_path=prompt_path,
        model=model_graph,
        out_graph_path=graph_out_path,
        retry=retry,
    )
    return graph_text
