from __future__ import annotations

import os
from typing import Optional

from .phase2 import _call_openai, _read_text, _write_text


def run_phase3(
    *,
    graph_a_path: str,
    graph_b_path: str,
    prompt_path: str,
    graph_out_path: str,
    model: str,
    retry: int,
) -> str:
    prompt_text = _read_text(prompt_path)
    graph_a = _read_text(graph_a_path)
    graph_b = _read_text(graph_b_path)

    input_payload = [
        {"role": "developer", "content": prompt_text},
        {"role": "user", "content": f"[INPUT_A]\n{graph_a}\n\n[INPUT_B]\n{graph_b}"},
    ]

    merged = _call_openai(
        model=model,
        input_payload=input_payload,
        output_path=graph_out_path,
        retry=retry,
    )
    _write_text(graph_out_path, merged)
    return merged
