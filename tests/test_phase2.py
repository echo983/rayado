from __future__ import annotations

from rayado import phase2
from rayado.srt_utils import SrtBlock, format_srt_blocks


def test_phase2_generate_graph_and_clean(monkeypatch, tmp_path):
    srt_blocks = [
        SrtBlock(start=0.0, end=1.0, text="hello"),
        SrtBlock(start=1.1, end=2.0, text="world"),
    ]
    srt_path = tmp_path / "input.srt"
    srt_path.write_text(format_srt_blocks(srt_blocks), encoding="utf-8")

    prompt_path = tmp_path / "SORAL.txt"
    prompt_path.write_text("PROMPT", encoding="utf-8")

    def fake_call_openai(*, model, input_payload, prompt_cache_key=None, prompt_cache_retention=None, retry=1):
        if model == "gpt-5.2":
            return "[ENTITY:1]{Name:\"A\"}"
        return ""

    monkeypatch.setattr(phase2, "_call_openai", fake_call_openai)

    graph_out = tmp_path / "graph.txt"
    clean_out = tmp_path / "clean.srt"

    phase2.run_phase2(
        srt_path=str(srt_path),
        prompt_path=str(prompt_path),
        graph_in_path=None,
        graph_out_path=str(graph_out),
        model_graph="gpt-5.2",
        retry=1,
    )

    assert graph_out.exists()
