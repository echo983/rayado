from __future__ import annotations

from rayado import phase3


def test_phase3_merge(monkeypatch, tmp_path):
    graph_a = tmp_path / "a.graph.txt"
    graph_b = tmp_path / "b.graph.txt"
    graph_a.write_text("[ENTITY:E1]{Name:\"A\"}", encoding="utf-8")
    graph_b.write_text("[ENTITY:E2]{Name:\"B\"}", encoding="utf-8")

    prompt_path = tmp_path / "SORAL_Merge.txt"
    prompt_path.write_text("PROMPT", encoding="utf-8")

    def fake_call_openai(*, model, input_payload, output_path=None, output_prefix="", retry=1, **kwargs):
        return "[ENTITY:E1]{Name:\"A\"}\n[ENTITY:E2]{Name:\"B\"}"

    monkeypatch.setattr(phase3, "_call_openai", fake_call_openai)

    out_path = tmp_path / "merged.graph.txt"
    phase3.run_phase3(
        graph_a_path=str(graph_a),
        graph_b_path=str(graph_b),
        prompt_path=str(prompt_path),
        graph_out_path=str(out_path),
        model="gpt-5.1",
        retry=1,
    )

    assert out_path.exists()
