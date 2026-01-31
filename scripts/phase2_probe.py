from __future__ import annotations

import argparse
import os
import time


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _call_openai(model: str, input_payload, output_path: str):
    from openai import OpenAI

    client = OpenAI(timeout=900)
    stream = client.responses.create(
        model=model,
        input=input_payload,
        stream=True,
        max_output_tokens=128000,
    )
    parts: list[str] = []
    last_flush = time.time()
    for event in stream:
        event_type = getattr(event, "type", None)
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                parts.append(delta)
        now = time.time()
        if now - last_flush >= 10:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("".join(parts))
            last_flush = now
        if event_type in {"response.completed", "response.failed"}:
            break
    content = "".join(parts).strip()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe object-graph generation only")
    parser.add_argument("srt", help="Input SRT path")
    parser.add_argument("--graph-in", default=None, help="Existing object graph file")
    parser.add_argument("--model", default="gpt-5.2", help="Model name")
    args = parser.parse_args()

    srt_text = _read_text(args.srt)
    base_graph = _read_text(args.graph_in) if args.graph_in else ""
    prompt = _read_text(os.path.join("prompts", "SORAL.txt"))
    if base_graph:
        user_content = f"[EXISTING_GRAPH]\n{base_graph}\n\n[NEW_SRT]\n{srt_text}"
    else:
        user_content = srt_text
    input_payload = [
        {"role": "developer", "content": prompt},
        {"role": "user", "content": user_content},
    ]

    start = time.time()
    out_path = os.path.join("out", "phase2_probe.graph.txt")
    output = _call_openai(args.model, input_payload, out_path)
    elapsed = time.time() - start

    print(f"elapsed_sec={elapsed:.2f}")
    print(f"output_chars={len(output)}")
    print(f"output_path={out_path}")


if __name__ == "__main__":
    main()
