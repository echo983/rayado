from __future__ import annotations

import argparse
import os
import time


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _call_openai(model: str, input_payload):
    from openai import OpenAI

    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=input_payload,
    )
    if hasattr(resp, "output_text"):
        return resp.output_text or ""
    output = getattr(resp, "output", None) or []
    parts = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(parts).strip()


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
    output = _call_openai(args.model, input_payload)
    elapsed = time.time() - start

    print(f"elapsed_sec={elapsed:.2f}")
    print(f"output_chars={len(output)}")
    out_path = os.path.join("out", "phase2_probe.graph.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"output_path={out_path}")


if __name__ == "__main__":
    main()
