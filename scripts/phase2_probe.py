from __future__ import annotations

import argparse
import os
import time

from rayado.srt_utils import chunk_srt_blocks, format_srt_blocks, parse_srt_blocks


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
    parser = argparse.ArgumentParser(description="Probe a single Phase2 chunk with GPT-5-mini")
    parser.add_argument("srt", help="Input SRT path")
    parser.add_argument("--graph-in", required=True, help="Object graph file")
    parser.add_argument("--chunk-chars", type=int, default=12000, help="Chunk size in chars")
    parser.add_argument("--chunk-index", type=int, default=1, help="1-based chunk index")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name")
    args = parser.parse_args()

    srt_text = _read_text(args.srt)
    graph_text = _read_text(args.graph_in)
    blocks = parse_srt_blocks(srt_text)
    chunks = chunk_srt_blocks(blocks, max_chars=args.chunk_chars)

    if not chunks:
        raise SystemExit("No chunks found")
    if args.chunk_index < 1 or args.chunk_index > len(chunks):
        raise SystemExit(f"chunk-index out of range (1..{len(chunks)})")

    chunk = chunks[args.chunk_index - 1]
    chunk_text = format_srt_blocks(chunk)

    prompt = (
        "You will clean and normalize SRT text for LLM consumption.\n"
        "Rules:\n"
        "- Output SRT only, no extra commentary.\n"
        "- Keep timestamps unchanged unless they are obviously invalid.\n"
        "- Preserve language (no translation).\n"
        "- Keep lines concise and readable.\n"
        f"\n[OBJECT_GRAPH]\n{graph_text}\n"
    )
    input_payload = [
        {"role": "developer", "content": prompt},
        {"role": "user", "content": chunk_text},
    ]

    start = time.time()
    output = _call_openai(args.model, input_payload)
    elapsed = time.time() - start

    print(f"chunk_index={args.chunk_index} chunks={len(chunks)} elapsed_sec={elapsed:.2f}")
    print(f"output_chars={len(output)}")
    out_path = os.path.join("out", "phase2_probe.srt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"output_path={out_path}")


if __name__ == "__main__":
    main()
