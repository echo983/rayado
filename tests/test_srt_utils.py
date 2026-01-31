from __future__ import annotations

from rayado.srt_utils import SrtBlock, chunk_srt_blocks, format_srt_blocks, parse_srt_blocks


def test_srt_roundtrip():
    blocks = [
        SrtBlock(start=0.0, end=1.0, text="hello"),
        SrtBlock(start=1.1, end=2.0, text="world"),
    ]
    srt = format_srt_blocks(blocks)
    parsed = parse_srt_blocks(srt)
    assert parsed == blocks


def test_chunk_srt_blocks_respects_limit():
    blocks = [
        SrtBlock(start=0.0, end=1.0, text="hello"),
        SrtBlock(start=1.1, end=2.0, text="world"),
        SrtBlock(start=2.2, end=3.0, text="again"),
    ]
    chunks = chunk_srt_blocks(blocks, max_chars=30)
    assert len(chunks) >= 2
    assert sum(len(c) for c in chunks) == len(blocks)
