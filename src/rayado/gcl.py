from __future__ import annotations

import os
from typing import Dict, Iterable


def ensure_header(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("GCL_HDR\n")
        f.write("ver 0.2\n")
        f.write("mode append_only\n")
        f.write("time_unit seconds_float\n\n")


def append_block(path: str, block_type: str, fields: Dict[str, str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{block_type}\n")
        for key, value in fields.items():
            f.write(f"{key} {value}\n")
        f.write("\n")


def append_blocks(path: str, blocks: Iterable[tuple[str, Dict[str, str]]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for block_type, fields in blocks:
            f.write(f"{block_type}\n")
            for key, value in fields.items():
                f.write(f"{key} {value}\n")
            f.write("\n")
