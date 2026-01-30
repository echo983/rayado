from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Iterable


def run_cmd(args: Iterable[str], *, capture_stderr: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else None,
        text=True,
        check=False,
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
