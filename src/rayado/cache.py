from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Optional

from .utils import ensure_dir


class Cache:
    def __init__(self, path: str) -> None:
        self.path = path
        ensure_dir(os.path.dirname(path))
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asr_cache (
                  key TEXT PRIMARY KEY,
                  request_hash TEXT NOT NULL,
                  response TEXT NOT NULL,
                  created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asr_cache_request_hash ON asr_cache (request_hash)"
            )

    def get(self, key: str, request_hash: str) -> Optional[dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT response, request_hash FROM asr_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        response_json, stored_hash = row
        if stored_hash != request_hash:
            return None
        return json.loads(response_json)

    def set(self, key: str, request_hash: str, response: dict[str, Any]) -> None:
        payload = json.dumps(response, ensure_ascii=False)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO asr_cache (key, request_hash, response, created_at) VALUES (?, ?, ?, ?)",
                (key, request_hash, payload, int(time.time())),
            )
