from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .models import Span


def _tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return [tok for tok in cleaned.split() if tok]


def extract_entities(spans: List[Span], *, min_count: int = 2) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Naive entity extractor: capitalized tokens and long tokens by frequency.
    Returns (entities, mentions).
    """
    counts: Counter[str] = Counter()
    mentions: List[Tuple[str, str]] = []  # (sid, surface)

    for span in spans:
        tokens = _tokenize(span.text_raw)
        for tok in tokens:
            if len(tok) < 3:
                continue
            if tok[0].isupper() or tok.isupper():
                counts[tok] += 1
                mentions.append((span.sid, tok))

    entities: List[Dict[str, str]] = []
    mention_blocks: List[Dict[str, str]] = []

    entity_id_map: Dict[str, str] = {}
    eid_index = 1
    for surface, cnt in counts.items():
        if cnt < min_count:
            continue
        eid = f"E{eid_index:04d}"
        entity_id_map[surface] = eid
        entities.append(
            {
                "eid": eid,
                "etype": "Unknown",
                "canon_name": surface,
                "conf": f"{min(1.0, 0.5 + 0.1 * cnt):.2f}",
            }
        )
        eid_index += 1

    for sid, surface in mentions:
        eid = entity_id_map.get(surface)
        if not eid:
            continue
        mention_blocks.append(
            {
                "mid": f"M_{sid}_{surface}",
                "sid": sid,
                "eid": eid,
                "surface": surface,
                "conf": "0.80",
            }
        )

    return entities, mention_blocks
