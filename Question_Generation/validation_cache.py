from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pubmed_graph.utils import ensure_dir, load_json, write_json


def _cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ValidationCache:
    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            ensure_dir(self.cache_dir)

    def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{_cache_key(payload)}.json"
        if not path.exists():
            return None
        return load_json(path)

    def put(self, payload: dict[str, Any], result: dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{_cache_key(payload)}.json"
        write_json(path, result)
