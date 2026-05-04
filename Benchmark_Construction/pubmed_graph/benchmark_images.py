"""On-demand image materialization for benchmark overlay.

The readers store image_ref dicts that describe WHERE an image lives
(parquet row, zip entry, base64 cell, etc.) rather than its final path.
This module materializes the subset of images that corresponds to
benchmark items that actually contributed triples to the graph, writing
each to a deterministic path under <output_dir>/benchmark_images/<dataset>/<qid>.<ext>.

Items with image_ref=None (e.g., ProteinLMBench) are no-ops.
"""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from .benchmark_triples import BenchmarkItem


_PARQUET_CACHE: dict[str, Any] = {}


def _safe_qid(item: BenchmarkItem) -> str:
    return item.question_id.replace("::", "_").replace("/", "_")


def _write_bytes(dest: Path, data: bytes, suffix: str) -> Path:
    dest = dest.with_suffix(suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def _materialize_parquet(ref: dict[str, Any], dest: Path) -> Path | None:
    import pyarrow.parquet as pq  # lazy

    path = ref["parquet_path"]
    column = ref.get("image_column", "image")
    key = f"{path}:{column}"
    if key not in _PARQUET_CACHE:
        table = pq.read_table(path, columns=[column])
        _PARQUET_CACHE[key] = table.column(column).to_pylist()
    rows = _PARQUET_CACHE[key]
    row_idx = int(ref["row"])
    if row_idx >= len(rows):
        return None
    entry = rows[row_idx]
    if isinstance(entry, dict):
        data = entry.get("bytes")
        name = entry.get("path") or ""
    elif isinstance(entry, (bytes, bytearray)):
        data, name = entry, ""
    else:
        return None
    if not data:
        return None
    suffix = Path(name).suffix.lower() if name else ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        suffix = ".jpg"
    return _write_bytes(dest, bytes(data), suffix)


def _materialize_zip(ref: dict[str, Any], dest: Path) -> Path | None:
    zip_path = ref["zip_path"]
    entry = ref["entry"]
    with zipfile.ZipFile(zip_path) as z:
        try:
            with z.open(entry) as src:
                data = src.read()
        except KeyError:
            alt = entry.split("/", 1)[-1] if "/" in entry else entry
            try:
                with z.open(alt) as src:
                    data = src.read()
            except KeyError:
                return None
    suffix = Path(entry).suffix.lower() or ".jpg"
    return _write_bytes(dest, data, suffix)


def _materialize_base64_tsv(ref: dict[str, Any], dest: Path) -> Path | None:
    import pandas as pd  # lazy

    path = ref["tsv_path"]
    key = f"tsv::{path}"
    if key not in _PARQUET_CACHE:
        _PARQUET_CACHE[key] = pd.read_csv(path, sep="\t")
    df = _PARQUET_CACHE[key]
    row_idx = int(ref["row"])
    if row_idx >= len(df):
        return None
    value = df.iloc[row_idx][ref.get("column", "image")]
    if not isinstance(value, str) or len(value) < 100:
        return None
    try:
        data = base64.b64decode(value)
    except Exception:
        return None
    return _write_bytes(dest, data, ".jpg")


def materialize(item: BenchmarkItem, out_root: Path) -> str:
    """Return absolute path to materialized image or empty string on skip/fail."""
    if item.image_ref is None:
        return ""
    dest = out_root / item.dataset / _safe_qid(item)
    ref_type = item.image_ref.get("type", "")
    try:
        if ref_type == "parquet":
            path = _materialize_parquet(item.image_ref, dest)
        elif ref_type == "zip":
            path = _materialize_zip(item.image_ref, dest)
        elif ref_type == "base64_tsv":
            path = _materialize_base64_tsv(item.image_ref, dest)
        elif ref_type == "file":
            src = Path(item.image_ref["path"])
            if src.exists():
                path = _write_bytes(dest, src.read_bytes(), src.suffix or ".jpg")
            else:
                path = None
        else:
            return ""
    except Exception as exc:
        print(f"  [image] failed to materialize {item.question_id}: {exc}")
        return ""
    return str(path.resolve()) if path else ""


def materialize_used_images(
    items: list[BenchmarkItem],
    out_root: Path,
) -> dict[str, str]:
    """Materialize images for the given items; returns question_id -> abs_path."""
    out_root = Path(out_root)
    resolved: dict[str, str] = {}
    for item in items:
        path = materialize(item, out_root)
        if path:
            resolved[item.question_id] = path
    return resolved
