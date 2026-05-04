"""Unified readers for life-science benchmarks used as Graph-layer seeds.

Each reader returns an iterable of BenchmarkItem. image_ref is a dict that
tells benchmark_images.py where to pull the image bytes from; None means
no image exists (e.g., ProteinLMBench).

Supported keys:
  ProteinLMBench        (no image)
  PathVQA               (parquet, HF Image feature; val/test splits)
  MedXpertQA_MM         (jsonl + images.zip)
  MedQ-Bench            (TSV with base64 image column)  -- discouraged, scope drift
  SLAKE_EN              (parquet, HF Image feature)
  SLAKE_Bilingual       (JSON + imgs.zip, English rows only)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

from .benchmark_triples import BenchmarkItem


# --------------------------------------------------------------------------- #
# ProteinLMBench — CSV, no images
# --------------------------------------------------------------------------- #

def read_proteinlmbench(dataset_root: Path, split: str = "all",
                         question_limit: int = 0) -> Iterator[BenchmarkItem]:
    csv_path = Path(dataset_root)
    if csv_path.is_dir():
        csv_path = csv_path / "ProteinLMBench.csv"
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if question_limit and idx >= question_limit:
                return
            question = (row.get("question") or "").strip()
            if not question:
                continue
            try:
                answer_idx = int(row.get("answer") or 0)
                answer = (row.get(f"option {answer_idx}") or "").strip()
            except (TypeError, ValueError):
                answer = ""
            yield BenchmarkItem(
                dataset="ProteinLMBench",
                split="all",
                question_id=f"ProteinLMBench::{idx}",
                question=question,
                answer=answer,
                question_type="mcq",
                image_ref=None,
            )


# --------------------------------------------------------------------------- #
# PathVQA — parquet shards
# --------------------------------------------------------------------------- #

def read_pathvqa(dataset_root: Path, split: str = "test",
                 question_limit: int = 0) -> Iterator[BenchmarkItem]:
    import pyarrow.parquet as pq  # lazy

    data_dir = Path(dataset_root) / "data"
    prefix = {"test": "test-", "val": "validation-", "validation": "validation-",
              "train": "train-"}.get(split, "test-")
    shards = sorted(p for p in data_dir.glob(f"{prefix}*.parquet"))
    emitted = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["question", "answer"])
        questions = table.column("question").to_pylist()
        answers = table.column("answer").to_pylist()
        for local_idx, (q, a) in enumerate(zip(questions, answers)):
            if question_limit and emitted >= question_limit:
                return
            q = (q or "").strip()
            if not q:
                continue
            yield BenchmarkItem(
                dataset="PathVQA",
                split=split if split != "val" else "validation",
                question_id=f"PathVQA::{split}::{shard.stem}::{local_idx}",
                question=q,
                answer=(a or "").strip(),
                question_type="open",
                image_ref={
                    "type": "parquet",
                    "parquet_path": str(shard),
                    "row": local_idx,
                    "image_column": "image",
                },
            )
            emitted += 1


# --------------------------------------------------------------------------- #
# MedXpertQA MM — JSONL + images.zip
# --------------------------------------------------------------------------- #

def read_medxpertqa_mm(dataset_root: Path, split: str = "test",
                       question_limit: int = 0) -> Iterator[BenchmarkItem]:
    root = Path(dataset_root)
    jsonl = root / "MM" / f"{split}.jsonl"
    zip_path = root / "images.zip"
    emitted = 0
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if question_limit and emitted >= question_limit:
                return
            row = json.loads(line)
            q = (row.get("question") or "").strip()
            if not q:
                continue
            options = row.get("options") or {}
            label = row.get("label") or ""
            if isinstance(options, dict):
                answer = str(options.get(label, label))
            else:
                answer = str(label)
            images = row.get("images") or []
            first_image = images[0] if images else None
            image_ref = None
            if first_image:
                image_ref = {
                    "type": "zip",
                    "zip_path": str(zip_path),
                    "entry": f"images/{first_image}",
                }
            yield BenchmarkItem(
                dataset="MedXpertQA_MM",
                split=split,
                question_id=f"MedXpertQA_MM::{split}::{row.get('id', emitted)}",
                question=q,
                answer=answer,
                question_type=str(row.get("question_type") or "mcq"),
                image_ref=image_ref,
            )
            emitted += 1


# --------------------------------------------------------------------------- #
# SLAKE English (parquet, HF Image feature)
# --------------------------------------------------------------------------- #

def read_slake_en(dataset_root: Path, split: str = "test",
                   question_limit: int = 0) -> Iterator[BenchmarkItem]:
    import pyarrow.parquet as pq

    data_dir = Path(dataset_root) / "data"
    prefix = {"test": "test-", "val": "validation-", "validation": "validation-",
              "train": "train-"}.get(split, "test-")
    shards = sorted(p for p in data_dir.glob(f"{prefix}*.parquet"))
    emitted = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["question", "answer"])
        questions = table.column("question").to_pylist()
        answers = table.column("answer").to_pylist()
        for local_idx, (q, a) in enumerate(zip(questions, answers)):
            if question_limit and emitted >= question_limit:
                return
            q = (q or "").strip()
            if not q:
                continue
            yield BenchmarkItem(
                dataset="SLAKE_EN",
                split=split if split != "val" else "validation",
                question_id=f"SLAKE_EN::{split}::{shard.stem}::{local_idx}",
                question=q,
                answer=(a or "").strip(),
                question_type="vqa",
                image_ref={
                    "type": "parquet",
                    "parquet_path": str(shard),
                    "row": local_idx,
                    "image_column": "image",
                },
            )
            emitted += 1


# --------------------------------------------------------------------------- #
# SLAKE bilingual — filter to English, imgs.zip lookup
# --------------------------------------------------------------------------- #

def read_slake_bilingual(dataset_root: Path, split: str = "test",
                          question_limit: int = 0) -> Iterator[BenchmarkItem]:
    root = Path(dataset_root)
    json_path = root / f"{split}.json"
    zip_path = root / "imgs.zip"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return
    emitted = 0
    for row in payload:
        if question_limit and emitted >= question_limit:
            return
        if not isinstance(row, dict):
            continue
        lang = (row.get("q_lang") or "").lower()
        if lang and lang != "en":
            continue
        q = (row.get("question") or "").strip()
        if not q:
            continue
        img_name = row.get("img_name") or ""
        image_ref = None
        if img_name:
            image_ref = {
                "type": "zip",
                "zip_path": str(zip_path),
                "entry": f"imgs/{img_name}" if not img_name.startswith("imgs/") else img_name,
            }
        yield BenchmarkItem(
            dataset="SLAKE_Bilingual",
            split=split,
            question_id=f"SLAKE_Bilingual::{split}::{row.get('qid', emitted)}",
            question=q,
            answer=str(row.get("answer") or ""),
            question_type=str(row.get("content_type") or row.get("base_type") or "vqa"),
            image_ref=image_ref,
        )
        emitted += 1


# --------------------------------------------------------------------------- #
# MedQ-Bench — TSV with base64 image column
# --------------------------------------------------------------------------- #

def read_medqbench(dataset_root: Path, subset: str = "QA",
                    split: str = "test", question_limit: int = 0) -> Iterator[BenchmarkItem]:
    root = Path(dataset_root)
    fname = {
        "QA":                   f"medqbench_QA_{split}.tsv",
        "description":          f"medqbench_description_{split}.tsv",
        "paired_description":   f"medqbench_paired_description_{split}.tsv",
    }.get(subset, f"medqbench_QA_{split}.tsv")
    tsv = root / fname
    import pandas as pd  # lazy
    df = pd.read_csv(tsv, sep="\t")
    emitted = 0
    for local_idx, row in df.iterrows():
        if question_limit and emitted >= question_limit:
            return
        q = str(row.get("question") or "").strip()
        if not q:
            continue
        answer = str(row.get("answer") or "")
        yield BenchmarkItem(
            dataset=f"MedQBench_{subset}",
            split=split,
            question_id=f"MedQBench_{subset}::{split}::{local_idx}",
            question=q,
            answer=answer,
            question_type=str(row.get("question_type") or "image_quality"),
            image_ref={
                "type": "base64_tsv",
                "tsv_path": str(tsv),
                "row": int(local_idx),
                "column": "image",
            },
        )
        emitted += 1


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

READERS = {
    "ProteinLMBench":     read_proteinlmbench,
    "PathVQA":            read_pathvqa,
    "MedXpertQA_MM":      read_medxpertqa_mm,
    "SLAKE_EN":           read_slake_en,
    "SLAKE_Bilingual":    read_slake_bilingual,
    "MedQBench":          read_medqbench,
}


def list_readers() -> list[str]:
    return sorted(READERS.keys())


def read_benchmark(name: str, dataset_root: Path, **kwargs) -> list[BenchmarkItem]:
    if name not in READERS:
        raise ValueError(f"Unknown benchmark: {name!r}. Available: {list_readers()}")
    return list(READERS[name](Path(dataset_root), **kwargs))
