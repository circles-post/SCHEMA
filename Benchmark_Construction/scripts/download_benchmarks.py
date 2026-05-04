"""Download life-science benchmarks for the pubmed_graph pipeline.

Pulls each dataset into <output-dir>/<short-name>/ via huggingface_hub
snapshot_download (resumable, hash-checked) and, when possible, writes a
flat <short-name>.csv with a `question` column so it can be consumed
directly by pubmed_graph.benchmark_seeds.LocalBenchmarkDataset.

Datasets handled:
  path-vqa         flaviagiammarino/path-vqa            (train/val/test)
  OmniMedVQA       foreverbeliever/OmniMedVQA           (open-access subset)
  MedXpertQA       TsinghuaC3I/MedXpertQA               (MM/test)
  MedQ-Bench       jiyaoliufd/MedQ-Bench                (Perception + Reasoning)
  SLAKE-en         mdwiratathya/SLAKE-vqa-english       (test split)
  SLAKE-bilingual  BoKelvin/SLAKE                       (optional mirror)

Usage:
  python scripts/download_benchmarks.py --output-dir /path/to/datasets
  python scripts/download_benchmarks.py --only path-vqa,medxpertqa
  python scripts/download_benchmarks.py --list
  python scripts/download_benchmarks.py --skip omnimedvqa --no-extract-csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download


DEFAULT_OUTPUT_DIR = "/mnt/shared-storage-user/fengxinshun/AISci/datasets"


@dataclass
class BenchmarkSpec:
    key: str
    repo_id: str
    subdir: str
    description: str
    extractor: Callable[[Path, Path, bool], dict] | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None


# --------------------------------------------------------------------------- #
# CSV extractors — each writes a flat <name>.csv with at least a "question"
# column so pubmed_graph.benchmark_seeds.LocalBenchmarkDataset can consume it.
# --------------------------------------------------------------------------- #

def _iter_dataset(repo_dir: Path, config_name: str | None, split: str | None):
    """Load a dataset via `datasets` lib, returning rows as dicts.

    Local load is tried first to avoid re-downloading; on failure fall back
    to the HF repo id.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(str(repo_dir), name=config_name, split=split)
    except Exception:
        ds = load_dataset(repo_dir.name, name=config_name, split=split)
    for row in ds:
        yield row


def _write_csv(rows: list[dict], path: Path, columns: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return len(rows)


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def extract_path_vqa(repo_dir: Path, out_dir: Path, verbose: bool) -> dict:
    from datasets import load_dataset

    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset(str(repo_dir), split=split)
        except Exception as exc:
            if verbose:
                print(f"  [path-vqa] skip split {split}: {exc}")
            continue
        for idx, row in enumerate(ds):
            rows.append({
                "index": len(rows),
                "split": split,
                "question": _clean_text(row.get("question")),
                "answer": _clean_text(row.get("answer")),
                "image_id": _clean_text(row.get("image")) if not isinstance(row.get("image"), dict) else "",
            })
    n = _write_csv(rows, out_dir / "path-vqa.csv",
                   ["index", "split", "question", "answer", "image_id"])
    return {"rows": n, "csv": str(out_dir / "path-vqa.csv")}


def extract_medxpertqa(repo_dir: Path, out_dir: Path, verbose: bool) -> dict:
    from datasets import load_dataset

    rows: list[dict] = []
    try:
        ds = load_dataset(str(repo_dir), name="MM", split="test")
    except Exception as exc:
        return {"error": f"load_dataset MM/test failed: {exc}"}
    for row in ds:
        rows.append({
            "index": len(rows),
            "question": _clean_text(row.get("question")),
            "answer": _clean_text(row.get("label") or row.get("answer")),
        })
    n = _write_csv(rows, out_dir / "MedXpertQA_MM_test.csv",
                   ["index", "question", "answer"])
    return {"rows": n, "csv": str(out_dir / "MedXpertQA_MM_test.csv")}


def extract_omnimedvqa(repo_dir: Path, out_dir: Path, verbose: bool) -> dict:
    """OmniMedVQA doesn't ship a HF loader script; scan its JSON files."""
    import json as _json

    question_keys = ("question", "question_text", "query")
    rows: list[dict] = []
    json_files = list(repo_dir.rglob("*.json"))
    if verbose:
        print(f"  [OmniMedVQA] scanning {len(json_files)} json files")
    for jf in json_files:
        try:
            payload = _json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload = payload.get("data") if isinstance(payload.get("data"), list) else [payload]
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            question = ""
            for k in question_keys:
                if entry.get(k):
                    question = _clean_text(entry.get(k))
                    break
            if not question:
                continue
            rows.append({
                "index": len(rows),
                "source_file": str(jf.relative_to(repo_dir)),
                "question": question,
                "answer": _clean_text(entry.get("gt_answer") or entry.get("answer") or ""),
            })
    n = _write_csv(rows, out_dir / "OmniMedVQA.csv",
                   ["index", "source_file", "question", "answer"])
    return {"rows": n, "csv": str(out_dir / "OmniMedVQA.csv")}


def extract_medqbench(repo_dir: Path, out_dir: Path, verbose: bool) -> dict:
    from datasets import load_dataset

    targets = [
        ("MedQ-Perception",             "MedQBench_MCQ"),
        ("MedQ-Reasoning",              "MedQBench_Reasoning"),
    ]
    summary = {}
    for config_name, filename in targets:
        rows: list[dict] = []
        for split in ("test", "validation", "train"):
            try:
                ds = load_dataset(str(repo_dir), name=config_name, split=split)
            except Exception:
                continue
            for row in ds:
                rows.append({
                    "index": len(rows),
                    "split": split,
                    "question": _clean_text(
                        row.get("question") or row.get("query") or row.get("prompt")
                    ),
                    "answer": _clean_text(
                        row.get("answer") or row.get("label") or row.get("response")
                    ),
                })
        if rows:
            n = _write_csv(rows, out_dir / f"{filename}.csv",
                           ["index", "split", "question", "answer"])
            summary[filename] = {"rows": n, "csv": str(out_dir / f"{filename}.csv")}
        else:
            summary[filename] = {"rows": 0, "note": "no loadable splits"}
    return summary


def extract_slake_en(repo_dir: Path, out_dir: Path, verbose: bool) -> dict:
    from datasets import load_dataset

    rows: list[dict] = []
    for split in ("test", "validation", "train"):
        try:
            ds = load_dataset(str(repo_dir), split=split)
        except Exception:
            continue
        for row in ds:
            rows.append({
                "index": len(rows),
                "split": split,
                "question": _clean_text(row.get("question")),
                "answer": _clean_text(row.get("answer")),
                "answer_type": _clean_text(row.get("answer_type") or ""),
                "modality": _clean_text(row.get("modality") or ""),
            })
    n = _write_csv(rows, out_dir / "SLAKE_EN_TEST.csv",
                   ["index", "split", "question", "answer", "answer_type", "modality"])
    return {"rows": n, "csv": str(out_dir / "SLAKE_EN_TEST.csv")}


def extract_slake_bilingual(repo_dir: Path, out_dir: Path, verbose: bool) -> dict:
    """BoKelvin/SLAKE ships JSON files; filter to English rows."""
    import json as _json

    rows: list[dict] = []
    for jf in sorted(repo_dir.rglob("*.json")):
        try:
            payload = _json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            lang = (entry.get("q_lang") or entry.get("lang") or "").lower()
            if lang and lang != "en":
                continue
            q = _clean_text(entry.get("question"))
            if not q:
                continue
            rows.append({
                "index": len(rows),
                "source_file": str(jf.relative_to(repo_dir)),
                "question": q,
                "answer": _clean_text(entry.get("answer") or ""),
                "answer_type": _clean_text(entry.get("answer_type") or ""),
                "modality": _clean_text(entry.get("modality") or ""),
            })
    n = _write_csv(rows, out_dir / "SLAKE_EN_bilingual.csv",
                   ["index", "source_file", "question", "answer", "answer_type", "modality"])
    return {"rows": n, "csv": str(out_dir / "SLAKE_EN_bilingual.csv")}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

BENCHMARKS: dict[str, BenchmarkSpec] = {
    "path-vqa": BenchmarkSpec(
        key="path-vqa",
        repo_id="flaviagiammarino/path-vqa",
        subdir="path-vqa",
        description="Pathology image VQA (train/val/test)",
        extractor=extract_path_vqa,
    ),
    "omnimedvqa": BenchmarkSpec(
        key="omnimedvqa",
        repo_id="foreverbeliever/OmniMedVQA",
        subdir="OmniMedVQA",
        description="Medical multimodal MCQ (open-access subset ~ tens of GB)",
        extractor=extract_omnimedvqa,
    ),
    "medxpertqa": BenchmarkSpec(
        key="medxpertqa",
        repo_id="TsinghuaC3I/MedXpertQA",
        subdir="MedXpertQA",
        description="Expert-level medical MM/test",
        extractor=extract_medxpertqa,
    ),
    "medq-bench": BenchmarkSpec(
        key="medq-bench",
        repo_id="jiyaoliufd/MedQ-Bench",
        subdir="MedQ-Bench",
        description="Medical image quality (Perception MCQ + Reasoning)",
        extractor=extract_medqbench,
    ),
    "slake-en": BenchmarkSpec(
        key="slake-en",
        repo_id="mdwiratathya/SLAKE-vqa-english",
        subdir="SLAKE-vqa-english",
        description="English-filtered SLAKE",
        extractor=extract_slake_en,
    ),
    "slake-bilingual": BenchmarkSpec(
        key="slake-bilingual",
        repo_id="BoKelvin/SLAKE",
        subdir="SLAKE-bilingual",
        description="Original bilingual SLAKE (self-filter English)",
        extractor=extract_slake_bilingual,
    ),
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def download_one(spec: BenchmarkSpec, output_root: Path, extract_csv: bool,
                 verbose: bool, dry_run: bool) -> dict:
    target = output_root / spec.subdir
    print(f"\n=== {spec.key}  ({spec.repo_id}) ===")
    print(f"  target      : {target}")
    print(f"  description : {spec.description}")
    if dry_run:
        print("  [dry-run] skipping snapshot_download")
        return {"status": "skipped-dry-run"}

    target.mkdir(parents=True, exist_ok=True)
    try:
        local_path = snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=spec.allow_patterns,
            ignore_patterns=spec.ignore_patterns,
        )
    except Exception as exc:
        print(f"  [FAIL] snapshot_download: {exc}")
        if verbose:
            traceback.print_exc()
        return {"status": "download-failed", "error": str(exc)}
    print(f"  [ok] snapshot at {local_path}")

    result = {"status": "ok", "local_path": local_path}
    if extract_csv and spec.extractor is not None:
        try:
            info = spec.extractor(Path(local_path), output_root, verbose)
            result["extraction"] = info
            print(f"  [ok] csv extraction: {info}")
        except Exception as exc:
            print(f"  [WARN] csv extraction failed: {exc}")
            if verbose:
                traceback.print_exc()
            result["extraction"] = {"error": str(exc)}
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download life-science benchmarks for the pubmed_graph pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Root directory for downloaded datasets")
    parser.add_argument("--only", default="",
                        help="Comma-separated subset of dataset keys to download")
    parser.add_argument("--skip", default="",
                        help="Comma-separated dataset keys to skip")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks and exit")
    parser.add_argument("--no-extract-csv", action="store_true",
                        help="Skip the text-only CSV extraction step")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without downloading")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full tracebacks for failures")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.list:
        print("Available benchmarks (use --only key1,key2):")
        for key, spec in BENCHMARKS.items():
            print(f"  {key:<18s} {spec.repo_id:<40s}  {spec.description}")
        return 0

    only = {k.strip().lower() for k in args.only.split(",") if k.strip()}
    skip = {k.strip().lower() for k in args.skip.split(",") if k.strip()}
    if only:
        unknown = only - set(BENCHMARKS.keys())
        if unknown:
            print(f"[FAIL] unknown --only keys: {sorted(unknown)}", file=sys.stderr)
            return 2
    selected = [spec for key, spec in BENCHMARKS.items()
                if (not only or key in only) and key not in skip]
    if not selected:
        print("[FAIL] no benchmarks selected", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"output dir   : {output_root}")
    print(f"selected     : {[s.key for s in selected]}")
    print(f"extract_csv  : {not args.no_extract_csv}")
    print(f"dry_run      : {args.dry_run}")
    print(f"HF_ENDPOINT  : {os.environ.get('HF_ENDPOINT', '<default>')}")
    print(f"HF_HOME      : {os.environ.get('HF_HOME', '<default>')}")
    print("=" * 60)

    results = {}
    for spec in selected:
        results[spec.key] = download_one(
            spec, output_root,
            extract_csv=not args.no_extract_csv,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok = sum(1 for r in results.values() if r.get("status") == "ok")
    for key, r in results.items():
        status = r.get("status", "?")
        extra = ""
        ext = r.get("extraction") or {}
        if isinstance(ext, dict) and ext.get("rows") is not None:
            extra = f" (csv rows={ext['rows']})"
        elif isinstance(ext, dict) and "error" in ext:
            extra = f" (csv failed: {ext['error'][:60]})"
        elif isinstance(ext, dict) and ext:
            extra = f" (subtasks={list(ext.keys())})"
        print(f"  {key:<18s} {status}{extra}")
    print(f"\n{ok}/{len(results)} downloaded cleanly")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
