"""Dedup sample_id collisions in v2 bench files.

Collisions appear because two_hop_patch / expcode_rerun appended new questions
that re-used the original qg_NNNNNN namespace. Each duplicate id maps to a
genuinely different question, so we keep all rows but rename later occurrences
with `__d{N}` suffixes (first occurrence stays unchanged so existing references
keep working).

Usage:
  python evaluation/scripts/dedup_sample_ids.py path/to/samples.jsonl

Behavior:
  * Backs up `<file>` → `<file>.bak_pre_dedup` (skipped if backup already exists)
  * Rewrites `<file>` in-place with deduplicated sample_ids
  * Prints rename summary
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def dedup_file(path: Path, *, dry_run: bool = False) -> dict:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    counts = Counter(r.get("sample_id", "") for r in rows)
    n_dups = sum(1 for v in counts.values() if v > 1)
    if n_dups == 0:
        print(f"[dedup] {path}: {len(rows)} rows, no duplicates. Nothing to do.")
        return {"path": str(path), "n_rows": len(rows), "renames": []}

    seen: dict[str, int] = {}
    renames: list[dict] = []
    for r in rows:
        sid = r.get("sample_id", "")
        if not sid:
            continue
        if sid in seen:
            seen[sid] += 1
            new_sid = f"{sid}__d{seen[sid]}"
            renames.append({"original": sid, "new": new_sid})
            r["sample_id"] = new_sid
        else:
            seen[sid] = 1

    print(f"[dedup] {path}: {len(rows)} rows, {n_dups} colliding ids, {len(renames)} renames.")
    for ren in renames[:5]:
        print(f"  {ren['original']} -> {ren['new']}")
    if len(renames) > 5:
        print(f"  ... and {len(renames) - 5} more")

    if dry_run:
        return {"path": str(path), "n_rows": len(rows), "renames": renames, "dry_run": True}

    backup = path.with_suffix(path.suffix + ".bak_pre_dedup")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  backup -> {backup}")
    else:
        print(f"  backup already exists at {backup}; not overwriting")

    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  rewrote {path}")
    return {"path": str(path), "n_rows": len(rows), "renames": renames}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSONL files to dedup (in place).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    summary = []
    for p in args.paths:
        path = Path(p)
        if not path.is_file():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            return 2
        summary.append(dedup_file(path, dry_run=args.dry_run))
    print()
    print(f"DONE. {len(summary)} files processed.")
    total_renames = sum(len(s.get("renames", [])) for s in summary)
    print(f"  total renames: {total_renames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
