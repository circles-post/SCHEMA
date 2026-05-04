"""Post-hoc option-order shuffler for a samples.jsonl.

Reads a question-generation output file and rewrites each multi-choice
sample with its ``options`` list reordered, preserving ``is_correct``
markers (the correct option still exists, just at a random position).
``answer.text`` / ``answer.canonical_text`` are unchanged.

Deterministic per sample_id so re-running gives identical output, and
different samples get different permutations.

Usage
-----
    python -m question_generation.scripts.shuffle_options \\
      --input  path/to/samples.jsonl \\
      --output path/to/samples.shuffled.jsonl      # optional; default rewrites input

    # dry run: just report the before/after position histograms
    python -m question_generation.scripts.shuffle_options \\
      --input samples.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

LETTERS = "ABCDEFGHIJKL"


def _seed_for(sample_id: str, salt: str) -> int:
    """Stable deterministic seed from (sample_id, salt)."""
    h = hashlib.sha1(f"{salt}::{sample_id}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def shuffle_sample_options(sample: dict, salt: str = "shuffle_v1") -> dict:
    """Return a new sample dict with options permuted. No-op if <2 options.

    Works on dicts (jsonl rows), not dataclasses — so it's safe to apply
    after ``exporter.export_samples`` has serialized the samples.
    """
    opts = sample.get("options") or []
    if len(opts) < 2:
        return sample
    sid = str(sample.get("sample_id", ""))
    rng = random.Random(_seed_for(sid, salt))
    perm = list(range(len(opts)))
    rng.shuffle(perm)
    new_opts = [opts[i] for i in perm]
    sample = dict(sample)
    sample["options"] = new_opts
    return sample


def _position_histogram(rows: list[dict]) -> dict[str, Counter]:
    """{question_type: Counter of correct-answer positions 'A'/'B'/...}."""
    hist: dict[str, Counter] = {}
    for r in rows:
        opts = r.get("options") or []
        if len(opts) < 2:
            continue
        qt = r.get("question_type", "")
        idx = next((i for i, o in enumerate(opts) if o.get("is_correct")), -1)
        if idx < 0:
            continue
        hist.setdefault(qt, Counter())[LETTERS[idx]] += 1
    return hist


def _format_hist(h: dict[str, Counter]) -> str:
    lines = []
    for qt, counts in sorted(h.items()):
        total = sum(counts.values())
        top_letter, top_n = counts.most_common(1)[0]
        skew = 100 * top_n / total if total else 0.0
        dist = ", ".join(f"{L}={counts.get(L, 0)}" for L in LETTERS if counts.get(L))
        lines.append(f"  {qt:20s}  n={total}  [{dist}]   top={top_letter} ({skew:.1f}%)")
    return "\n".join(lines) or "  (no multichoice samples)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="samples.jsonl to read")
    p.add_argument("--output", default=None,
                   help="output path. Defaults to overwriting --input (keeps a .bak_unshuffled).")
    p.add_argument("--salt", default="shuffle_v1",
                   help="salt mixed with sample_id for determinism; change to reshuffle.")
    p.add_argument("--dry-run", action="store_true",
                   help="print before/after histograms but don't write output.")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(2)

    rows = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"loaded {len(rows)} samples from {in_path}", file=sys.stderr)

    before = _position_histogram(rows)
    print("\nBEFORE shuffle — correct-answer position distribution:", file=sys.stderr)
    print(_format_hist(before), file=sys.stderr)

    shuffled = [shuffle_sample_options(r, salt=args.salt) for r in rows]

    after = _position_histogram(shuffled)
    print("\nAFTER shuffle — correct-answer position distribution:", file=sys.stderr)
    print(_format_hist(after), file=sys.stderr)

    if args.dry_run:
        print("\n(dry-run — no file written)", file=sys.stderr)
        return

    out_path = Path(args.output) if args.output else in_path
    if out_path == in_path:
        backup = in_path.with_suffix(in_path.suffix + ".bak_unshuffled")
        if not backup.exists():
            shutil.copy2(in_path, backup)
            print(f"\nbacked up original to {backup}", file=sys.stderr)
        else:
            print(f"\nbackup {backup} already exists — not overwriting", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in shuffled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(shuffled)} samples → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
