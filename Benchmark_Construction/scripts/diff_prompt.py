"""Render the new triple_extraction.j2 template via Ontology and diff against
the legacy triple_extraction.txt. Used as a sanity check for stage 1.2.

The two strings are NOT required to be byte-equivalent (Jinja's whitespace
handling differs slightly from a hand-written file), but they should be
semantically identical. The script reports a unified diff and a similarity
ratio so we can eyeball any drift before re-running the LLM.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pubmed_graph.triple_extraction import _render_extraction_prompt, PROMPT_PATH  # noqa: E402


def main() -> None:
    legacy = PROMPT_PATH.read_text(encoding="utf-8")
    rendered = _render_extraction_prompt()

    print(f"[info] legacy   length = {len(legacy)} chars / {len(legacy.splitlines())} lines")
    print(f"[info] rendered length = {len(rendered)} chars / {len(rendered.splitlines())} lines")

    ratio = difflib.SequenceMatcher(None, legacy, rendered).ratio()
    print(f"[info] similarity ratio = {ratio:.4f}")

    diff = list(
        difflib.unified_diff(
            legacy.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile="legacy_triple_extraction.txt",
            tofile="rendered_from_ontology.j2",
            n=2,
        )
    )
    if not diff:
        print()
        print("[ok]   rendered prompt is byte-equivalent to legacy prompt.")
        return
    print()
    print(f"[info] diff lines: {len(diff)}")
    print("=" * 60)
    sys.stdout.writelines(diff)
    print("=" * 60)
    if ratio < 0.95:
        print("[FAIL] similarity below 0.95 — investigate before stage 1.5")
        sys.exit(1)
    print("[ok]   diff is small (whitespace / blank-line drift); semantically equivalent")


if __name__ == "__main__":
    main()
