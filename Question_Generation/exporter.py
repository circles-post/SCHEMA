from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from pubmed_graph.utils import ensure_dir, write_json, write_jsonl

from .models import QuestionSample


def export_samples(samples: list[QuestionSample], output_path: str | Path) -> None:
    out = Path(output_path)
    ensure_dir(out.parent)
    write_jsonl(out, [asdict(sample) for sample in samples])


def export_summary(summary: dict, output_path: str | Path) -> None:
    out = Path(output_path)
    ensure_dir(out.parent)
    write_json(out, summary)
