"""IO for the hallucination pipeline.

Responsibilities:
  * Load the four sibling JSONL files a run produces (dataset, trajectory,
    answers, scored_results) and join them by ``sample_id``.
  * Write halu_results.jsonl / halu_summary.json.
  * Validate that required runs exist; fail loudly with an actionable message
    when trajectory.jsonl is missing (the runner had --no-emit-trajectory).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable

from .types import HaluResult, SampleRecord


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _index_by_id(rows: list[dict[str, Any]], key: str = "sample_id") -> dict[str, dict[str, Any]]:
    return {str(r.get(key, "")): r for r in rows if r.get(key)}


def discover_model_dirs(runs_dir: Path) -> list[Path]:
    """Return subdirs under runs_dir that look like per-model outputs.

    Criterion: contains trajectory.jsonl (required) AND
    scored_results.jsonl (required for error-trajectory filtering).
    """
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")
    dirs: list[Path] = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        has_traj = (child / "trajectory.jsonl").is_file()
        has_scored = (child / "scored_results.jsonl").is_file()
        if has_traj and has_scored:
            dirs.append(child)
    if not dirs:
        raise FileNotFoundError(
            f"No model subdirs with trajectory.jsonl + scored_results.jsonl found under {runs_dir}. "
            "Re-run evaluation.runner with --emit-trajectory (default on)."
        )
    return dirs


def load_joined(
    model_dir: Path,
    dataset: list[dict[str, Any]],
    *,
    only_errors: bool,
) -> list[SampleRecord]:
    """Join dataset ⋈ trajectory ⋈ answers ⋈ scored_results by sample_id."""
    model = model_dir.name
    trajectory_path = model_dir / "trajectory.jsonl"
    answers_path = model_dir / "answers.jsonl"
    scored_path = model_dir / "scored_results.jsonl"
    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Missing {trajectory_path}. Re-run evaluation.runner with --emit-trajectory."
        )
    if not scored_path.is_file():
        raise FileNotFoundError(
            f"Missing {scored_path}. Re-run evaluation.runner — halu pipeline needs per-sample scoring."
        )

    trajectories = _index_by_id(_load_jsonl(trajectory_path))
    answers = _index_by_id(_load_jsonl(answers_path))
    scored = _index_by_id(_load_jsonl(scored_path))
    dataset_idx = _index_by_id(dataset)

    out: list[SampleRecord] = []
    for sid, traj in trajectories.items():
        s_row = dataset_idx.get(sid)
        sc_row = scored.get(sid)
        if s_row is None or sc_row is None:
            continue
        if only_errors and bool(sc_row.get("is_correct")):
            continue
        out.append(
            SampleRecord(
                sample_id=sid,
                model=model,
                sample=s_row,
                answer=answers.get(sid, {}),
                scored=sc_row,
                trajectory_messages=list(traj.get("messages") or []),
                prompt=str(traj.get("prompt", "")),
            )
        )
    return out


def write_results(
    out_dir: Path,
    results: Iterable[HaluResult],
    summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "halu_results.jsonl"
    summary_path = out_dir / "halu_summary.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        for r in results:
            # Convert nested dataclasses for JSONL
            fh.write(
                json.dumps(
                    _to_jsonable(r),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj
