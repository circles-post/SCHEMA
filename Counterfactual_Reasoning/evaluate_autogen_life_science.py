"""Evaluate the AutoGen agent on life science datasets.

Currently this runner targets datasets that expose both:
1. a row -> prompt formatter
2. an evaluation function over predictions

At the moment that means `ProteinLMBench`, whose loader already defines
`build_prompt(...)` and `evaluate(...)`.

Usage:
    python evaluate_autogen_life_science.py --dataset ProteinLMBench --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from dataset.life_science_datasets import get_dataset, list_datasets


def _require_supported_dataset(dataset: Any, dataset_name: str) -> None:
    if not hasattr(dataset, "build_prompt") or not callable(dataset.build_prompt):
        raise ValueError(
            f"Dataset {dataset_name!r} does not implement build_prompt(...). "
            "This evaluator currently supports ProteinLMBench-style datasets only."
        )
    if not hasattr(dataset, "evaluate") or not callable(dataset.evaluate):
        raise ValueError(
            f"Dataset {dataset_name!r} does not implement evaluate(...). "
            "This evaluator currently supports ProteinLMBench-style datasets only."
        )


def _format_task(dataset: Any, row: dict[str, Any]) -> str:
    base_prompt = dataset.build_prompt(row).rstrip()
    prediction_mode = (
        dataset.prediction_mode() if hasattr(dataset, "prediction_mode") else "mcq"
    )
    if prediction_mode == "mcq":
        answer_hint = (
            "\n\nFinal answer requirement:\n"
            "- Put only the final option identifier inside <answer>...</answer>.\n"
            "- Example: <answer>3</answer> or <answer>B</answer>\n"
            "- After the answer, output TERMINATE on a new line.\n"
        )
    else:
        answer_hint = (
            "\n\nFinal answer requirement:\n"
            "- Put your final concise answer inside <answer>...</answer>.\n"
            "- Example: <answer>pneumonia</answer>\n"
            "- After the answer, output TERMINATE on a new line.\n"
        )
    return base_prompt + answer_hint


def _extract_answer_tag(response: str) -> str | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _build_wrong_record(
    *,
    sample_index: int,
    row: dict[str, Any],
    task: str,
    response: str,
    extracted_answer: str | None,
    normalized_prediction: str,
    normalized_answer: str,
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "question": row.get("question", ""),
        "ground_truth": row.get("answer", ""),
        "normalized_ground_truth": normalized_answer,
        "prediction_raw": extracted_answer,
        "prediction_normalized": normalized_prediction,
        "full_response": response,
        "task": task,
        "row": row,
    }


def _sanitize_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text or "unknown"


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


async def evaluate_dataset(
    *,
    dataset_name: str,
    model_name: str,
    cache_dir: str | None,
    start: int,
    limit: int | None,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    try:
        from test_autogen import build_team, get_runtime_model_config, run_task
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AutoGen dependencies are not installed in the current environment. "
            "Install the packages required by test_autogen.py before running this evaluator."
        ) from exc

    runtime_model_config = get_runtime_model_config()
    actual_model_name = runtime_model_config["model_name"]
    print(
        "[evaluator] Effective AutoGen model config: "
        f"model={actual_model_name}, "
        f"base_url={runtime_model_config['api_base_url']}, "
        f"api_key={runtime_model_config['api_key_masked']}"
    )

    dataset = get_dataset(dataset_name, cache_dir=cache_dir)
    _require_supported_dataset(dataset, dataset_name)

    df = dataset.load().copy()
    total_available = len(df)
    if start < 0 or start >= total_available:
        raise ValueError(f"--start must be in [0, {total_available - 1}], got {start}.")

    end = total_available if limit is None else min(start + limit, total_available)
    subset = df.iloc[start:end].copy().reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    wrong_path = output_dir / "wrong_questions.jsonl"
    summary_path = output_dir / "summary.json"

    if overwrite:
        for path in (predictions_path, wrong_path, summary_path):
            if path.exists():
                path.unlink()

    existing_prediction_rows = _load_jsonl_records(predictions_path)
    completed_by_index = {
        int(record["dataset_index"]): record
        for record in existing_prediction_rows
        if "dataset_index" in record
    }
    completed_indexes_in_range = {
        idx for idx in completed_by_index if start <= idx < end
    }

    team, workbench, model_client = await build_team()
    prediction_rows: list[dict[str, Any]] = list(existing_prediction_rows)
    wrong_rows: list[dict[str, Any]] = _load_jsonl_records(wrong_path)

    try:
        for local_idx, (_, series) in enumerate(subset.iterrows()):
            sample_index = start + local_idx
            if sample_index in completed_by_index:
                existing = completed_by_index[sample_index]
                print(f"\n{'#' * 72}")
                print(f"# Sample {local_idx + 1}/{len(subset)} (dataset index {sample_index})")
                print("# Skipped: already evaluated in existing predictions.jsonl")
                print(f"# Existing correctness: {existing.get('is_correct')}")
                print("#" * 72)
                continue

            row = dict(series)
            task = _format_task(dataset, row)

            print(f"\n{'#' * 72}")
            print(f"# Sample {local_idx + 1}/{len(subset)} (dataset index {sample_index})")
            print(f"# Question: {str(row.get('question', ''))[:180]}")
            print("#" * 72)

            await team.reset()
            response = await run_task(team, task)

            extracted_answer = _extract_answer_tag(response)
            normalized_prediction = dataset.normalize_prediction(extracted_answer or response)
            normalized_answer = dataset.normalize_answer(str(row.get("answer", "")))
            is_correct = normalized_prediction == normalized_answer

            prediction_row = row.copy()
            prediction_row["dataset_index"] = sample_index
            prediction_row["prediction"] = response
            prediction_row["prediction_raw_answer"] = extracted_answer
            prediction_row["prediction_normalized"] = normalized_prediction
            prediction_row["answer_normalized"] = normalized_answer
            prediction_row["is_correct"] = int(is_correct)
            prediction_rows.append(prediction_row)

            with predictions_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")

            if is_correct:
                print(f"Result: CORRECT | pred={normalized_prediction} gt={normalized_answer}")
            else:
                wrong_record = _build_wrong_record(
                    sample_index=sample_index,
                    row=row,
                    task=task,
                    response=response,
                    extracted_answer=extracted_answer,
                    normalized_prediction=normalized_prediction,
                    normalized_answer=normalized_answer,
                )
                wrong_rows.append(wrong_record)
                with wrong_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(wrong_record, ensure_ascii=False) + "\n")
                print(f"Result: WRONG   | pred={normalized_prediction} gt={normalized_answer}")

        prediction_df = pd.DataFrame(prediction_rows)
        metric = dataset.evaluate(prediction_df)
        evaluated_in_range = [row for row in prediction_rows if start <= int(row.get("dataset_index", -1)) < end]
        resumed_count = len(completed_indexes_in_range)
        newly_evaluated = len(evaluated_in_range) - resumed_count
        summary = {
            "dataset": dataset_name,
            "model_name": actual_model_name,
            "start": start,
            "limit": limit,
            "available_total": total_available,
            "range_end_exclusive": end,
            "evaluated_total_records": len(prediction_rows),
            "evaluated_in_requested_range": len(evaluated_in_range),
            "resumed_skipped": resumed_count,
            "newly_evaluated": newly_evaluated,
            "overwrite": overwrite,
            "available_total": total_available,
            "metrics": metric,
            "wrong_count": len(wrong_rows),
            "predictions_file": str(predictions_path),
            "wrong_questions_file": str(wrong_path),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        await model_client.close()
        await workbench.stop()


def _default_output_dir(dataset_name: str, model_name: str) -> Path:
    safe_dataset = _sanitize_name(dataset_name)
    safe_model = _sanitize_name(model_name)
    return Path(__file__).resolve().parent / "outputs" / f"autogen_eval_{safe_dataset}_{safe_model}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AutoGen agent on a life science dataset.")
    parser.add_argument(
        "--dataset",
        default="ProteinLMBench",
        help=f"Dataset name. Available: {', '.join(list_datasets())}",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional dataset cache directory.")
    parser.add_argument("--start", type=int, default=0, help="Start index within the dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to evaluate.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation outputs for the same dataset/model instead of resuming.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store predictions, wrong questions, and summary.",
    )
    args = parser.parse_args()

    try:
        from test_autogen import get_runtime_model_config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AutoGen dependencies are not installed in the current environment. "
            "Install the packages required by test_autogen.py before running this evaluator."
        ) from exc

    runtime_model_config = get_runtime_model_config()
    actual_model_name = runtime_model_config["model_name"]
    output_dir = args.output_dir or _default_output_dir(args.dataset, actual_model_name)
    summary = asyncio.run(
        evaluate_dataset(
            dataset_name=args.dataset,
            model_name=actual_model_name,
            cache_dir=args.cache_dir,
            start=args.start,
            limit=args.limit,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
    )

    print("\n" + "=" * 72)
    print("Evaluation complete")
    print("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
