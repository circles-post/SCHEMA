"""Auto-evaluate the agent on bio-graph MCQ examples.

Loads questions from the same data source as browse_bio_graph_cluster_examples.py,
feeds them to the ToolUniverse + WebSearch agent, checks the answer, and stops
on the first wrong answer.

Usage:
    # Test on component 1 (press Enter between questions)
    python test_auto_eval.py --component-id 1

    # Start from a specific example index
    python test_auto_eval.py --component-id 1 --start 5

    # Limit how many examples to test
    python test_auto_eval.py --component-id 1 --limit 20
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# Reuse data loading from the browse script
sys.path.insert(0, str(Path("/mnt/shared-storage-user/fengxinshun/AISci/datasets")))
from browse_bio_graph_cluster_examples import (
    load_component_nodes,
    load_examples_in_component,
    annotate_focus_nodes,
    normalize_text,
)

from test_autogen import build_team, run_task


# ---------------------------------------------------------------------------
# Format example as agent task
# ---------------------------------------------------------------------------
def format_task(example: dict) -> str:
    """Format an example dict into the task string the agent expects."""
    lines = []
    lines.append(f"Q: {example['question']}")

    focus_node = normalize_text(example.get("focus_node", ""))
    if focus_node:
        lines.append(f"Focus node: {focus_node}")

    lines.append("")
    lines.append("Options:")
    for name, text in example["options"]:
        lines.append(f"  - {name}: {text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extract answer from agent response
# ---------------------------------------------------------------------------
def extract_answer(response: str) -> str | None:
    """Extract the answer from <answer>...</answer> tags in the response."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def normalize_answer(raw: str, num_options: int = 6) -> str:
    """Normalize an answer string to a comparable format like 'option4'.

    Handles multiple formats:
      - "option4", "option 4", "option_4" → "option4"
      - "4" → "option4"
      - "D", "d" → "option4"  (A=1, B=2, C=3, D=4, ...)
      - "Option D" → "option4"
    """
    raw = raw.strip()

    # Map A-Z letters to option numbers
    _LETTER_TO_NUM = {chr(ord("A") + i): i + 1 for i in range(26)}

    # Remove common prefixes like "Option ", "option:", etc.
    cleaned = re.sub(r"^option[\s_:]*", "", raw, flags=re.IGNORECASE).strip()

    # Case 1: single letter like "D" or "d"
    if len(cleaned) == 1 and cleaned.upper() in _LETTER_TO_NUM:
        n = _LETTER_TO_NUM[cleaned.upper()]
        if 1 <= n <= num_options:
            return f"option{n}"

    # Case 2: pure digit like "4"
    if cleaned.isdigit():
        return f"option{cleaned}"

    # Case 3: already "option4" style after stripping spaces/underscores
    collapsed = raw.lower().replace(" ", "").replace("_", "")
    m = re.match(r"^option(\d+)$", collapsed)
    if m:
        return f"option{m.group(1)}"

    # Case 4: letter after "option" prefix, e.g. "option D"
    m = re.match(r"^option\s*([a-zA-Z])$", raw, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        if letter in _LETTER_TO_NUM:
            n = _LETTER_TO_NUM[letter]
            if 1 <= n <= num_options:
                return f"option{n}"

    # Fallback: return lowercased collapsed form
    return collapsed


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
async def evaluate(
    component_id: int,
    input_path: Path,
    graph_dir: Path,
    start: int,
    limit: int | None,
) -> None:
    # Load examples
    components_csv = graph_dir / "components.csv"
    if not components_csv.exists():
        print(f"ERROR: components.csv not found: {components_csv}")
        sys.exit(1)

    component_nodes = load_component_nodes(components_csv, component_id)
    examples = load_examples_in_component(input_path, component_nodes)
    annotate_focus_nodes(examples)
    total = len(examples)
    print(f"Component {component_id}: {total} examples loaded, {len(component_nodes)} nodes")

    if start >= total:
        print(f"ERROR: --start {start} >= total examples {total}")
        sys.exit(1)

    end = total if limit is None else min(start + limit, total)
    examples_to_test = examples[start:end]
    print(f"Testing examples {start}..{end - 1} ({len(examples_to_test)} questions)\n")

    # Build agent team
    print("Starting ToolUniverse MCP server (this may take 1-2 minutes)...", flush=True)
    team, workbench, model_client = await build_team()
    print("Agent team ready.\n")

    correct = 0
    wrong = 0

    try:
        for i, example in enumerate(examples_to_test):
            global_idx = start + i
            gt_answer = normalize_answer(str(example["answer"]), num_options=len(example["options"]))

            print(f"\n{'#' * 60}")
            print(f"# Example {global_idx + 1}/{total}  (line {example['line_no']})")
            print(f"# Ground truth: {example['answer']}")
            print(f"{'#' * 60}")

            task = format_task(example)

            # Reset team for each question
            await team.reset()
            response = await run_task(team, task)

            # Extract and check answer
            agent_answer_raw = extract_answer(response)
            if agent_answer_raw is None:
                print("\n[WARNING] No <answer> tag found in response!")
                agent_answer = ""
            else:
                agent_answer = normalize_answer(agent_answer_raw, num_options=len(example["options"]))

            is_correct = agent_answer == gt_answer

            print(f"\n{'─' * 60}")
            print(f"  Agent answer:  {agent_answer_raw or '(not found)'}")
            print(f"  Ground truth:  {example['answer']}")
            print(f"  Normalized:    agent={agent_answer}  gt={gt_answer}")
            if is_correct:
                correct += 1
                print(f"  Result:        ✓ CORRECT  ({correct}/{correct + wrong} so far)")
            else:
                wrong += 1
                print(f"  Result:        ✗ WRONG  ({correct}/{correct + wrong} so far)")
                print(f"\n{'!' * 60}")
                print(f"! STOPPED: Agent answered incorrectly on example {global_idx + 1}")
                print(f"! Score: {correct}/{correct + wrong} ({correct / (correct + wrong) * 100:.1f}%)")
                print(f"{'!' * 60}")
                break

            # Wait for user to press Enter before next question
            remaining = len(examples_to_test) - (i + 1)
            if remaining > 0:
                try:
                    input(f"\n  Press Enter for next question ({remaining} remaining)... ")
                except (EOFError, KeyboardInterrupt):
                    print("\nStopped by user.")
                    break
        else:
            # All examples passed
            print(f"\n{'=' * 60}")
            print(f"  ALL {correct} EXAMPLES CORRECT!")
            print(f"  Score: {correct}/{correct} (100%)")
            print(f"{'=' * 60}")

    finally:
        print(f"\nFinal score: {correct}/{correct + wrong}", end="")
        if correct + wrong > 0:
            print(f" ({correct / (correct + wrong) * 100:.1f}%)")
        else:
            print()
        await model_client.close()
        await workbench.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    datasets_dir = Path("/mnt/shared-storage-user/fengxinshun/AISci/datasets")

    parser = argparse.ArgumentParser(description="Auto-evaluate agent on bio-graph MCQ examples")
    parser.add_argument(
        "--component-id", type=int, required=True,
        help="Component ID to test.",
    )
    parser.add_argument(
        "--input", type=Path, default=datasets_dir / "Protein_professional.jsonl",
        help="Path to JSONL examples.",
    )
    parser.add_argument(
        "--graph-dir", type=Path, default=datasets_dir / "bio_graph_output_professional",
        help="Directory containing components.csv.",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start from this example index (0-based). Default: 0.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of examples to test. Default: all.",
    )
    args = parser.parse_args()

    asyncio.run(evaluate(
        component_id=args.component_id,
        input_path=args.input,
        graph_dir=args.graph_dir,
        start=args.start,
        limit=args.limit,
    ))


if __name__ == "__main__":
    main()
