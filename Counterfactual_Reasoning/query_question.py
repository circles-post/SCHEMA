#!/usr/bin/env python3
"""Search for a question in the bio-graph dataset and display full details.

Usage:
    # Search by keyword
    python query_question.py --component-id 1 --query "protein P7"

    # Search by line number in the JSONL
    python query_question.py --component-id 1 --line 3

    # Show all questions containing a keyword (list mode)
    python query_question.py --component-id 1 --query "RNA polymerase" --list
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/mnt/shared-storage-user/fengxinshun/AISci/datasets")))
from browse_bio_graph_cluster_examples import (
    load_component_nodes,
    load_examples_in_component,
    annotate_focus_nodes,
    normalize_text,
)


def print_full_example(ex: dict, idx: int, total: int) -> None:
    """Print all fields of one example."""
    print(f"\n{'=' * 70}")
    print(f"Example {idx + 1}/{total}  (JSONL line {ex['line_no']})")
    print("=" * 70)

    print(f"\nQ: {ex['question']}")

    focus_node = normalize_text(ex.get("focus_node", ""))
    if focus_node:
        print(f"Focus node: {focus_node}")

    print("\nOptions:")
    for name, text in ex["options"]:
        print(f"  - {name}: {text}")

    print(f"\nAnswer: {ex['answer']}")

    print(f"\nGraph nodes: {', '.join(ex['nodes'])}")

    print("\nGraph edges:")
    if ex["edges"]:
        print(f"  {'source':<30} {'relation':<30} {'target'}")
        print(f"  {'─' * 30} {'─' * 30} {'─' * 30}")
        for src, rel, dst in ex["edges"]:
            print(f"  {src:<30} {rel:<30} {dst}")
    else:
        print("  (none)")


def search_examples(examples: list, query: str) -> list:
    """Return (index, example) pairs where query appears in question or options."""
    query_lower = query.lower()
    results = []
    for i, ex in enumerate(examples):
        # Search in question
        if query_lower in str(ex["question"]).lower():
            results.append((i, ex))
            continue
        # Search in options
        for _, text in ex["options"]:
            if query_lower in text.lower():
                results.append((i, ex))
                break
        else:
            # Search in answer
            if query_lower in str(ex["answer"]).lower():
                results.append((i, ex))
                continue
            # Search in nodes
            for node in ex["nodes"]:
                if query_lower in node.lower():
                    results.append((i, ex))
                    break
    return results


def main() -> None:
    datasets_dir = Path("/mnt/shared-storage-user/fengxinshun/AISci/datasets")

    parser = argparse.ArgumentParser(description="Search questions in bio-graph dataset")
    parser.add_argument("--component-id", type=int, required=True, help="Component ID")
    parser.add_argument("--query", "-q", type=str, default=None, help="Keyword to search")
    parser.add_argument("--line", "-l", type=int, default=None, help="JSONL line number")
    parser.add_argument("--list", action="store_true", help="List mode: show brief summaries")
    parser.add_argument(
        "--input", type=Path, default=datasets_dir / "Protein_professional.jsonl",
        help="Path to JSONL examples.",
    )
    parser.add_argument(
        "--graph-dir", type=Path, default=datasets_dir / "bio_graph_output_professional",
        help="Directory containing components.csv.",
    )
    args = parser.parse_args()

    if args.query is None and args.line is None:
        parser.error("Provide --query or --line")

    components_csv = args.graph_dir / "components.csv"
    component_nodes = load_component_nodes(components_csv, args.component_id)
    examples = load_examples_in_component(args.input, component_nodes)
    annotate_focus_nodes(examples)
    print(f"Component {args.component_id}: {len(examples)} examples, {len(component_nodes)} nodes")

    # Search by line number
    if args.line is not None:
        found = [ex for ex in examples if ex["line_no"] == args.line]
        if not found:
            print(f"\nNo example found at line {args.line}")
            # Try to read the raw JSONL line directly
            print(f"Reading raw JSONL line {args.line}...")
            with open(args.input, "r") as f:
                for i, raw_line in enumerate(f, 1):
                    if i == args.line:
                        obj = json.loads(raw_line)
                        print(json.dumps(obj, indent=2, ensure_ascii=False)[:3000])
                        return
            print(f"Line {args.line} not found in file")
            return

        for ex in found:
            idx = examples.index(ex)
            print_full_example(ex, idx, len(examples))
        return

    # Search by keyword
    results = search_examples(examples, args.query)
    if not results:
        print(f'\nNo results found for: "{args.query}"')
        return

    print(f'\nFound {len(results)} match(es) for: "{args.query}"\n')

    if args.list:
        for idx, ex in results:
            answer = ex["answer"]
            print(f"  [{idx + 1}] (line {ex['line_no']}) {ex['question'][:100]}...")
            print(f"      Answer: {answer}  |  Focus: {normalize_text(ex.get('focus_node', ''))}")
        print(f"\nUse without --list to see full details, or --line N to query a specific line.")
    else:
        for idx, ex in results:
            print_full_example(ex, idx, len(examples))


if __name__ == "__main__":
    main()
