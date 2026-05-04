"""Step 8: 测试 run_dataset_autodebug.py 的数据集导入链路。

验证点:
- DATASETS_DIR 路径是否存在
- browse_bio_graph_cluster_examples 模块是否可导入
- load_component_nodes / load_examples_in_component / annotate_focus_nodes 函数是否可用
- 加载一个 component 的数据并检查格式
"""

import sys
from pathlib import Path

REPO_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")
DATASETS_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/datasets")
sys.path.insert(0, str(REPO_DIR))
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))


def main():
    # 8a. 检查 DATASETS_DIR 是否存在
    print("=" * 60)
    print("Step 8a: Check DATASETS_DIR")
    print("=" * 60)
    print(f"  Path: {DATASETS_DIR}")
    print(f"  Exists: {DATASETS_DIR.exists()}")
    if not DATASETS_DIR.exists():
        print("[FAIL] DATASETS_DIR does not exist. Cannot continue.")
        sys.exit(1)
    print("[OK]")

    # 8b. 检查关键文件是否存在
    print("\n" + "=" * 60)
    print("Step 8b: Check required files")
    print("=" * 60)
    required_files = {
        "browse_bio_graph_cluster_examples.py": DATASETS_DIR / "browse_bio_graph_cluster_examples.py",
        "Protein_professional.jsonl": DATASETS_DIR / "Protein_professional.jsonl",
        "bio_graph_output_professional/components.csv": DATASETS_DIR / "bio_graph_output_professional" / "components.csv",
    }
    all_found = True
    for name, path in required_files.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name}: {path}")
        if not exists:
            all_found = False
    if not all_found:
        print("[WARN] Some files are missing. Subsequent steps may fail.")

    # 8c. 导入 browse_bio_graph_cluster_examples
    print("\n" + "=" * 60)
    print("Step 8c: Import browse_bio_graph_cluster_examples")
    print("=" * 60)
    try:
        from browse_bio_graph_cluster_examples import (
            annotate_focus_nodes,
            load_component_nodes,
            load_examples_in_component,
            normalize_text,
        )
        print(f"  load_component_nodes:      {load_component_nodes}")
        print(f"  load_examples_in_component: {load_examples_in_component}")
        print(f"  annotate_focus_nodes:       {annotate_focus_nodes}")
        print(f"  normalize_text:             {normalize_text}")
        print("[OK] All functions imported successfully")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        sys.exit(1)

    # 8d. 加载一个 component 的数据
    print("\n" + "=" * 60)
    print("Step 8d: Load component data (component_id=0)")
    print("=" * 60)
    components_csv = DATASETS_DIR / "bio_graph_output_professional" / "components.csv"
    input_path = DATASETS_DIR / "Protein_professional.jsonl"

    if not components_csv.exists() or not input_path.exists():
        print("[SKIP] Required files not found, skipping data load test")
        return

    try:
        component_nodes = load_component_nodes(components_csv, 0)
        print(f"  Component 0 nodes: {len(component_nodes)}")
        if len(component_nodes) == 0:
            print("  [WARN] Component 0 has no nodes. Try a different component_id.")
            # Try finding a valid component
            import pandas as pd
            comp_df = pd.read_csv(components_csv)
            if "component" in comp_df.columns:
                valid_ids = sorted(comp_df["component"].unique())[:5]
                print(f"  Available component IDs (first 5): {valid_ids}")
                if valid_ids:
                    component_nodes = load_component_nodes(components_csv, valid_ids[0])
                    print(f"  Component {valid_ids[0]} nodes: {len(component_nodes)}")

        if len(component_nodes) > 0:
            examples = load_examples_in_component(input_path, component_nodes)
            print(f"  Examples loaded: {len(examples)}")
            annotate_focus_nodes(examples)

            if examples:
                ex = examples[0]
                print(f"\n  First example keys: {list(ex.keys())}")
                print(f"  question: {str(ex.get('question', ''))[:120]}")
                print(f"  answer:   {ex.get('answer', '')}")
                options = ex.get("options", [])
                print(f"  options ({len(options)}):")
                for name, text in options[:3]:
                    print(f"    - {name}: {str(text)[:80]}")
                if len(options) > 3:
                    print(f"    ... and {len(options) - 3} more")
                print("[OK] Data loaded and formatted correctly")
            else:
                print("[WARN] No examples found in this component")
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[PASS] Step 8 completed successfully.")


if __name__ == "__main__":
    main()
