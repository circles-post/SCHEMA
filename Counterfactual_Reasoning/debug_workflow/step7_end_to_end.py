"""Step 7: 端到端小规模测试 - 完整跑 2 条数据。

验证点:
- evaluate_dataset() 完整流程是否正常
- 输出文件 (predictions.jsonl, wrong_questions.jsonl, summary.json) 是否正确生成
- 输出内容是否可读、可解析
"""

import sys
sys.path.insert(0, "/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")

import asyncio
import json
from pathlib import Path

from evaluate_autogen_life_science import evaluate_dataset


async def main():
    output_dir = Path(__file__).resolve().parent / "test_output_e2e"

    print("=" * 60)
    print("Step 7: End-to-end test (2 samples)")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    # 清理旧的测试输出
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print("[INFO] Cleaned previous test output")

    summary = await evaluate_dataset(
        dataset_name="ProteinLMBench",
        cache_dir=None,
        start=0,
        limit=2,
        output_dir=output_dir,
    )

    # 检查输出
    print("\n" + "=" * 60)
    print("Step 7: Verify outputs")
    print("=" * 60)

    # 7a. summary
    print("\n--- 7a: Summary ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 7b. predictions.jsonl
    predictions_path = output_dir / "predictions.jsonl"
    print(f"\n--- 7b: predictions.jsonl (exists={predictions_path.exists()}) ---")
    if predictions_path.exists():
        lines = predictions_path.read_text(encoding="utf-8").strip().split("\n")
        print(f"  Lines: {len(lines)}")
        for i, line in enumerate(lines):
            data = json.loads(line)
            print(f"  Row {i}: correct={data.get('is_correct')}, "
                  f"pred={data.get('prediction_num')}, gt={data.get('answer_num')}")
    else:
        print("[WARN] predictions.jsonl not found!")

    # 7c. wrong_questions.jsonl
    wrong_path = output_dir / "wrong_questions.jsonl"
    print(f"\n--- 7c: wrong_questions.jsonl (exists={wrong_path.exists()}) ---")
    if wrong_path.exists():
        lines = wrong_path.read_text(encoding="utf-8").strip().split("\n")
        print(f"  Wrong answers: {len(lines)}")
    else:
        print("  No wrong answers file (all correct, or no wrong answers)")

    # 7d. summary.json
    summary_path = output_dir / "summary.json"
    print(f"\n--- 7d: summary.json (exists={summary_path.exists()}) ---")
    if summary_path.exists():
        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert saved_summary["evaluated"] == 2, f"Expected 2 evaluated, got {saved_summary['evaluated']}"
        print(f"  Evaluated: {saved_summary['evaluated']}")
        print(f"  Metrics: {saved_summary['metrics']}")
        print("[OK] summary.json is valid")
    else:
        print("[WARN] summary.json not found!")

    print("\n[PASS] Step 7 completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
