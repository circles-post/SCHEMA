"""Step 5: 单条推理测试 - 用 Agent 回答一道题并检查输出。

验证点:
- Agent 能否正常接收 task 并返回响应
- 响应中是否包含 <answer> 标签
- 答案提取和归一化流程是否跑通
- team.reset() 是否正常工作
"""

import sys
sys.path.insert(0, "/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")

import asyncio
from dataset.life_science_datasets import get_dataset
from evaluate_autogen_life_science import (
    _extract_answer_tag,
    _format_task,
    _normalize_prediction,
)


async def main():
    from test_autogen import build_team, run_task

    ds = get_dataset("ProteinLMBench")
    df = ds.load()
    row = dict(df.iloc[0])

    # 5a. 构建 task
    print("=" * 60)
    print("Step 5a: Build task for row 0")
    print("=" * 60)
    task = _format_task(ds, row)
    print(f"Task (first 300 chars):\n{task[:300]}...")

    # 5b. 构建 Agent 并推理
    print("\n" + "=" * 60)
    print("Step 5b: Run inference")
    print("=" * 60)
    team, workbench, model_client = await build_team()

    try:
        response = await run_task(team, task)
        print(f"\nResponse (first 500 chars):\n{response[:500]}")
        if len(response) > 500:
            print("... (truncated)")

        # 5c. 提取并归一化答案
        print("\n" + "=" * 60)
        print("Step 5c: Extract and normalize answer")
        print("=" * 60)
        extracted = _extract_answer_tag(response)
        normalized_pred = _normalize_prediction(extracted or response)
        normalized_gt = _normalize_prediction(str(row.get("answer", "")))

        print(f"  Raw extracted:         {extracted!r}")
        print(f"  Normalized prediction: {normalized_pred!r}")
        print(f"  Ground truth (raw):    {row.get('answer', '')!r}")
        print(f"  Ground truth (norm):   {normalized_gt!r}")
        print(f"  Correct:               {normalized_pred == normalized_gt}")

        if extracted is None:
            print("\n[WARN] Agent did NOT output <answer>...</answer> tag.")
            print("  This means the answer was extracted from raw response text.")
            print("  Check if the system prompt is being followed.")

        # 5d. 测试 team.reset()
        print("\n" + "=" * 60)
        print("Step 5d: Test team.reset()")
        print("=" * 60)
        await team.reset()
        print("[OK] team.reset() succeeded")

        print("\n[PASS] Step 5 completed successfully.")
    finally:
        await model_client.close()
        await workbench.stop()


if __name__ == "__main__":
    asyncio.run(main())
