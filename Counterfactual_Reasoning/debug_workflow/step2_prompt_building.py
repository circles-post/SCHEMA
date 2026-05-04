"""Step 2: 测试 Prompt 构建是否正常。

验证点:
- dataset.build_prompt() 能否正确格式化题目
- _format_task() 追加的 answer hint 是否正确
"""

import sys
sys.path.insert(0, "/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")

from dataset.life_science_datasets import get_dataset
from evaluate_autogen_life_science import _format_task


def main():
    ds = get_dataset("ProteinLMBench")
    df = ds.load()

    # 测试前 3 条数据的 prompt
    for i in range(min(3, len(df))):
        row = dict(df.iloc[i])
        print(f"{'=' * 60}")
        print(f"Sample {i}")
        print(f"{'=' * 60}")

        # 2a. 原始 build_prompt
        raw_prompt = ds.build_prompt(row)
        print(f"\n--- build_prompt (length={len(raw_prompt)}) ---")
        print(raw_prompt[:400])
        if len(raw_prompt) > 400:
            print("... (truncated)")

        # 2b. _format_task (带 answer hint)
        task = _format_task(ds, row)
        print(f"\n--- _format_task (length={len(task)}) ---")
        # 只显示追加的部分
        extra = task[len(raw_prompt.rstrip()):]
        print(f"Appended hint:\n{extra}")

        # 2c. 检查 answer hint 关键标记
        assert "<answer>" in task, "Missing <answer> tag in hint"
        assert "TERMINATE" in task, "Missing TERMINATE in hint"
        print("\n[OK] hint contains <answer> and TERMINATE tags")

    print(f"\n[PASS] Step 2 completed successfully.")


if __name__ == "__main__":
    main()
