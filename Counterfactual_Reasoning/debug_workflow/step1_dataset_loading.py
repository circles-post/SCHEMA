"""Step 1: 测试数据集加载是否正常。

验证点:
- list_datasets() 能否正常列出所有数据集
- get_dataset() 能否正常获取 ProteinLMBench
- load() 能否正常加载数据
- 数据列名和内容是否符合预期
"""

import sys
sys.path.insert(0, "/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")

from dataset.life_science_datasets import get_dataset, list_datasets


def main():
    # 1a. 列出所有可用数据集
    print("=" * 60)
    print("Step 1a: list_datasets()")
    print("=" * 60)
    datasets = list_datasets()
    print(f"Available datasets ({len(datasets)}):")
    for name in datasets:
        print(f"  - {name}")

    # 1b. 获取 ProteinLMBench 数据集对象
    print("\n" + "=" * 60)
    print("Step 1b: get_dataset('ProteinLMBench')")
    print("=" * 60)
    ds = get_dataset("ProteinLMBench")
    print(f"Dataset object type: {type(ds)}")
    print(f"Has build_prompt: {hasattr(ds, 'build_prompt')}")
    print(f"Has evaluate: {hasattr(ds, 'evaluate')}")

    # 1c. 加载数据
    print("\n" + "=" * 60)
    print("Step 1c: load()")
    print("=" * 60)
    df = ds.load()
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # 1d. 查看第一行数据
    print("\n" + "=" * 60)
    print("Step 1d: Sample row (index 0)")
    print("=" * 60)
    row = dict(df.iloc[0])
    for k, v in row.items():
        val_str = str(v)
        if len(val_str) > 120:
            val_str = val_str[:120] + "..."
        print(f"  {k}: {val_str}")

    print("\n[PASS] Step 1 completed successfully.")


if __name__ == "__main__":
    main()
