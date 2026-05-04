"""Step 6: 测试评估指标计算是否正常。

验证点:
- dataset.evaluate() 能否正常接收 DataFrame 并返回指标
- 全对/全错/部分对的场景是否输出合理
- JSON 序列化是否有 numpy 类型问题
"""

import sys
sys.path.insert(0, "/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")

import json
import pandas as pd
from dataset.life_science_datasets import get_dataset


def main():
    ds = get_dataset("ProteinLMBench")
    df = ds.load()
    sample = df.head(5).copy()

    # 6a. 全部正确的场景
    print("=" * 60)
    print("Step 6a: All correct")
    print("=" * 60)
    df_correct = sample.copy()
    df_correct["prediction"] = df_correct["answer"]
    metric = ds.evaluate(df_correct)
    print(f"  Metric: {metric}")
    assert metric["accuracy"] == 1.0, f"Expected accuracy=1.0, got {metric['accuracy']}"
    print("[OK] accuracy = 1.0")

    # 6b. 全部错误的场景
    print("\n" + "=" * 60)
    print("Step 6b: All wrong")
    print("=" * 60)
    df_wrong = sample.copy()
    df_wrong["prediction"] = "7"  # 不可能的答案
    metric2 = ds.evaluate(df_wrong)
    print(f"  Metric: {metric2}")
    assert metric2["accuracy"] == 0.0, f"Expected accuracy=0.0, got {metric2['accuracy']}"
    print("[OK] accuracy = 0.0")

    # 6c. 部分正确的场景
    print("\n" + "=" * 60)
    print("Step 6c: Partial correct")
    print("=" * 60)
    df_partial = sample.copy()
    predictions = list(df_partial["answer"])
    # 让前2个正确，后3个错误
    for i in range(2, len(predictions)):
        predictions[i] = "7"
    df_partial["prediction"] = predictions
    metric3 = ds.evaluate(df_partial)
    print(f"  Metric: {metric3}")
    expected_acc = 2 / len(df_partial)
    assert abs(metric3["accuracy"] - expected_acc) < 1e-9, (
        f"Expected accuracy={expected_acc}, got {metric3['accuracy']}"
    )
    print(f"[OK] accuracy = {expected_acc}")

    # 6d. 测试 JSON 序列化 (检查 numpy 类型问题)
    print("\n" + "=" * 60)
    print("Step 6d: JSON serialization check")
    print("=" * 60)
    row = dict(sample.iloc[0])
    row["prediction"] = "3"
    row["is_correct"] = 1
    try:
        json_str = json.dumps(row, ensure_ascii=False)
        print(f"  JSON output (first 200 chars): {json_str[:200]}")
        print("[OK] json.dumps succeeded")
    except TypeError as e:
        print(f"[FAIL] json.dumps failed: {e}")
        print("  This is the numpy serialization issue!")
        print("  Suggestion: add a custom NumpyEncoder to evaluate_autogen_life_science.py")
        # 演示修复方法
        import numpy as np

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if pd.isna(obj):
                    return None
                return super().default(obj)

        json_str = json.dumps(row, ensure_ascii=False, cls=NumpyEncoder)
        print(f"  With NumpyEncoder: {json_str[:200]}")
        print("[OK] NumpyEncoder workaround works")

    print("\n[PASS] Step 6 completed successfully.")


if __name__ == "__main__":
    main()
