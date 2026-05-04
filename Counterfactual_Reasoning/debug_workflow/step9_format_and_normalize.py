"""Step 9: 测试 run_dataset_autodebug.py 的 format_task / normalize_answer / extract_answer。

验证点:
- format_task() 能否正确格式化带选项的题目
- normalize_answer() 各种输入格式的归一化结果
- extract_answer_from_messages() 能否从消息列表中提取答案
"""

import sys
from pathlib import Path

REPO_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")
sys.path.insert(0, str(REPO_DIR))

from run_dataset_autodebug import (
    extract_answer_from_messages,
    format_task,
    normalize_answer,
)


def test_format_task():
    print("=" * 60)
    print("Step 9a: format_task()")
    print("=" * 60)

    example = {
        "question": "What is the primary function of hemoglobin?",
        "focus_node": "hemoglobin",
        "options": [
            ("option1", "Oxygen transport"),
            ("option2", "DNA replication"),
            ("option3", "Protein synthesis"),
        ],
        "answer": "option1",
    }
    task = format_task(example)
    print(f"  Output:\n{task}")
    assert "Q:" in task, "Missing Q: prefix"
    assert "Focus node:" in task, "Missing Focus node"
    assert "Options:" in task, "Missing Options section"
    assert "option1" in task, "Missing option1"
    print("[OK]")

    # 无 focus_node 的情况
    example_no_focus = {
        "question": "Test question?",
        "options": [("option1", "A"), ("option2", "B")],
    }
    task2 = format_task(example_no_focus)
    assert "Focus node:" not in task2, "Should not have Focus node when empty"
    print("[OK] No focus_node case handled")


def test_normalize_answer():
    print("\n" + "=" * 60)
    print("Step 9b: normalize_answer()")
    print("=" * 60)

    cases = [
        # (input, num_options, expected)
        ("option1", 6, "option1"),
        ("Option 1", 6, "option1"),
        ("option_2", 6, "option2"),
        ("OPTION 3", 6, "option3"),
        ("A", 6, "option1"),
        ("B", 6, "option2"),
        ("C", 6, "option3"),
        ("1", 6, "option1"),
        ("3", 6, "option3"),
        ("Option A", 6, "option1"),
        ("Option B", 6, "option2"),
        # 边界：字母超出选项范围
        ("Z", 3, "optionz"),  # Z=26, >3, fallback to collapsed
    ]

    all_pass = True
    for raw, num_opts, expected in cases:
        result = normalize_answer(raw, num_opts)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] normalize_answer({raw!r}, {num_opts}) => expected={expected!r}, got={result!r}")

    return all_pass


def test_extract_answer_from_messages():
    print("\n" + "=" * 60)
    print("Step 9c: extract_answer_from_messages()")
    print("=" * 60)

    # 简单消息
    msgs1 = [
        {"content": "Let me think about this..."},
        {"content": "The answer is <answer>option2</answer>"},
    ]
    result = extract_answer_from_messages(msgs1)
    print(f"  Simple case: {result!r}")
    assert result == "option2", f"Expected 'option2', got {result!r}"
    print("  [OK]")

    # 嵌套结构
    msgs2 = [
        {"inner": {"content": "blah <answer>3</answer>"}},
    ]
    result2 = extract_answer_from_messages(msgs2)
    print(f"  Nested case: {result2!r}")
    assert result2 == "3", f"Expected '3', got {result2!r}"
    print("  [OK]")

    # 多个 answer tag -> 取最后一个
    msgs3 = [
        {"content": "<answer>1</answer>"},
        {"content": "<answer>2</answer>"},
    ]
    result3 = extract_answer_from_messages(msgs3)
    print(f"  Multi-tag case (expect last): {result3!r}")
    assert result3 == "2", f"Expected '2', got {result3!r}"
    print("  [OK]")

    # 无 answer tag
    msgs4 = [{"content": "I have no idea"}]
    result4 = extract_answer_from_messages(msgs4)
    print(f"  No tag case: {result4!r}")
    assert result4 is None, f"Expected None, got {result4!r}"
    print("  [OK]")

    return True


def main():
    test_format_task()
    p1 = test_normalize_answer()
    p2 = test_extract_answer_from_messages()

    print()
    if p1 and p2:
        print("[PASS] Step 9 completed successfully.")
    else:
        print("[FAIL] Some tests failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
