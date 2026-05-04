"""Step 3: 测试答案提取与归一化逻辑。

验证点:
- _extract_answer_tag() 能否从各种格式中提取答案
- _normalize_prediction() 能否正确归一化为数字
- 边界情况处理是否正确
"""

import sys
sys.path.insert(0, "/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")

from evaluate_autogen_life_science import _extract_answer_tag, _normalize_prediction


def test_extract_answer_tag():
    print("=" * 60)
    print("Step 3a: _extract_answer_tag()")
    print("=" * 60)

    cases = [
        ("blah <answer>3</answer> blah", "3"),
        ("<answer> 2 </answer>", "2"),
        ("no tag here", None),
        ("<ANSWER>5</ANSWER>", "5"),
        ("<answer>\n 4 \n</answer>", "4"),
        ("prefix <answer>1</answer> suffix <answer>2</answer>", "1"),  # 取第一个
        ("", None),
    ]

    all_pass = True
    for text, expected in cases:
        result = _extract_answer_tag(text)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] input={text[:50]!r} => expected={expected!r}, got={result!r}")

    return all_pass


def test_normalize_prediction():
    print("\n" + "=" * 60)
    print("Step 3b: _normalize_prediction()")
    print("=" * 60)

    cases = [
        ("3", "3"),
        ("A", "1"),
        ("B", "2"),
        ("C", "3"),
        ("Option B", "2"),
        ("OPTION C", "3"),
        (None, "7"),
        ("", "7"),
        ("The answer is 4", "4"),
        ("  5  ", "5"),
        ("xyz", "7"),        # 无法识别 -> 默认7
        ("12", "12"),        # 多位数字
    ]

    all_pass = True
    for raw, expected in cases:
        result = _normalize_prediction(raw)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] input={raw!r} => expected={expected!r}, got={result!r}")

    return all_pass


def main():
    p1 = test_extract_answer_tag()
    p2 = test_normalize_prediction()

    print()
    if p1 and p2:
        print("[PASS] Step 3 completed successfully.")
    else:
        print("[FAIL] Some tests failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
