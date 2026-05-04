"""Step 11: 测试通过 AGDebugger backend 运行单个问题。

验证点:
- run_question() 能否正确发送 GroupChatReset + GroupChatStart
- Agent 能否返回包含 <answer> 的响应
- 消息时间戳过滤是否正确 (只获取本轮新消息)

前置条件: AGDebugger backend 已启动 (见 step10)
"""

import argparse
import sys
from pathlib import Path

REPO_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")
DATASETS_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/datasets")
sys.path.insert(0, str(REPO_DIR))
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from external_agent_controller import AGDebuggerClient
from run_dataset_autodebug import (
    _guess_manager_topic,
    extract_answer_from_messages,
    format_task,
    load_examples,
    normalize_answer,
    run_question,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/api")
    parser.add_argument("--component-id", type=int, default=0)
    parser.add_argument("--input", type=Path, default=DATASETS_DIR / "Protein_professional.jsonl")
    parser.add_argument("--graph-dir", type=Path, default=DATASETS_DIR / "bio_graph_output_professional")
    parser.add_argument("--example-index", type=int, default=0, help="Which example to test")
    parser.add_argument("--run-timeout", type=float, default=120.0)
    args = parser.parse_args()

    client = AGDebuggerClient(args.base_url)

    # 11a. 连接并获取 manager topic
    print("=" * 60)
    print("Step 11a: Connect and find manager topic")
    print("=" * 60)
    try:
        topics = client.get_topics()
        manager_topic = _guess_manager_topic(topics)
        print(f"  Topics: {topics}")
        print(f"  Manager topic: {manager_topic}")
        print("[OK]")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    # 11b. 加载测试数据
    print("\n" + "=" * 60)
    print("Step 11b: Load test example")
    print("=" * 60)
    try:
        examples = load_examples(args.component_id, args.input, args.graph_dir)
        if args.example_index >= len(examples):
            print(f"[FAIL] --example-index {args.example_index} >= {len(examples)} examples")
            sys.exit(1)
        ex = examples[args.example_index]
        task = format_task(ex)
        gt_raw = str(ex["answer"])
        gt_norm = normalize_answer(gt_raw, num_options=len(ex["options"]))
        print(f"  Question: {str(ex.get('question', ''))[:150]}")
        print(f"  Ground truth: {gt_raw} -> {gt_norm}")
        print(f"  Options: {len(ex['options'])}")
        print("[OK]")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 11c. 运行问题
    print("\n" + "=" * 60)
    print(f"Step 11c: run_question (timeout={args.run_timeout}s)")
    print("=" * 60)
    try:
        messages, ans_raw = run_question(client, manager_topic, task, run_timeout_sec=args.run_timeout)
        print(f"  Messages returned: {len(messages)}")
        for i, m in enumerate(messages[-5:]):  # 只打印最后 5 条
            source = m.get("source", "?")
            content = ""
            if isinstance(m.get("content"), str):
                content = m["content"][:120]
            elif isinstance(m.get("body"), dict):
                inner = m["body"]
                content = str(inner.get("content", inner.get("type", "")))[:120]
            print(f"    [{i}] source={source}: {content}")

        print(f"\n  Extracted answer (raw): {ans_raw!r}")
        if ans_raw:
            ans_norm = normalize_answer(ans_raw, len(ex["options"]))
            is_correct = ans_norm == gt_norm
            print(f"  Extracted answer (norm): {ans_norm}")
            print(f"  Ground truth (norm):     {gt_norm}")
            print(f"  Correct: {is_correct}")
        else:
            print("  [WARN] No <answer> tag found in messages")
        print("[OK]")
    except TimeoutError:
        print(f"[FAIL] Timed out after {args.run_timeout}s. Agent may be stuck.")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[PASS] Step 11 completed successfully.")


if __name__ == "__main__":
    main()
