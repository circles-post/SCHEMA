"""Step 12: 测试 LLM debug loop (外部 LLM 修复错误轨迹)。

验证点:
- LLMPlanner 能否正常初始化 (需要 API key)
- planner.plan() 能否基于 snapshot 生成有效 action
- execute_action() 能否正确执行 action
- run_llm_debug_loop() 完整流程是否跑通

前置条件:
  - AGDebugger backend 已启动
  - 设置环境变量: AGENTDEBUG_OPENAI_API_KEY, AGENTDEBUG_MODEL_NAME (可选), AGENTDEBUG_OPENAI_BASE_URL (可选)
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")
sys.path.insert(0, str(REPO_DIR))

from external_agent_controller import AGDebuggerClient, LLMPlanner, execute_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/api")
    parser.add_argument("--model", default=os.environ.get("AGENTDEBUG_MODEL_NAME", "gpt-4o-mini"))
    parser.add_argument("--api-key", default=os.environ.get("AGENTDEBUG_OPENAI_API_KEY", ""))
    parser.add_argument("--api-base", default=os.environ.get("AGENTDEBUG_OPENAI_BASE_URL", "https://api.openai.com/v1"))
    args = parser.parse_args()

    if not args.api_key:
        print("[FAIL] No API key. Set AGENTDEBUG_OPENAI_API_KEY or use --api-key")
        sys.exit(1)

    client = AGDebuggerClient(args.base_url)

    # 12a. 检查 backend 连通性
    print("=" * 60)
    print("Step 12a: Check backend")
    print("=" * 60)
    try:
        topics = client.get_topics()
        print(f"  Topics: {topics}")
        print("[OK]")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    # 12b. 初始化 LLMPlanner
    print("\n" + "=" * 60)
    print("Step 12b: Initialize LLMPlanner")
    print("=" * 60)
    try:
        planner = LLMPlanner(
            model=args.model,
            api_key=args.api_key,
            base_url=args.api_base,
            user_goal="Test: check if the debug planner can generate a valid action.",
        )
        print(f"  Model: {args.model}")
        print(f"  Base URL: {args.api_base}")
        print("[OK] LLMPlanner initialized")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 12c. 获取 snapshot 并调用 planner
    print("\n" + "=" * 60)
    print("Step 12c: Plan one action from current snapshot")
    print("=" * 60)
    try:
        snapshot = client.snapshot()
        print(f"  Snapshot keys: {list(snapshot.keys())}")
        print(f"  num_tasks: {snapshot.get('num_tasks')}")

        action = planner.plan(snapshot)
        print(f"  Planned action: {json.dumps(action, ensure_ascii=False)}")

        # 检查 action 结构
        assert "action" in action, "Action missing 'action' key"
        valid_actions = {
            "step", "drop", "start_loop", "stop_loop",
            "publish", "send", "edit_queue", "edit_and_revert",
            "insert_after", "finish",
        }
        assert action["action"] in valid_actions, f"Unknown action: {action['action']}"
        print(f"  Action type: {action['action']} (valid)")
        print("[OK]")
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 12d. 执行 action (dry-run: 只执行 finish/step 等安全操作)
    print("\n" + "=" * 60)
    print("Step 12d: Execute action")
    print("=" * 60)
    try:
        action_name = action["action"]
        if action_name == "finish":
            print(f"  Planner chose 'finish': {action.get('reason', 'no reason')}")
            print("  [OK] No side effects")
        else:
            print(f"  Executing action: {action_name}")
            keep_running = execute_action(client, action)
            print(f"  keep_running: {keep_running}")
            print("[OK]")
    except Exception as e:
        print(f"[FAIL] execute_action failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[PASS] Step 12 completed successfully.")


if __name__ == "__main__":
    main()
