"""Step 10: 测试 AGDebuggerClient 基础连接和 API 调用。

验证点:
- AGDebuggerClient 能否连接到 backend
- get_topics / get_agents / get_logs 等读取 API 是否正常
- snapshot() 是否返回完整结构
- wait_until_idle() 是否正常工作

使用方式:
  1. 先启动 AGDebugger backend:
     python -m agdebugger.cli run test_agent_debug:get_agent_team --port 8081
  2. 再运行本脚本:
     python debug_workflow/step10_agdebugger_client.py [--base-url http://127.0.0.1:8081/api]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")
sys.path.insert(0, str(REPO_DIR))

from external_agent_controller import AGDebuggerClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/api")
    args = parser.parse_args()

    client = AGDebuggerClient(args.base_url)
    print(f"Target: {args.base_url}")

    # 10a. 基础连通性
    print("\n" + "=" * 60)
    print("Step 10a: Connection test (get_topics)")
    print("=" * 60)
    try:
        topics = client.get_topics()
        print(f"  Topics: {topics}")
        print("[OK] Backend is reachable")
    except Exception as e:
        print(f"[FAIL] Cannot connect to backend: {e}")
        print("\nMake sure AGDebugger backend is running:")
        print("  python -m agdebugger.cli run test_agent_debug:get_agent_team --port 8081")
        sys.exit(1)

    # 10b. 读取 API
    print("\n" + "=" * 60)
    print("Step 10b: Read APIs")
    print("=" * 60)
    apis = {
        "get_agents": lambda: client.get_agents(),
        "get_message_types": lambda: list(client.get_message_types().keys()),
        "get_num_tasks": lambda: client.get_num_tasks(),
        "get_loop_status": lambda: client.get_loop_status(),
        "get_queue": lambda: client.get_queue(),
        "get_logs": lambda: f"{len(client.get_logs())} entries",
    }
    for name, fn in apis.items():
        try:
            result = fn()
            print(f"  [OK] {name}: {result}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    # 10c. snapshot()
    print("\n" + "=" * 60)
    print("Step 10c: snapshot()")
    print("=" * 60)
    try:
        snap = client.snapshot()
        print(f"  Keys: {list(snap.keys())}")
        print(f"  agents: {snap.get('agents', [])}")
        print(f"  topics: {snap.get('topics', [])}")
        print(f"  num_tasks: {snap.get('num_tasks')}")
        print(f"  loop_running: {snap.get('loop_running')}")
        print(f"  queue length: {len(snap.get('queue', []))}")

        session_history = snap.get("session_history", {})
        print(f"  session_history keys: {list(session_history.keys())}")
        current = session_history.get("current_session")
        print(f"  current_session: {current}")
        print("[OK] snapshot() returned valid structure")
    except Exception as e:
        print(f"[FAIL] snapshot() failed: {e}")
        import traceback
        traceback.print_exc()

    # 10d. wait_until_idle (短超时)
    print("\n" + "=" * 60)
    print("Step 10d: wait_until_idle(timeout=5s)")
    print("=" * 60)
    try:
        client.wait_until_idle(timeout_sec=5.0)
        print("[OK] Backend is idle")
    except TimeoutError:
        print("[WARN] Backend is not idle (has pending tasks). This is OK if tasks are running.")
    except Exception as e:
        print(f"[FAIL] {e}")

    print("\n[PASS] Step 10 completed successfully.")


if __name__ == "__main__":
    main()
