"""Step 14: 端到端测试 run_dataset_autodebug.py (1-2 条数据)。

验证点:
- 完整流程: 启动 backend -> 加载数据 -> 运行问题 -> (可选) LLM debug -> 输出结果
- 使用 --reuse-server 模式，假设 backend 已手动启动

使用方式:
  1. 启动 backend:
     python -m agdebugger.cli run test_agent_debug:get_agent_team --port 8081
  2. 运行本脚本:
     python debug_workflow/step14_end_to_end_autodebug.py --component-id <ID> [--enable-llm-debug]
"""

import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--component-id", type=int, required=True)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--limit", type=int, default=2, help="Number of examples to test")
    parser.add_argument("--enable-llm-debug", action="store_true")
    parser.add_argument("--run-timeout", type=float, default=120.0)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Step 14: End-to-end autodebug test ({args.limit} examples)")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(REPO_DIR / "run_dataset_autodebug.py"),
        "--component-id", str(args.component_id),
        "--reuse-server",
        "--port", str(args.port),
        "--start", "0",
        "--limit", str(args.limit),
        "--run-timeout", str(args.run_timeout),
    ]
    if args.enable_llm_debug:
        cmd.append("--enable-llm-debug")

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_DIR),
            text=True,
            timeout=args.run_timeout * args.limit + 60,  # 给些余量
        )
        print(f"\nExit code: {result.returncode}")
        if result.returncode == 0:
            print("[PASS] Step 14 completed successfully.")
        else:
            print(f"[FAIL] Process exited with code {result.returncode}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"[FAIL] Timed out after {args.run_timeout * args.limit + 60}s")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
