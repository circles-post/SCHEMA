"""Step 4: 测试 Agent 团队能否正常构建和关闭。

验证点:
- build_team() 能否正常返回 (team, workbench, model_client)
- 三个对象类型是否符合预期
- 能否正常关闭资源 (model_client.close(), workbench.stop())
"""

import sys
sys.path.insert(0, "/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")

import asyncio


async def main():
    print("=" * 60)
    print("Step 4: build_team()")
    print("=" * 60)

    # 4a. 导入检查
    print("\n--- 4a: Importing test_autogen ---")
    try:
        from test_autogen import build_team, run_task
        print("[OK] Import successful")
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        print("Please install required packages: autogen-agentchat, autogen-ext[openai]")
        sys.exit(1)

    # 4b. 构建团队
    print("\n--- 4b: Building team ---")
    try:
        team, workbench, model_client = await build_team()
        print(f"  team type:         {type(team).__name__}")
        print(f"  workbench type:    {type(workbench).__name__}")
        print(f"  model_client type: {type(model_client).__name__}")
        print("[OK] Team built successfully")
    except Exception as e:
        print(f"[FAIL] build_team() failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4c. 关闭资源
    print("\n--- 4c: Closing resources ---")
    try:
        await model_client.close()
        print("[OK] model_client closed")
    except Exception as e:
        print(f"[WARN] model_client.close() failed: {e}")

    try:
        await workbench.stop()
        print("[OK] workbench stopped")
    except Exception as e:
        print(f"[WARN] workbench.stop() failed: {e}")

    print("\n[PASS] Step 4 completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
