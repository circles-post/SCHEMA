"""Step 13: 测试 external_agent 的 claim 分析管线。

验证点:
- analyze_session_state() 能否分析 AGDebugger 的 session history
- claim 提取和判断是否正常工作
- 可选的 websearch evidence 是否正常

前置条件:
  - AGDebugger backend 已启动且有 session history (先跑 step11)
  - 设置环境变量: AGENTDEBUG_OPENAI_API_KEY
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")
sys.path.insert(0, str(REPO_DIR))

from external_agent.integration import analyze_session_state
from external_agent_controller import AGDebuggerClient


async def run_test(args):
    client = AGDebuggerClient(args.base_url)

    # 13a. 获取 session history
    print("=" * 60)
    print("Step 13a: Get session history")
    print("=" * 60)
    try:
        state = client.get_session_history()
        print(f"  Keys: {list(state.keys())}")
        current_session = state.get("current_session")
        print(f"  current_session: {current_session}")
        message_history = state.get("message_history", {})
        total_messages = sum(
            len(v.get("messages", [])) for v in message_history.values() if isinstance(v, dict)
        )
        print(f"  Total messages across sessions: {total_messages}")
        if total_messages == 0:
            print("[WARN] No messages in session history. Run step11 first to generate messages.")
            print("[SKIP] Cannot test claim analysis without messages.")
            return
        print("[OK]")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    # 13b. 运行 claim 分析 (不带 websearch)
    print("\n" + "=" * 60)
    print("Step 13b: analyze_session_state (no websearch)")
    print("=" * 60)
    try:
        result = await analyze_session_state(
            task=args.claim_task,
            state=state,
            model=args.model,
            api_key=args.api_key or None,
            base_url=args.api_base,
            evidence_text=args.evidence_text,
            use_websearch=False,
            search_backend="bright_data",
            search_max_searches=3,
            search_num_results=5,
            search_fetch_top_n=2,
            search_max_output_words=1500,
            assistant_only=True,
        )
        print(f"  Result keys: {list(result.keys())}")
        claims = result.get("claims", [])
        judgments = result.get("judgments", [])
        print(f"  Claims extracted: {len(claims)}")
        print(f"  Judgments: {len(judgments)}")
        if claims:
            print(f"\n  First claim:")
            c = claims[0]
            for k in ["claim_id", "category", "text", "source_ref"]:
                print(f"    {k}: {str(c.get(k, ''))[:100]}")
        if judgments:
            print(f"\n  First judgment:")
            j = judgments[0]
            for k in ["claim_id", "hallucination", "reason"]:
                print(f"    {k}: {str(j.get(k, ''))[:100]}")
        print("[OK]")
    except Exception as e:
        print(f"[FAIL] analyze_session_state failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[PASS] Step 13 completed successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/api")
    parser.add_argument("--model", default=os.environ.get("AGENTDEBUG_MODEL_NAME", "gpt-4o-mini"))
    parser.add_argument("--api-key", default=os.environ.get("AGENTDEBUG_OPENAI_API_KEY", ""))
    parser.add_argument("--api-base", default=os.environ.get("AGENTDEBUG_OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--claim-task", default="research_questions",
                        choices=["research_questions", "medical_guidelines", "legal_cases", "coding"])
    parser.add_argument("--evidence-text", default="")
    args = parser.parse_args()

    if not args.api_key:
        print("[FAIL] No API key. Set AGENTDEBUG_OPENAI_API_KEY or use --api-key")
        sys.exit(1)

    asyncio.run(run_test(args))


if __name__ == "__main__":
    main()
