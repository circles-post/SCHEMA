from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from external_agent.integration import analyze_session_state, analyze_text


def _read_text_arg(text: str | None, text_file: Path | None) -> str:
    if text is not None:
        return text
    if text_file is not None:
        return text_file.read_text(encoding="utf-8")
    raise ValueError("Provide --text, --text-file, or --history-state-file.")


async def _run(args: argparse.Namespace) -> None:
    evidence_text = _read_text_arg(args.evidence, args.evidence_file) if (args.evidence or args.evidence_file) else ""
    if args.history_state_file is not None:
        state = json.loads(args.history_state_file.read_text(encoding="utf-8"))
        result = await analyze_session_state(
            task=args.task,
            state=state,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            evidence_text=evidence_text,
            use_websearch=args.use_websearch,
            search_backend=args.search_backend,
            search_max_searches=args.search_max_searches,
            search_num_results=args.search_num_results,
            search_fetch_top_n=args.search_fetch_top_n,
            search_max_output_words=args.search_max_output_words,
            conversation_id=args.conversation_id,
            session_id=args.session_id,
            assistant_only=args.assistant_only and not args.include_user_turns,
        )
    else:
        text = _read_text_arg(args.text, args.text_file)
        result = await analyze_text(
            task=args.task,
            text=text,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            evidence_text=evidence_text,
            use_websearch=args.use_websearch,
            search_backend=args.search_backend,
            search_max_searches=args.search_max_searches,
            search_num_results=args.search_num_results,
            search_fetch_top_n=args.search_fetch_top_n,
            search_max_output_words=args.search_max_output_words,
            conversation_id=args.conversation_id,
            turn_number=args.turn_number,
        )

    print(json.dumps(
            {
                "conversation_id": result["conversation_id"],
                "claim_count": result["claim_count"],
                "judgment_count": result["judgment_count"],
                "hallucination_yes_count": result["hallucination_yes_count"],
                "verification_error_yes_count": result["verification_error_yes_count"],
                "claims": result["claims"],
                "judgments": result["judgments"],
            },
            ensure_ascii=False,
            indent=2,
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run claim extraction and judging from text or AGDebugger state.")
    parser.add_argument("--task", required=True, choices=["research_questions", "medical_guidelines", "legal_cases", "coding"])
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--history-state-file", type=Path, default=None)
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--evidence-file", type=Path, default=None)
    parser.add_argument("--use-websearch", action="store_true")
    parser.add_argument("--search-backend", choices=["bright_data", "serper"], default="bright_data")
    parser.add_argument("--search-max-searches", type=int, default=3)
    parser.add_argument("--search-num-results", type=int, default=5)
    parser.add_argument("--search-fetch-top-n", type=int, default=2)
    parser.add_argument("--search-max-output-words", type=int, default=1500)
    parser.add_argument("--conversation-id", type=int, default=0)
    parser.add_argument("--turn-number", type=int, default=0)
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--assistant-only", action="store_true")
    parser.add_argument("--include-user-turns", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
