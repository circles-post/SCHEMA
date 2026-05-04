# External Agent Claim/Judge Framework

This directory contains a lightweight framework for two tasks:

1. Extract `claim` objects from assistant responses.
2. Judge each claim against evidence.

It is intentionally smaller than HalluHard, but it follows the same split:

- `strategies.py`: domain-specific schema and prompts
- `claim_extractor.py`: LLM-based extraction
- `judge.py`: per-claim judgment
- `pipeline.py`: orchestration
- `agdebugger_adapter.py`: convert AGDebugger session history into turns
- `cli.py`: runnable entrypoint

## Supported domains

- `research_questions`
- `medical_guidelines`
- `legal_cases`
- `coding`

## Run on plain text

From the `agdebugger` repo root:

```bash
python -m external_agent.cli \
  --task research_questions \
  --text-file response.txt \
  --evidence-file evidence.txt \
  --model gpt-4o-mini
```

## Run on AGDebugger session state

If you saved the `/getSessionHistory` response to `session_state.json`:

```bash
python -m external_agent.cli \
  --task research_questions \
  --history-state-file session_state.json \
  --assistant-only \
  --evidence-file evidence.txt \
  --model gpt-4o-mini
```

## Integration direction

If you want to use this inside the external AGDebugger controller loop, the normal pattern is:

1. Pull current session history from AGDebugger.
2. Convert it with `load_turns_from_agdebugger_state(...)`.
3. Run `ClaimJudgePipeline`.
4. Use the extracted claims or judgments to decide whether to `edit_and_revert`, `insert_after`, `send`, or `step`.
