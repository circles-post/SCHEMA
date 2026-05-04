# AGDebugger debug-loop / stop_loop fix report

## Scope

This note records the investigation and fix for two linked issues:

1. `stop_loop during debug-step-X did not finish cleanly`
2. debug run hangs with `GroupChatTermination` / `list.remove(x): x not in list`

## Root cause

### 1. Group chat transient state was lost across `edit_and_revert`

The actual crash came from AutoGen's group chat manager:

- `autogen_agentchat/.../_base_group_chat_manager.py`
- `self._active_speakers.remove(message.name)`
- exception: `ValueError: list.remove(x): x not in list`

In our flow, `edit_and_revert` does:

1. load a saved runtime checkpoint
2. resend the repaired message
3. let the trajectory continue

But the checkpoint only restored persisted agent state. It did **not** restore the manager's transient `_active_speakers` list. So after replay, the manager received a `GroupChatAgentResponse`, tried to remove the speaker from `_active_speakers`, and crashed because the list was empty.

This corrupted the runtime state, and the later `stop_loop` cleanup started failing as a consequence.

Relevant code/log refs:

- `src/agdebugger/backend.py:159`
- `logs/20260320/run_20260320_231353/server.log`
- `logs/20260320/run_20260320_231353/run.jsonl`

### 2. Repair targeting counted non-editable system messages

`target_turn` resolution previously counted any non-user message as an assistant turn. That included:

- `RoundRobinGroupChatManager` messages
- termination events
- other non-editable envelopes

This made `edit_and_revert target_turn` fragile and could point repairs at the wrong timestamp or wrong envelope.

Relevant code refs:

- `external_agent_controller.py:319`
- `external_agent_controller.py:575`
- `run_dataset_autodebug.py:481`

### 3. `stop_loop` 500 was partly a cleanup bug

After the runtime had already died/stopped, fallback cleanup called `runtime.stop()` again and surfaced:

- `RuntimeError: Runtime is not started`

That turned a cleanup path into an HTTP 500 even when the real failure had already happened earlier.

Relevant code ref:

- `src/agdebugger/backend.py:130`

## Fixes applied

### Backend state repair

Updated `src/agdebugger/backend.py` to:

- store extra checkpoint metadata for transient group-chat state
- capture `_active_speakers` at checkpoint time
- restore `_active_speakers` after `load_state()`
- fall back to the replay message speaker name if metadata is missing
- tolerate `Runtime is not started` during forced/non-forced stop cleanup

Main refs:

- `src/agdebugger/backend.py:45`
- `src/agdebugger/backend.py:130`
- `src/agdebugger/backend.py:159`
- `src/agdebugger/backend.py:198`
- `src/agdebugger/backend.py:243`

### Repair target filtering

Updated `external_agent_controller.py` and `run_dataset_autodebug.py` so assistant-turn counting only includes editable assistant trajectory entries:

- `ThoughtEvent`
- `TextMessage`
- `ToolCallRequestEvent`
- `ToolCallExecutionEvent`
- `GroupChatAgentResponse`

and explicitly excludes:

- `RoundRobinGroupChatManager*`
- termination entries
- user messages

Main refs:

- `external_agent_controller.py:319`
- `external_agent_controller.py:575`
- `external_agent_controller.py:646`
- `external_agent_controller.py:1248`
- `run_dataset_autodebug.py:463`
- `run_dataset_autodebug.py:481`

## Tests added

- `tests/test_external_agent_controller.py`
  - verifies repair target summaries ignore manager/termination entries
  - verifies turn-to-timestamp mapping stays aligned with editable assistant entries
- `tests/test_backend.py`
  - verifies capture/restore of transient `_active_speakers`
  - verifies replay-speaker fallback restoration

Refs:

- `tests/test_external_agent_controller.py:1`
- `tests/test_backend.py:117`

## Validation results

### Successful end-to-end real sample

Command pattern:

- `AGDEBUGGER_FORCE_INITIAL_ANSWER=option6 ... bash run_with_models.sh --component-id 0 --start 70 --limit 1 --debug-max-steps 2`

Observed result:

- initial answer forced wrong: `option6`
- debug step repaired trajectory successfully
- final answer corrected to `option1`
- run exited cleanly with code `0`
- no `stop_loop during ... did not finish cleanly`
- no `list.remove(x): x not in list`

Refs:

- `logs/20260320/run_20260320_233645/run.jsonl`
- `logs/20260320/run_20260320_233645/server.log`

### Additional rerun attempts

I attempted to expand validation to more real samples, but the next runs were blocked earlier by a separate backend-startup issue:

- ToolUniverse MCP initialization timed out
- backend stayed unavailable (`503`) during launcher readiness probing

This is a different failure mode from the fixed debug-loop issue.

Refs:

- `logs/20260320/run_20260320_235950/server.log`
- `logs/20260321/run_20260321_000218/server.log`

Key error:

- `mcp.shared.exceptions.McpError: Timed out while waiting for response to ClientRequest. Waited 120.0 seconds.`

## Current conclusion

The original debug-loop corruption issue is fixed:

- replay no longer loses `_active_speakers`
- cleanup no longer turns an already-stopped runtime into a 500
- repair targeting no longer drifts onto manager/termination messages

The remaining blocker for broader sample sweeps is a separate ToolUniverse MCP startup/readiness timeout, not the `stop_loop` / `list.remove` bug.
