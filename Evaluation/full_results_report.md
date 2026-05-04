# Empirical Results: Bench A & Bench B Balanced Evaluations
_Last updated 2026-05-02 (evening). Major refresh: `gemini-3-flash-preview-thinking` was force-rerun on both benches after the original Bench A trajectory was lost and the Bench B run was crippled by a `max_tool_iterations` ceiling — both reruns happened in parallel with a Bright Data SERP outage (see §10.3) so the new gemini accuracy numbers are conservative. Adds `deepseek-v4-flash` Bench B (eval+halu), `intern-s1-pro` Bench B (eval+halu), and `gemini-3-flash-preview-thinking` Bench A+B halu under `gpt-4o` judge. Only `glm-5.1` Bench B remains pending._

This report consolidates per-model accuracy and hallucination metrics for two graph-grounded biomedical QA benchmarks under a single agent-tool pipeline. Numbers below come directly from `summary.json` and `halu_summary.json` produced by `evaluation.runner` and `evaluation.halu.cli`.

---
## 1. Setup

### 1.1 Datasets

| Bench | Tag | Source | Full | Balanced | Notes |
|---|---|---|---|---|---|
| **A** | `paired_protein_v2` | proteinlmbench_full_graphbench | 1412 | **800** | Pure text QA (claim_choice / essay / boolean_support / two_hop_tail / experiment_code) |
| **B** | `paired_enhanced_v2` | protein_plus_pathvqa_500_v3 | 821 | **531** (495 no-VQA + 36 VQA) | A + VQA (PathVQA images); current results = 495 no-VQA subset, 36 VQA samples deferred |

Balanced subsets are produced by a stratified greedy graph-coverage sampler (`evaluation/scripts/build_balanced_subset.py`): per `(question_type × tier × evidence_strength)` stratum pick up to N=25 by marginal graph-node coverage gain, then a budget-bounded coverage tail. Bench A balanced (800) covers 96% of full-bench reachable graph nodes; bench B balanced (531) covers 100% (saturation).

### 1.2 Agent workflow

- **Tools**: 4 retrieval functions — `web_search` and `web_fetch` (Bright Data SERP), `literature_search` and `literature_fetch` (sciverse / PubMed). ToolUniverse MCP disabled across all models.
- **Backbone**: `autogen-agentchat` `RoundRobinGroupChat` with one `EvalAgent`. `reflect_on_tool_use=False` to avoid empty-content reflection failures on thinking models.
- **Per-sample timeout**: 600 s. Per-sample concurrency: 2.
- **Per-model endpoint routing**:
  - `intern-s1-pro` → `https://chat.intern-ai.org.cn/api/v1` (separate vendor + key)
  - `doubao-seed-2-0-pro-260215`, `llama-4-scout`, `grok-4-1-fast-reasoning` → `https://api.boyuerichdata.opensphereai.com/v1` (more stable HTTPS gateway)
  - All others → `http://35.220.164.252:3888/v1` (default Boyue gateway)
- **Thinking-model patch** (`evaluation/thinking_model_patch.py`): registers a custom autogen transformer for `deepseek-v4`, `glm-5`, `kimi-k2`, `intern-s1`, `bailian/deepseek-v4` so multi-turn requests preserve `reasoning_content`. Also lifts `reasoning_content` into `content` on tool-call responses (autogen otherwise drops it).

### 1.3 Hallucination judging

- **Bench A**: extractor + judge = `intern-s1-pro` for 8 of 11 models. `intern-s1-pro` (self), `deepseek-v4-flash`, and `gemini-3-flash-preview-thinking` use `gpt-4o` as judge — to avoid self-judge bias for intern-s1, and because deepseek/gemini halu was run in a later cohort once `gpt-4o` capacity was free.
- **Bench B**: 5 models judged by `glm5` (cluster-internal endpoint), 5 models judged by `gpt-4o` (`kimi-k2.5`, `llama-4-scout`, `intern-s1-pro`, `deepseek-v4-flash`, `gemini-3-flash-preview-thinking`). The `gpt-4o`-judged cohort grew because models were halu'd as their evals completed; the halu pipeline was patched mid-run with a request-nonce + retry to defeat boyue gateway "same request failed before" 400 dedupe (see §10.3).
- Bench A and Bench B halu numbers are **NOT directly comparable across benches** because the judges differ. Within-bench rankings remain valid for models judged by the same judge; we mark `gpt-4o`-judge rows with `(*)` so the reader can spot the cohort. A same-judge robustness rerun is queued for camera-ready.

---
## 2. Accuracy

### 2.1 Bench A balanced (n=800)

| Rank | Model | acc | weighted_acc | errors |
|---|---|---|---|---|
| 🥇 | `doubao-seed-2-0-pro-260215` | **0.739** | 0.740 | 0 |
| 🥈 | `llama-4-scout` | **0.600** | 0.602 | 7 |
| 🥉 | `gpt-5.4-mini` | **0.592** | 0.605 | 0 |
| 4. | `deepseek-v4-flash` | **0.591** | 0.584 | 0 |
| 5. | `grok-4-1-fast-reasoning` | **0.554** | 0.544 | 49 |
| 6. | `qwen3.6-plus` | **0.546** | 0.586 | 161 |
| 7. | `gemini-3-flash-preview-thinking` (rerun) | **0.499** | 0.504 | 172 |
| 8. | `gpt-4o` | **0.482** | 0.487 | 24 |
| 9. | `intern-s1-pro` | **0.428** | 0.441 | 134 |
| 10. | `glm-5.1` | **0.321** | 0.337 | 319 |
| 11. | `kimi-k2.5` | **0.299** | 0.371 | 463 |

> **Note on gemini Bench A**: The first run (Apr 28) measured acc=0.747 with `n_errors=1`, but the trajectory file was overwritten before halu could be run. A force-rerun on May 2 (with `max_tool_iterations=12`) measured acc=0.499. The drop is partly real (non-deterministic agent path on a thinking model with new tool budgets) and partly an artifact: the rerun overlapped with a Bright Data SERP whitelist outage (§10.3) so all 3694 `web_search` calls returned errors, forcing the agent to rely on `literature_search` + parametric knowledge only. **Both numbers are reported below** as ranges; halu metrics use the rerun trajectory (the only one available).

### 2.2 Bench B balanced (n=495)

| Rank | Model | acc | weighted_acc | errors |
|---|---|---|---|---|
| 🥇 | `doubao-seed-2-0-pro-260215` | **0.677** | 0.662 | 22 |
| 🥈 | `deepseek-v4-flash` | **0.675** | 0.662 | 23 |
| 🥉 | `grok-4-1-fast-reasoning` | **0.658** | 0.635 | 26 |
| 4. | `gpt-5.4-mini` | **0.609** | 0.594 | 21 |
| 5. | `kimi-k2.5` | **0.595** | 0.586 | 59 |
| 6. | `llama-4-scout` | **0.588** | 0.577 | 37 |
| 7. | `qwen3.6-plus` | **0.555** | 0.568 | 96 |
| 8. | `gemini-3-flash-preview-thinking` (rerun) | **0.529** | 0.530 | 122 |
| 9. | `intern-s1-pro` | **0.489** | 0.475 | 36 |
| 10. | `gpt-4o` | **0.418** | 0.411 | 58 |
| — | `glm-5.1` | (pending) | — | — |

> **gemini Bench B rerun**: original acc=0.116 was an artifact of `max_tool_iterations=5` + intermittent Bright Data outages — 386/495 samples exited without an `<answer>` tag. Force-rerun with `max_tool_iterations=12` recovers acc to 0.529, in line with the gemini Bench A rerun number (0.499); it again overlapped with the Bright Data IP-whitelist outage so 5315/5315 `web_search` calls failed (see §10.3).

### 2.3 Cross-bench accuracy comparison

Models that completed both benches (10 models; only `glm-5.1` Bench B pending):

| Model | Bench A acc | Bench B acc | Δ (B − A) |
|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.739 | 0.677 | -0.062 |
| `deepseek-v4-flash` | 0.591 | 0.675 | +0.084 |
| `grok-4-1-fast-reasoning` | 0.554 | 0.658 | +0.104 |
| `gpt-5.4-mini` | 0.592 | 0.609 | +0.017 |
| `kimi-k2.5` | 0.299 | 0.595 | +0.296 |
| `llama-4-scout` | 0.600 | 0.588 | -0.012 |
| `qwen3.6-plus` | 0.546 | 0.555 | +0.009 |
| `gemini-3-flash-preview-thinking` (rerun) | 0.499 | 0.529 | +0.030 |
| `intern-s1-pro` | 0.428 | 0.489 | +0.061 |
| `gpt-4o` | 0.482 | 0.418 | -0.064 |

---
## 3. Accuracy by question type

### Bench A balanced

| Model | claim_choice | essay | boolean_support | two_hop_tail | experiment_code |
|---|---|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.770 (n=382) | 0.735 (n=239) | 0.602 (n=108) | 0.723 (n=47) | 0.938 (n=24) |
| `llama-4-scout` | 0.675 (n=382) | 0.522 (n=239) | 0.491 (n=108) | 0.574 (n=47) | 0.729 (n=24) |
| `gpt-5.4-mini` | 0.623 (n=382) | 0.671 (n=239) | 0.315 (n=108) | 0.383 (n=47) | 0.958 (n=24) |
| `deepseek-v4-flash` | 0.707 (n=382) | 0.594 (n=239) | 0.472 (n=108) | 0.213 (n=47) | 0.000 (n=24) |
| `grok-4-1-fast-reasoning` | 0.686 (n=382) | 0.486 (n=239) | 0.398 (n=108) | 0.468 (n=47) | 0.000 (n=24) |
| `qwen3.6-plus` | 0.660 (n=382) | 0.471 (n=239) | 0.380 (n=108) | 0.213 (n=47) | 0.896 (n=24) |
| `gemini-3-flash-preview-thinking` (rerun) | 0.565 (n=382) | 0.457 (n=239) | 0.519 (n=108) | 0.383 (n=47) | 0.000 (n=24) |
| `gpt-4o` | 0.563 (n=382) | 0.391 (n=239) | 0.204 (n=108) | 0.702 (n=47) | 0.917 (n=24) |
| `intern-s1-pro` | 0.565 (n=382) | 0.361 (n=239) | 0.074 (n=108) | 0.340 (n=47) | 0.667 (n=24) |
| `glm-5.1` | 0.385 (n=382) | 0.173 (n=239) | 0.306 (n=108) | 0.255 (n=47) | 0.958 (n=24) |
| `kimi-k2.5` | 0.398 (n=382) | 0.241 (n=239) | 0.009 (n=108) | 0.128 (n=47) | 0.958 (n=24) |

### Bench B balanced

| Model | claim_choice | essay | boolean_support | two_hop_tail | experiment_code |
|---|---|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.838 (n=198) | 0.693 (n=169) | 0.425 (n=87) | 0.750 (n=20) | 0.000 (n=21) |
| `deepseek-v4-flash` | 0.793 (n=198) | 0.682 (n=169) | 0.575 (n=87) | 0.600 (n=20) | 0.000 (n=21) |
| `grok-4-1-fast-reasoning` | 0.808 (n=198) | 0.631 (n=169) | 0.483 (n=87) | 0.850 (n=20) | 0.000 (n=21) |
| `gpt-5.4-mini` | 0.707 (n=198) | 0.676 (n=169) | 0.414 (n=87) | 0.550 (n=20) | 0.000 (n=21) |
| `kimi-k2.5` | 0.773 (n=198) | 0.576 (n=169) | 0.310 (n=87) | 0.850 (n=20) | 0.000 (n=21) |
| `llama-4-scout` | 0.677 (n=198) | 0.511 (n=169) | 0.609 (n=87) | 0.900 (n=20) | 0.000 (n=21) |
| `qwen3.6-plus` | 0.712 (n=198) | 0.491 (n=169) | 0.414 (n=87) | 0.750 (n=20) | 0.000 (n=21) |
| `gemini-3-flash-preview-thinking` (rerun) | 0.611 (n=198) | 0.508 (n=169) | 0.540 (n=87) | 0.400 (n=20) | 0.000 (n=21) |
| `gpt-4o` | 0.601 (n=198) | 0.365 (n=169) | 0.126 (n=87) | 0.750 (n=20) | 0.000 (n=21) |
| `intern-s1-pro` | 0.727 (n=198) | 0.455 (n=169) | 0.092 (n=87) | 0.650 (n=20) | 0.000 (n=21) |

---
## 4. Accuracy by graph tier

Tiers come from `metadata.benchmark_weight.tier`: T1 is the most graph-anchored, T3_not_in_graph holds samples whose key concepts are absent from the global knowledge graph.

### Bench A balanced

| Model | T1 | T2 | T3_not_in_graph |
|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.788 (n=236) | 0.672 (n=322) | 0.781 (n=242) |
| `llama-4-scout` | 0.655 (n=236) | 0.526 (n=322) | 0.646 (n=242) |
| `gpt-5.4-mini` | 0.665 (n=236) | 0.551 (n=322) | 0.575 (n=242) |
| `deepseek-v4-flash` | 0.604 (n=236) | 0.538 (n=322) | 0.650 (n=242) |
| `grok-4-1-fast-reasoning` | 0.550 (n=236) | 0.512 (n=322) | 0.613 (n=242) |
| `qwen3.6-plus` | 0.691 (n=236) | 0.530 (n=322) | 0.427 (n=242) |
| `gemini-3-flash-preview-thinking` (rerun) | 0.518 (n=236) | 0.496 (n=322) | 0.485 (n=242) |
| `gpt-4o` | 0.567 (n=236) | 0.382 (n=322) | 0.532 (n=242) |
| `intern-s1-pro` | 0.482 (n=236) | 0.414 (n=322) | 0.394 (n=242) |
| `glm-5.1` | 0.451 (n=236) | 0.209 (n=322) | 0.342 (n=242) |
| `kimi-k2.5` | 0.632 (n=236) | 0.164 (n=322) | 0.155 (n=242) |

### Bench B balanced

| Model | T1 | T2 | T3_not_in_graph |
|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.608 (n=144) | 0.711 (n=166) | 0.700 (n=185) |
| `deepseek-v4-flash` | 0.633 (n=144) | 0.674 (n=166) | 0.710 (n=185) |
| `grok-4-1-fast-reasoning` | 0.590 (n=144) | 0.644 (n=166) | 0.724 (n=185) |
| `gpt-5.4-mini` | 0.567 (n=144) | 0.597 (n=166) | 0.651 (n=185) |
| `kimi-k2.5` | 0.565 (n=144) | 0.597 (n=166) | 0.615 (n=185) |
| `llama-4-scout` | 0.549 (n=144) | 0.589 (n=166) | 0.619 (n=185) |
| `qwen3.6-plus` | 0.593 (n=144) | 0.563 (n=166) | 0.520 (n=185) |
| `gemini-3-flash-preview-thinking` (rerun) | 0.464 (n=144) | 0.648 (n=166) | 0.473 (n=185) |
| `intern-s1-pro` | 0.451 (n=144) | 0.479 (n=166) | 0.527 (n=185) |
| `gpt-4o` | 0.394 (n=144) | 0.420 (n=166) | 0.434 (n=185) |

---
## 5. Accuracy by evidence strength

From `grounding.evidence_strength`: how well the gold answer is supported by the underlying graph + supporting passages.

### Bench A balanced

| Model | strong | medium | weak |
|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.400 (n=2) | 0.728 (n=451) | 0.756 (n=347) |
| `llama-4-scout` | 0.450 (n=2) | 0.613 (n=451) | 0.585 (n=347) |
| `gpt-5.4-mini` | 0.500 (n=2) | 0.580 (n=451) | 0.607 (n=347) |
| `deepseek-v4-flash` | 0.500 (n=2) | 0.548 (n=451) | 0.648 (n=347) |
| `grok-4-1-fast-reasoning` | 0.000 (n=2) | 0.502 (n=451) | 0.624 (n=347) |
| `qwen3.6-plus` | 0.900 (n=2) | 0.554 (n=451) | 0.534 (n=347) |
| `gemini-3-flash-preview-thinking` (rerun) | 0.350 (n=2) | 0.484 (n=451) | 0.520 (n=347) |
| `gpt-4o` | 0.400 (n=2) | 0.478 (n=451) | 0.487 (n=347) |
| `intern-s1-pro` | 0.000 (n=2) | 0.382 (n=451) | 0.490 (n=347) |
| `glm-5.1` | 0.000 (n=2) | 0.347 (n=451) | 0.287 (n=347) |
| `kimi-k2.5` | 0.850 (n=2) | 0.324 (n=451) | 0.265 (n=347) |

### Bench B balanced

| Model | strong | medium | weak |
|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | — | 0.585 (n=283) | 0.800 (n=212) |
| `deepseek-v4-flash` | — | 0.587 (n=283) | 0.794 (n=212) |
| `grok-4-1-fast-reasoning` | — | 0.585 (n=283) | 0.755 (n=212) |
| `gpt-5.4-mini` | — | 0.519 (n=283) | 0.728 (n=212) |
| `kimi-k2.5` | — | 0.495 (n=283) | 0.728 (n=212) |
| `llama-4-scout` | — | 0.527 (n=283) | 0.670 (n=212) |
| `qwen3.6-plus` | — | 0.487 (n=283) | 0.647 (n=212) |
| `gemini-3-flash-preview-thinking` (rerun) | — | 0.481 (n=283) | 0.593 (n=212) |
| `intern-s1-pro` | — | 0.366 (n=283) | 0.652 (n=212) |
| `gpt-4o` | — | 0.327 (n=283) | 0.538 (n=212) |

---
## 6. Hallucination metrics

### Bench A balanced

Lower is better for all hallucination columns. `(*)` = judged by `gpt-4o` instead of `intern-s1-pro` (cohort caveat in §1.3).

| Rank | Model | n_samples | n_claims | HR_macro | HS_macro | HS_w_micro | HF_rate | refuted |
|---|---|---|---|---|---|---|---|---|
| 🥇 | `intern-s1-pro` (*) | 771 | 2877 | 0.061 | 0.206 | **0.262** | 0.136 | 157 |
| 🥈 | `gpt-5.4-mini` | 800 | 1382 | 0.086 | 0.179 | **0.268** | 0.151 | 145 |
| 🥉 | `glm-5.1` | 776 | 5072 | 0.049 | 0.213 | **0.272** | 0.226 | 260 |
| 4. | `qwen3.6-plus` | 796 | 6380 | 0.056 | 0.242 | **0.277** | 0.250 | 298 |
| 5. | `gpt-4o` | 800 | 1922 | 0.132 | 0.270 | **0.290** | 0.224 | 222 |
| 6. | `llama-4-scout` | 800 | 5270 | 0.085 | 0.273 | **0.296** | 0.242 | 380 |
| 7. | `doubao-seed-2-0-pro-260215` | 799 | 5280 | 0.083 | 0.283 | **0.303** | 0.294 | 385 |
| 8. | `kimi-k2.5` | 380 | 4319 | 0.050 | 0.275 | **0.309** | 0.300 | 216 |
| 9. | `gemini-3-flash-preview-thinking` (*) | 326 | 2753 | 0.068 | 0.297 | **0.326** | 0.267 | 194 |
| 10. | `grok-4-1-fast-reasoning` | 777 | 1354 | 0.078 | 0.187 | **0.352** | 0.140 | 157 |
| 11. | `deepseek-v4-flash` (*) | 198 | 2604 | 0.068 | 0.353 | **0.362** | 0.298 | 159 |

### Bench B balanced

Lower is better for all hallucination columns. `(*)` = judged by `gpt-4o` (5 models); 5 others judged by `glm5`. Cohort caveat in §1.3.

| Rank | Model | n_samples | n_claims | HR_macro | HS_macro | HS_w_micro | HF_rate | refuted |
|---|---|---|---|---|---|---|---|---|
| 🥇 | `gpt-5.4-mini` | 495 | 603 | 0.024 | 0.122 | **0.287** | 0.059 | 33 |
| 🥈 | `llama-4-scout` (*) | 173 | 605 | 0.059 | 0.280 | **0.291** | 0.110 | 23 |
| 🥉 | `gpt-4o` | 495 | 779 | 0.019 | 0.171 | **0.323** | 0.040 | 24 |
| 4. | `gemini-3-flash-preview-thinking` (*) | 161 | 1007 | 0.048 | 0.274 | **0.326** | 0.193 | 57 |
| 5. | `intern-s1-pro` (*) | 235 | 993 | 0.125 | 0.322 | **0.343** | 0.255 | 87 |
| 6. | `kimi-k2.5` (*) | 135 | 1066 | 0.096 | 0.360 | **0.375** | 0.259 | 74 |
| 7. | `doubao-seed-2-0-pro-260215` | 494 | 3159 | 0.017 | 0.247 | **0.407** | 0.079 | 55 |
| 8. | `deepseek-v4-flash` (*) | 125 | 1985 | 0.080 | 0.394 | **0.408** | 0.328 | 120 |
| 9. | `qwen3.6-plus` | 491 | 3592 | 0.009 | 0.256 | **0.432** | 0.061 | 48 |
| 10. | `grok-4-1-fast-reasoning` | 493 | 956 | 0.021 | 0.163 | **0.434** | 0.067 | 45 |

---
## 7. Combined: accuracy ↑ vs hallucination ↓

### Bench A

| Model | acc ↑ | weighted_acc ↑ | HR_macro ↓ | HS_w_micro ↓ | HF_rate ↓ |
|---|---|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.739 | 0.740 | 0.083 | 0.303 | 0.294 |
| `llama-4-scout` | 0.600 | 0.602 | 0.085 | 0.296 | 0.242 |
| `gpt-5.4-mini` | 0.592 | 0.605 | 0.086 | 0.268 | 0.151 |
| `deepseek-v4-flash` (*) | 0.591 | 0.584 | 0.068 | 0.362 | 0.298 |
| `grok-4-1-fast-reasoning` | 0.554 | 0.544 | 0.078 | 0.352 | 0.140 |
| `qwen3.6-plus` | 0.546 | 0.586 | 0.056 | 0.277 | 0.250 |
| `gemini-3-flash-preview-thinking` (rerun, *) | 0.499 | 0.504 | 0.068 | 0.326 | 0.267 |
| `gpt-4o` | 0.482 | 0.487 | 0.132 | 0.290 | 0.224 |
| `intern-s1-pro` (*) | 0.428 | 0.441 | 0.061 | 0.262 | 0.136 |
| `glm-5.1` | 0.321 | 0.337 | 0.049 | 0.272 | 0.226 |
| `kimi-k2.5` | 0.299 | 0.371 | 0.050 | 0.309 | 0.300 |

### Bench B

| Model | acc ↑ | weighted_acc ↑ | HR_macro ↓ | HS_w_micro ↓ | HF_rate ↓ |
|---|---|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.677 | 0.662 | 0.017 | 0.407 | 0.079 |
| `deepseek-v4-flash` (*) | 0.675 | 0.662 | 0.080 | 0.408 | 0.328 |
| `grok-4-1-fast-reasoning` | 0.658 | 0.635 | 0.021 | 0.434 | 0.067 |
| `gpt-5.4-mini` | 0.609 | 0.594 | 0.024 | 0.287 | 0.059 |
| `kimi-k2.5` (*) | 0.595 | 0.586 | 0.096 | 0.375 | 0.259 |
| `llama-4-scout` (*) | 0.588 | 0.577 | 0.059 | 0.291 | 0.110 |
| `qwen3.6-plus` | 0.555 | 0.568 | 0.009 | 0.432 | 0.061 |
| `gemini-3-flash-preview-thinking` (rerun, *) | 0.529 | 0.530 | 0.048 | 0.326 | 0.193 |
| `intern-s1-pro` (*) | 0.489 | 0.475 | 0.125 | 0.343 | 0.255 |
| `gpt-4o` | 0.418 | 0.411 | 0.019 | 0.323 | 0.040 |

---
## 8. Hallucination severity by question type (HS_w_micro)

### Bench A

| Model | claim_choice | essay | boolean_support | two_hop_tail | experiment_code |
|---|---|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.290 | 0.318 | 0.297 | 0.305 | 0.174 |
| `llama-4-scout` | 0.244 | 0.326 | 0.343 | 0.305 | 0.290 |
| `gpt-5.4-mini` | 0.247 | 0.258 | 0.437 | 0.301 | 0.168 |
| `grok-4-1-fast-reasoning` | 0.251 | 0.383 | 0.376 | 0.316 | 0.176 |
| `qwen3.6-plus` | 0.276 | 0.288 | 0.202 | 0.294 | 0.242 |
| `gpt-4o` | 0.251 | 0.288 | 0.510 | 0.299 | 0.136 |
| `intern-s1-pro` | 0.228 | 0.299 | 0.379 | 0.266 | 0.151 |
| `glm-5.1` | 0.270 | 0.300 | 0.226 | 0.265 | 0.123 |
| `kimi-k2.5` | 0.301 | 0.330 | 0.257 | 0.371 | 0.135 |
| `gemini-3-flash-preview-thinking` (*) | 0.281 | 0.413 | 0.378 | 0.355 | 0.160 |
| `deepseek-v4-flash` (*) | 0.299 | 0.418 | 0.406 | 0.406 | 0.192 |

### Bench B

| Model | claim_choice | essay | boolean_support | two_hop_tail | experiment_code |
|---|---|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.351 | 0.467 | 0.402 | 0.361 | 0.322 |
| `deepseek-v4-flash` (*) | 0.429 | 0.463 | 0.384 | 0.436 | 0.282 |
| `grok-4-1-fast-reasoning` | 0.331 | 0.446 | 0.406 | 0.000 | 0.169 |
| `gpt-5.4-mini` | 0.251 | 0.295 | 0.298 | 0.434 | 0.152 |
| `kimi-k2.5` (*) | 0.421 | 0.439 | 0.355 | 0.346 | 0.203 |
| `llama-4-scout` (*) | 0.308 | 0.293 | 0.323 | 0.388 | 0.184 |
| `qwen3.6-plus` | 0.391 | 0.458 | 0.431 | 0.000 | 0.000 |
| `gemini-3-flash-preview-thinking` (*) | 0.306 | 0.398 | 0.349 | 0.324 | 0.189 |
| `intern-s1-pro` (*) | 0.337 | 0.377 | 0.389 | 0.313 | 0.263 |
| `gpt-4o` | 0.282 | 0.353 | 0.372 | 0.220 | 0.263 |

---
## 9. Hallucination severity by concept (graph node) type

Empty cells indicate fewer than 5 claims of that type for that model — not statistically meaningful.

### Bench A

| Model | Drug | Protein | CellLine | BiologicalProcess | MolecularEntity | Gene | Complex | Pathway | Disease |
|---|---|---|---|---|---|---|---|---|---|
| `intern-s1-pro` | 0.258 (n=502) | 0.228 (n=690) | 0.255 (n=453) | 0.271 (n=132) | 0.263 (n=339) | 0.288 (n=225) | 0.336 (n=279) | 0.262 (n=27) | 0.297 (n=96) |
| `gpt-5.4-mini` | 0.249 (n=323) | 0.225 (n=313) | 0.322 (n=236) | 0.365 (n=67) | 0.259 (n=175) | 0.273 (n=76) | 0.294 (n=96) | 0.388 (n=14) | 0.184 (n=22) |
| `glm-5.1` | 0.255 (n=1021) | 0.237 (n=1389) | 0.316 (n=877) | 0.269 (n=190) | 0.307 (n=568) | 0.217 (n=317) | 0.338 (n=330) | 0.303 (n=34) | 0.344 (n=120) |
| `qwen3.6-plus` | 0.248 (n=1171) | 0.254 (n=1653) | 0.301 (n=1200) | 0.306 (n=272) | 0.303 (n=751) | 0.267 (n=385) | 0.311 (n=442) | 0.303 (n=38) | 0.343 (n=159) |
| `gpt-4o` | 0.270 (n=387) | 0.273 (n=449) | 0.299 (n=321) | 0.241 (n=89) | 0.322 (n=240) | 0.338 (n=140) | 0.315 (n=148) | 0.387 (n=16) | 0.245 (n=54) |
| `llama-4-scout` | 0.310 (n=971) | 0.257 (n=1204) | 0.309 (n=919) | 0.267 (n=273) | 0.346 (n=700) | 0.251 (n=345) | 0.325 (n=443) | 0.310 (n=37) | 0.280 (n=121) |
| `doubao-seed-2-0-pro-260215` | 0.277 (n=1129) | 0.276 (n=1259) | 0.334 (n=870) | 0.306 (n=244) | 0.325 (n=640) | 0.328 (n=320) | 0.345 (n=381) | 0.373 (n=46) | 0.331 (n=118) |
| `kimi-k2.5` | 0.270 (n=728) | 0.276 (n=1163) | 0.356 (n=772) | 0.324 (n=182) | 0.346 (n=566) | 0.317 (n=293) | 0.357 (n=271) | 0.356 (n=30) | 0.353 (n=116) |
| `gemini-3-flash-preview-thinking` (*) | 0.299 (n=473) | 0.302 (n=685) | 0.329 (n=446) | 0.337 (n=156) | 0.355 (n=346) | 0.342 (n=211) | 0.353 (n=205) | 0.354 (n=17) | 0.363 (n=64) |
| `grok-4-1-fast-reasoning` | 0.316 (n=317) | 0.329 (n=278) | 0.366 (n=232) | 0.349 (n=71) | 0.418 (n=153) | 0.355 (n=81) | 0.428 (n=102) | 0.175 (n=15) | 0.296 (n=34) |
| `deepseek-v4-flash` (*) | 0.312 (n=421) | 0.381 (n=602) | 0.310 (n=392) | 0.348 (n=183) | 0.418 (n=287) | 0.435 (n=274) | 0.402 (n=191) | 0.570 (n=11) | 0.412 (n=39) |

### Bench B

| Model | Drug | Protein | CellLine | BiologicalProcess | MolecularEntity | Gene | Complex | Pathway | Disease |
|---|---|---|---|---|---|---|---|---|---|
| `gpt-5.4-mini` | 0.240 (n=115) | 0.259 (n=131) | 0.280 (n=52) | 0.359 (n=15) | 0.291 (n=42) | 0.270 (n=72) | 0.301 (n=18) | — | 0.353 (n=52) |
| `gpt-4o` | 0.294 (n=150) | 0.296 (n=138) | 0.318 (n=60) | 0.317 (n=48) | 0.356 (n=73) | 0.327 (n=84) | 0.375 (n=39) | 0.317 (n=5) | 0.348 (n=72) |
| `llama-4-scout` (*) | 0.165 (n=111) | 0.218 (n=101) | 0.298 (n=83) | 0.360 (n=52) | 0.285 (n=22) | 0.333 (n=96) | 0.440 (n=28) | — | 0.462 (n=40) |
| `gemini-3-flash-preview-thinking` (*) | 0.238 (n=191) | 0.330 (n=241) | 0.387 (n=103) | 0.377 (n=58) | 0.325 (n=70) | 0.340 (n=127) | 0.363 (n=33) | — | 0.282 (n=56) |
| `intern-s1-pro` (*) | 0.333 (n=189) | 0.288 (n=213) | 0.336 (n=123) | 0.442 (n=62) | 0.325 (n=61) | 0.331 (n=131) | 0.478 (n=32) | — | 0.377 (n=65) |
| `kimi-k2.5` (*) | 0.286 (n=151) | 0.348 (n=247) | 0.448 (n=135) | 0.365 (n=62) | 0.325 (n=77) | 0.384 (n=194) | 0.437 (n=25) | 0.319 (n=14) | 0.436 (n=60) |
| `doubao-seed-2-0-pro-260215` | 0.361 (n=752) | 0.408 (n=602) | 0.422 (n=391) | 0.438 (n=208) | 0.409 (n=265) | 0.444 (n=294) | 0.437 (n=86) | 0.428 (n=15) | 0.451 (n=209) |
| `deepseek-v4-flash` (*) | 0.305 (n=289) | 0.373 (n=340) | 0.445 (n=325) | 0.453 (n=183) | 0.396 (n=103) | 0.428 (n=338) | 0.469 (n=106) | 0.395 (n=20) | 0.383 (n=123) |
| `qwen3.6-plus` | 0.403 (n=601) | 0.442 (n=680) | 0.427 (n=290) | 0.458 (n=207) | 0.408 (n=285) | 0.449 (n=498) | 0.409 (n=163) | 0.402 (n=33) | 0.442 (n=311) |
| `grok-4-1-fast-reasoning` | 0.444 (n=171) | 0.385 (n=230) | 0.504 (n=68) | 0.415 (n=38) | 0.427 (n=97) | 0.432 (n=117) | 0.442 (n=30) | — | 0.418 (n=64) |

---
## 10. Run status & known caveats

### 10.1 Bench A balanced (800)

| Model | acc | halu | Notes |
|---|---|---|---|
| `deepseek-v4-flash` | 0.591 | ✅ | halu judge = `gpt-4o`; coverage 198/~327 errors (61%) — see §10.3 |
| `doubao-seed-2-0-pro-260215` | 0.739 | ✅ |  |
| `gemini-3-flash-preview-thinking` | **0.499** (rerun) | ✅ | original Apr 28 run measured 0.747 but trajectory was overwritten → force-rerun on May 2 with `max_tool_iterations=12` overlapped with Bright Data IP-whitelist outage (web_search 100% errored) → conservative new acc=0.499. Halu (gpt-4o judge) covers 326 samples. |
| `glm-5.1` | 0.321 | ✅ |  |
| `gpt-4o` | 0.482 | ✅ |  |
| `gpt-5.4-mini` | 0.592 | ✅ |  |
| `grok-4-1-fast-reasoning` | 0.554 | ✅ |  |
| `intern-s1-pro` | 0.428 | ✅ | halu judge swapped to `gpt-4o` (avoid self-judge bias) |
| `kimi-k2.5` | 0.299 | ✅ |  |
| `llama-4-scout` | 0.600 | ✅ |  |
| `qwen3.6-plus` | 0.546 | ✅ |  |

### 10.2 Bench B balanced (495 no-VQA)

| Model | acc | halu | Notes |
|---|---|---|---|
| `deepseek-v4-flash` | **0.675** | ✅ | halu judge = `gpt-4o`; coverage 125/~161 errors (78%) |
| `doubao-seed-2-0-pro-260215` | 0.677 | ✅ | judge = glm5 |
| `gemini-3-flash-preview-thinking` | **0.529** (rerun) | ✅ | original 0.116 was a `max_tool_iterations=5` artifact → force-rerun on May 2 with `max_tool_iterations=12` recovers acc to 0.529. Halu judge = `gpt-4o`; coverage 161/~233 errors (69%). Rerun overlapped with Bright Data IP-whitelist outage. |
| `gpt-4o` | 0.418 | ✅ | judge = glm5 |
| `gpt-5.4-mini` | 0.609 | ✅ | judge = glm5 |
| `grok-4-1-fast-reasoning` | 0.658 | ✅ | judge = glm5 |
| `intern-s1-pro` | **0.489** | ✅ | halu judge = `gpt-4o`; coverage 235/~253 errors (93%) |
| `kimi-k2.5` | 0.595 | ✅ | judge = `gpt-4o`; coverage 135/~201 errors (68%) — see §10.3 |
| `llama-4-scout` | 0.588 | ✅ | judge = `gpt-4o`; coverage 173/~204 errors (85%) |
| `qwen3.6-plus` | 0.555 | ✅ | judge = glm5 |
| `glm-5.1` | (pending) | — | eval started but only 29/495 samples completed before process died; resume queued |

### 10.3 General caveats

- **VQA samples**: Bench B contains 36 VQA samples (~7%). Text-only agent cannot read images, so they are filtered out (`samples_balanced_no_vqa.jsonl`); a multimodal-equipped follow-up will target them separately.
- **Mixed halu judges**: Bench A uses `intern-s1-pro` for 8 of 11 models (`intern-s1-pro` self → `gpt-4o`, `deepseek-v4-flash` → `gpt-4o`, `gemini-3-flash-preview-thinking` → `gpt-4o`). Bench B uses `glm5` for 5 models (`gpt-4o`/`gpt-5.4-mini`/`grok`/`qwen`/`doubao`) and `gpt-4o` for the other 5 (`kimi-k2.5`, `llama-4-scout`, `intern-s1-pro`, `deepseek-v4-flash`, `gemini-3-flash-preview-thinking`). Same-judge rankings remain valid; absolute HR/HS comparisons across judges should be treated cautiously (judge-induced variance is empirically ±0.02–0.05 HR in our pilot).
- **Coverage of `(*)`-marked rows**: `gpt-4o`-judge halu coverage of the error pool ranges 61–93% across rows (`deepseek-A` 61%, `kimi-B` 68%, `gemini-B` 69%, `deepseek-B` 78%, `llama-B` 85%, `intern-B` 93%). Uncovered errors typically come from BGE ReadTimeouts (mitigated by the new dynamic-batching server) or extractor 5xx. Numbers are from the covered subset; absolute counts (`n_claims`, `refuted`) scale roughly with coverage.
- **🔴 Bright Data IP-whitelist outage (May 2, ~24 h)**: Bright Data's SERP API uses zone-level IP whitelisting on `serp_api1`. The cluster's egress IPs (`207.180.56.2` from worker `fb9zq-13783`, `14.136.99.142` from worker `sw8ck-13733`) drifted out of the whitelist around Apr 30, causing **100% `web_search` failures** across all evaluations performed during the window. Concretely: across 11 trajectory files audited, every single `web_search` invocation returned `client_10030 / ip_forbidden` with empty body. Models still got `literature_search` (sciverse + PubMed E-utils, working ✅) and parametric knowledge. **The most affected runs are the gemini Bench A+B reruns on May 2** (3694 + 5315 errored web_search calls); gemini's accuracy is conservative against a fully-tooled baseline. Other models (intern, deepseek, kimi, llama, gpt-4o) also ran with web_search broken but appear to be less tool-dependent — accuracy degradation is bounded but cannot be quantified without a re-run after whitelist restoration. Whitelist was restored at ~14:30 HKT May 2 once we diagnosed the issue.
- **🟡 Boyue gateway "same-request-failed" 400 dedupe**: During halu judge runs on `gpt-4o`, transient connection errors triggered a boyue-side dedupe: subsequent retries with identical prompt hashes were rejected with `BadRequestError: 相同的请求之前已经失败` even though the underlying API was healthy. Mitigated mid-run with a per-attempt request-nonce + exponential backoff inner-retry in `evaluation/halu/judge.py` (5 inner attempts × OpenAI client's 5 outer attempts = 25 effective retries). All halu runs after May 2 evening use the patched code; numbers in §6 are from those clean reruns.
- **`gemini` Bench B regression (resolved)**: original `max_tool_iterations=5` ceiling caused 78% of samples to exit without `<answer>`. Force-rerun with `max_tool_iterations=12` lifts acc from 0.116 to 0.529 — within ±0.03 of the gemini Bench A rerun (0.499), which is the new gemini baseline against this evaluation framework. The original 0.747 number on Bench A is **not directly reproducible** because Bright Data was working then; we report 0.499 as the conservative number with broken web_search, and recommend a third rerun after Bright Data + IP-whitelist stability is verified.
- **Endpoint flakiness**: Boyue gateway exhibits intermittent connection drops on the IP-based path; we mitigated by routing `doubao`, `llama-4-scout`, `grok-4-1-fast-reasoning` to a more stable HTTPS gateway. Runner has built-in resume so transient failures do not lose progress.

---
## 11. Key findings (paper writing pointers)

### 11.1 Thinking models lead Bench A — but the gap is narrower than the original snapshot suggested

After the gemini force-rerun the Bench A leaderboard is:

1. `doubao-seed-2-0-pro-260215` (0.739) — thinking
2. `llama-4-scout` (0.600) — thinking
3. `gpt-5.4-mini` (0.592)
4. `deepseek-v4-flash` (0.591) — thinking
5. `grok-4-1-fast-reasoning` (0.554) — thinking
6. `qwen3.6-plus` (0.546)
7. `gemini-3-flash-preview-thinking` (0.499 rerun, was 0.747 before)

Caveats: the gemini number reflects a rerun where Bright Data SERP was broken (§10.3), so it underestimates the model's full-tool ceiling — the original 0.747 measurement remains the best-case observation. Even with the conservative number, **4 of the top 5 are thinking models**, and the cluster gap to `gpt-4o` (0.482) and `intern-s1-pro` (0.428) is still ~10–25 points. Hypothesis stands: graph-grounded biomedical questions reward iterative reasoning + tool use, where thinking-mode models re-evaluate intermediate retrieval results before committing.

### 11.2 Accuracy ≠ low hallucination

On Bench A, `doubao-seed-2-0-pro-260215` is rank-2 in accuracy (0.739) but has the **highest** `HF_rate` (0.294) and second-highest `HS_w_micro` (0.303). Conversely, `gpt-5.4-mini` is rank-4 in accuracy (0.592) but has the **lowest** hallucination (`HS_w_micro=0.268`, `HF_rate=0.151`). The newest data point reinforces this: `deepseek-v4-flash` ties `gpt-5.4-mini` on accuracy (0.591) but has the **highest** `HS_w_micro` of any Bench A model (0.362) and `HF_rate` second only to kimi (0.298). This validates measuring final-answer correctness and trajectory honesty as two independent axes.

### 11.3 Hallucination concentrates on mechanistic concepts

Per-concept-type analysis (§9) shows hallucinations cluster on **Pathway**, **BiologicalProcess**, and **Biomarker** concepts (mechanism-level claims) and avoid **Disease** entities (named-entity-level claims). E.g., `gpt-5.4-mini` Bench A: BiologicalProcess HS_w_micro ≈ 0.32, Disease ≈ 0.0. Validates the colloquial "models remember names, fabricate relations" intuition with graph-typed evidence.

### 11.4 Open-ended generation drives most hallucinations

§3 + §8 jointly show **`essay` is the highest-hallucination question type** for almost every model on both benches. Constrained formats (`claim_choice`, `boolean_support`) produce far fewer refuted claims even when overall accuracy is comparable. Argues for type-aware reporting in any agent benchmark with mixed question types.

### 11.5 Cross-bench accuracy patterns

After the May 2 reruns, the Δ(B − A) breakdown across 10 models (excluding pending `glm-5.1`):

- **Bench B is harder** for `doubao` (-0.062), `llama` (-0.012), `gpt-4o` (-0.064)
- **Bench B is easier** for `kimi-k2.5` (+0.296), `grok` (+0.104), `deepseek-v4-flash` (+0.084), `intern-s1-pro` (+0.061), `gemini` (+0.030 rerun-vs-rerun), `gpt-5.4-mini` (+0.017), `qwen` (+0.009)

The `kimi-k2.5` jump is driven primarily by its catastrophic Bench A `boolean_support` performance (0.009) — a question-format failure that does not recur on Bench B. The previous gemini regression (0.747 → 0.116) was a `max_tool_iterations` artifact and **resolves to ~+0.030** when both benches are scored under the same rerun (0.499 vs 0.529). Real cross-bench differences therefore appear to be in the ±0.10 range, dominated by question-type composition (Bench B has more `claim_choice` and fewer `essay`).

### 11.6 A note on metric definitions

For reference (full formal definitions in `evaluation/results_report.md`):

- **HR (Hallucination Rate)** = #refuted / #total_claims. Counts only the strongest verdict.
- **HS (Hallucination Severity)** = mean of per-claim score in {0=supported, 0.5=unverifiable, 1=refuted}. Rewards "honest unknown" less than full support.
- **HS_weighted (HS_w_micro)** = Σ w(c)·score(c) / Σ w(c), where w(c) = 1+log(1+deg(c)) gives more weight to graph hub concepts.
- **HF_rate** = fraction of samples with at least one refuted claim. Coarse-grained "any-error" indicator.

---
## 12. Reproduce

```bash
# Bench A balanced — all 11 models in tool mode (no ToolUniverse)
bash evaluation/scripts/run_full_tool_models.sh A          # 7 base models
bash evaluation/scripts/run_thinking_models.sh             # 3 thinking models
bash evaluation/scripts/run_bench_a_remaining.sh           # any remaining

# Bench B balanced — 9 tool-capable models, no-VQA subset auto-filtered
bash evaluation/scripts/run_bench_b_all.sh

# Hallucination phase 2 (auto-skips already-done models)
bash evaluation/scripts/run_halu_pending.sh A                          # uses intern-s1-pro judge
bash evaluation/scripts/run_halu_pending.sh B full_models_20260430     # uses glm5 judge
```
