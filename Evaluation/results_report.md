# bench A balanced — Evaluation Results Report
_Generated: 2026-04-29 21:15._
Dataset: `qgen_paired_protein_v2/samples_balanced.jsonl` (800 samples)
Run: `paired_protein_v2_balanced__full_tool_models_20260427`

---
## 1. Run status
| Model | answers | scored | trajectory | acc | halu? |
|---|---|---|---|---|---|
| `gemini-3-flash-preview-thinking` | 146 | 800 | 0 | 0.747 | — |
| `doubao-seed-2-0-pro-260215` | 800 | 800 | 799 | 0.739 | ✅ |
| `llama-4-scout` | 800 | 800 | 800 | 0.600 | ✅ |
| `gpt-5.4-mini` | 800 | 800 | 800 | 0.592 | ✅ |
| `qwen3.6-plus` | 800 | 800 | 796 | 0.546 | ✅ |
| `gpt-4o` | 800 | 800 | 800 | 0.482 | ✅ |
| `glm-5.1` | 800 | 800 | 776 | 0.321 | ✅ |
| `kimi-k2.5` | 800 | 800 | 380 | 0.299 | — |
| `deepseek-v4-flash` | 82 | 0 | 80 | — | — |
| `grok-4-1-fast-reasoning` | 673 | 0 | 652 | — | — |
| `intern-s1-pro` | 290 | 0 | 266 | — | — |

**Completed (with summary.json)**: 8
**In progress / failed**: 3

---
## 2. Accuracy leaderboard
Sorted by overall `acc` descending. `weighted_acc` weights each sample by `metadata.benchmark_weight.weight` from the bench generator.

| Rank | Model | acc | weighted_acc | errors |
|---|---|---|---|---|
| 🥇 | `gemini-3-flash-preview-thinking` | **0.747** | 0.748 | 1 |
| 🥈 | `doubao-seed-2-0-pro-260215` | **0.739** | 0.740 | 0 |
| 🥉 | `llama-4-scout` | **0.600** | 0.602 | 7 |
| 4. | `gpt-5.4-mini` | **0.592** | 0.605 | 0 |
| 5. | `qwen3.6-plus` | **0.546** | 0.586 | 161 |
| 6. | `gpt-4o` | **0.482** | 0.487 | 24 |
| 7. | `glm-5.1` | **0.321** | 0.337 | 319 |
| 8. | `kimi-k2.5` | **0.299** | 0.371 | 463 |

---
## 3. Hallucination metrics (lower is better)
All metrics computed via `evaluation.halu.cli` with extractor + judge = `intern-s1-pro`. Verdicts mapped deterministically: supported→0.0, unverifiable→0.5, refuted→1.0.

See [Hallucination metric definitions](#metric-definitions) below for what each column means.

| Rank | Model | n_claims | HR_macro | HS_macro | HS_w_micro | HF_rate | refuted |
|---|---|---|---|---|---|---|---|
| 🥇 | `gpt-5.4-mini` | 1382 | 0.086 | 0.179 | **0.268** | 0.151 | 145 |
| 🥈 | `glm-5.1` | 5072 | 0.049 | 0.213 | **0.272** | 0.226 | 260 |
| 🥉 | `qwen3.6-plus` | 6380 | 0.056 | 0.242 | **0.277** | 0.250 | 298 |
| 4. | `gpt-4o` | 1922 | 0.132 | 0.270 | **0.290** | 0.224 | 222 |
| 5. | `llama-4-scout` | 5270 | 0.085 | 0.273 | **0.296** | 0.242 | 380 |
| 6. | `doubao-seed-2-0-pro-260215` | 5280 | 0.083 | 0.283 | **0.303** | 0.294 | 385 |

---
## 4. Combined view (acc + halu)
Models with both eval and halu done. Sort by acc desc; HR / HS_w show separately for hallucination judgment.

| Model | acc ↑ | weighted_acc ↑ | HR_macro ↓ | HS_w_micro ↓ | HF_rate ↓ |
|---|---|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.739 | 0.740 | 0.083 | 0.303 | 0.294 |
| `llama-4-scout` | 0.600 | 0.602 | 0.085 | 0.296 | 0.242 |
| `gpt-5.4-mini` | 0.592 | 0.605 | 0.086 | 0.268 | 0.151 |
| `qwen3.6-plus` | 0.546 | 0.586 | 0.056 | 0.277 | 0.250 |
| `gpt-4o` | 0.482 | 0.487 | 0.132 | 0.290 | 0.224 |
| `glm-5.1` | 0.321 | 0.337 | 0.049 | 0.272 | 0.226 |

---
## 5. Accuracy by question type
Question types in bench A balanced: `claim_choice` (382), `essay` (239), `boolean_support` (108), `two_hop_tail` (47), `experiment_code` (24).

| Model | claim_choice | essay | boolean_support | two_hop_tail | experiment_code |
|---|---|---|---|---|---|
| `gemini-3-flash-preview-thinking` | 0.764 | 0.679 | 0.787 | 0.766 | 0.938 |
| `doubao-seed-2-0-pro-260215` | 0.770 | 0.735 | 0.602 | 0.723 | 0.938 |
| `llama-4-scout` | 0.675 | 0.522 | 0.491 | 0.574 | 0.729 |
| `gpt-5.4-mini` | 0.623 | 0.671 | 0.315 | 0.383 | 0.958 |
| `qwen3.6-plus` | 0.660 | 0.471 | 0.380 | 0.213 | 0.896 |
| `gpt-4o` | 0.563 | 0.391 | 0.204 | 0.702 | 0.917 |
| `glm-5.1` | 0.385 | 0.173 | 0.306 | 0.255 | 0.958 |
| `kimi-k2.5` | 0.398 | 0.241 | 0.009 | 0.128 | 0.958 |

---
## 6. Accuracy by graph tier
Tiers from `metadata.benchmark_weight.tier`: `T1` (most graph-anchored), `T2`, `T3_not_in_graph` (concepts outside the global graph).

| Model | T1 | T2 | T3_not_in_graph |
|---|---|---|---|
| `gemini-3-flash-preview-thinking` | 0.801 | 0.674 | 0.792 |
| `doubao-seed-2-0-pro-260215` | 0.788 | 0.672 | 0.781 |
| `llama-4-scout` | 0.655 | 0.526 | 0.646 |
| `gpt-5.4-mini` | 0.665 | 0.551 | 0.575 |
| `qwen3.6-plus` | 0.691 | 0.530 | 0.427 |
| `gpt-4o` | 0.567 | 0.382 | 0.532 |
| `glm-5.1` | 0.451 | 0.209 | 0.342 |
| `kimi-k2.5` | 0.632 | 0.164 | 0.155 |

---
## 7. Accuracy by evidence strength
From `grounding.evidence_strength`: `weak` / `medium` / `strong`. Reflects how well the gold answer is supported by the underlying graph + supporting passages.

| Model | strong | medium | weak |
|---|---|---|---|
| `gemini-3-flash-preview-thinking` | 0.250 (n=2) | 0.741 (n=451) | 0.757 (n=347) |
| `doubao-seed-2-0-pro-260215` | 0.400 (n=2) | 0.728 (n=451) | 0.756 (n=347) |
| `llama-4-scout` | 0.450 (n=2) | 0.613 (n=451) | 0.585 (n=347) |
| `gpt-5.4-mini` | 0.500 (n=2) | 0.580 (n=451) | 0.607 (n=347) |
| `qwen3.6-plus` | 0.900 (n=2) | 0.554 (n=451) | 0.534 (n=347) |
| `gpt-4o` | 0.400 (n=2) | 0.478 (n=451) | 0.487 (n=347) |
| `glm-5.1` | 0.000 (n=2) | 0.347 (n=451) | 0.287 (n=347) |
| `kimi-k2.5` | 0.850 (n=2) | 0.324 (n=451) | 0.265 (n=347) |

---
## 8. Hallucination severity by question type (HS_w_micro)
Slice from `halu_summary.json.aggregate.per_question_type`. Lower = fewer hallucinations on that type.

| Model | claim_choice | essay | boolean_support | two_hop_tail | experiment_code |
|---|---|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 0.290 | 0.318 | 0.297 | 0.305 | 0.174 |
| `llama-4-scout` | 0.244 | 0.326 | 0.343 | 0.305 | 0.290 |
| `gpt-5.4-mini` | 0.247 | 0.258 | 0.437 | 0.301 | 0.168 |
| `qwen3.6-plus` | 0.276 | 0.288 | 0.202 | 0.294 | 0.242 |
| `gpt-4o` | 0.251 | 0.288 | 0.510 | 0.299 | 0.136 |
| `glm-5.1` | 0.270 | 0.300 | 0.226 | 0.265 | 0.123 |

---
## 9. Setup
* **Agent workflow**: 4 retrieval tools — `web_search`, `web_fetch` (Bright Data SERP), `literature_search`, `literature_fetch` (sciverse / PubMed). ToolUniverse MCP disabled.
* **Reflect on tool use**: disabled globally (thinking models emit empty content alongside tool_calls; autogen reflection step would error).
* **Per-sample timeout**: 600 s.
* **Parallelism**: 2 concurrent samples per model.
* **Per-model endpoint routing**:
  * `intern-s1-pro` → `https://chat.intern-ai.org.cn/api/v1` (different vendor + key)
  * `doubao-*` / `llama-4-scout` / `grok-*` → `<boyue-https-base-url>` (more stable HTTPS gateway via local proxy)
  * Other models → `<boyue-http-base-url>` (direct connect, default Boyue gateway)
* **Thinking-model patch** (`evaluation/thinking_model_patch.py`): registers a custom autogen transformer for `deepseek-v4`, `glm-5`, `kimi-k2`, `intern-s1`, `bailian/deepseek-v4` so multi-turn requests preserve `reasoning_content`. Also lifts `reasoning_content` into `content` on tool-call responses (autogen otherwise drops it).

---
## Metric definitions
### Accuracy
* **`acc`**: unweighted mean of `is_correct` over all samples. For multi-choice = exact letter match; for `essay` = LLM judge score (gpt-4o on Boyue, [0,1]); for `experiment_code` = unit-test pass rate from sandbox execution.
* **`weighted_acc`**: `Σ w_i · c_i / Σ w_i` where `w_i` is the bench-generator-assigned sample weight (higher for high-tier graph-anchored samples with stronger evidence).

### Hallucination
Per claim, the judge picks one verdict; the score map ρ is fixed in code (NOT from the LLM):

| Verdict | Score ρ | Meaning |
|---|---|---|
| `supported` | 0.0 | Evidence explicitly implies the claim |
| `unverifiable` | 0.5 | Evidence silent / tangential |
| `refuted` | 1.0 | Evidence contradicts |

Per-sample metrics (denote refuted/unverifiable/supported counts as `r,u,s` with `r+u+s = |C|`):

* **`HR_sample` (Hallucination Rate)** = `r / |C|` — fraction of claims actively contradicted.
* **`HS_sample` (Hallucination Severity)** = `(0·s + 0.5·u + 1·r) / |C|` — average severity, including the unverifiable middle ground.
* **`HS_weighted_sample`** = `Σ w_g(c)·ρ(v) / Σ w_g(c)` where `w_g(c) = 1 + log(1 + degree(c))` if `c` matches a graph node (BGE cos ≥ 0.6), else 1. Emphasises hub-concept errors.
* **`HF_sample` (Hallucination Flag)** = 1 if any claim was refuted, else 0.

Aggregations across N samples:

* **`*_macro`**: mean of per-sample value — each sample contributes equally regardless of how many claims it produced.
* **`*_micro`**: weighted by claim count — `Σ_i numerator_i / Σ_i denominator_i`.
* **`HF_rate`**: mean of `HF_sample` — fraction of samples containing at least one refuted claim.
* **`n_refuted`**: integer total of refuted claims across all samples.

---
## Reproduce
```bash
# evaluation
bash evaluation/scripts/run_full_tool_models.sh A     # tool-mode 7 base models
bash evaluation/scripts/run_thinking_models.sh        # glm-5.1, kimi-k2.5, deepseek-v4-flash
bash evaluation/scripts/run_bench_a_remaining.sh      # gemini, qwen, doubao, llama, grok
# halu phase 2
bash evaluation/scripts/run_halu_pending.sh A         # auto-skip already-done models
```
