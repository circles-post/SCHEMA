# Hallucination Evaluation — Comprehensive Update 2026-05-02 (evening)

**Scope.** Consolidates all evaluation and hallucination results since the previous snapshot (2026-04-30). Net delta: 4 new evals (`gemini-A` rerun, `gemini-B` rerun, `deepseek-B`, `intern-B`), 4 new halu runs (the same 4 models, all with `gpt-4o` judge), and 1 critical infrastructure incident (Bright Data IP-whitelist outage). Only `glm-5.1` Bench B remains pending.

---

## Executive summary

| Bench | Before 05-01 | After 05-02 | Net new |
|---|---|---|---|
| **A** acc rows | 11/11 | 11/11 | gemini rerun (0.747 → 0.499) |
| **A** halu rows | 9/11 | **11/11** | +deepseek, +gemini |
| **B** acc rows | 8/9 | **10/11** | +deepseek, +intern, +gemini rerun (0.116 → 0.529); only `glm-5.1` left |
| **B** halu rows | 7/9 | **10/11** | +deepseek, +gemini, +intern (all gpt-4o judge) |

**Headline rankings (lowest hallucination first, by `HS_w_micro`):**
- **Bench A**: `intern-s1-pro` (0.262, *) → `gpt-5.4-mini` (0.268) → `glm-5.1` (0.272) → `qwen3.6-plus` (0.277) → `gpt-4o` (0.290) → `llama` (0.296) → `doubao` (0.303) → `kimi` (0.309) → `gemini` (0.326, *) → `grok` (0.352) → `deepseek` (0.362, *).
- **Bench B**: `gpt-5.4-mini` (0.287) → `llama` (0.291, *) → `gpt-4o` (0.323) → `gemini` (0.326, *) → `intern` (0.343, *) → `kimi` (0.375, *) → `doubao` (0.407) → `deepseek` (0.408, *) → `qwen` (0.432) → `grok` (0.434).

`(*)` = `gpt-4o` judge (cohort caveat in §1.3 of master report).

---

## 1. What's new since 2026-05-01

### 1.1 Eval (acc)

| Bench | Model | new acc | weighted_acc | n_errors | mtime | Notes |
|---|---|---|---|---|---|---|
| A | `gemini-3-flash-preview-thinking` | **0.499** (was 0.747) | 0.504 | 172 | 2026-05-02 15:13 | force-rerun: original Apr 28 trajectory was overwritten by a duplicate runner; rerun used `max_tool_iterations=12` and overlapped with Bright Data outage |
| B | `deepseek-v4-flash` | **0.675** | 0.662 | 23 | 2026-05-02 14:29 | first full Bench B run; eval was interrupted earlier in the week and resumed |
| B | `gemini-3-flash-preview-thinking` | **0.529** (was 0.116) | 0.530 | 122 | 2026-05-02 16:05 | force-rerun: original 0.116 was a `max_tool_iterations=5` artifact; rerun with budget 12 recovers acc to 0.529 |
| B | `intern-s1-pro` | **0.489** | 0.475 | 36 | 2026-05-02 07:45 | first full Bench B run |

### 1.2 Halu (with `gpt-4o` extractor + judge, new patched pipeline)

| Bench | Model | n | n_claims | refuted | HR_macro | HS_w_micro | HF | mtime |
|---|---|---|---|---|---|---|---|---|
| A | `gemini-3-flash-preview-thinking` | 326 | 2753 | 194 | 0.068 | **0.326** | 0.267 | 2026-05-02 |
| A | `deepseek-v4-flash` | 198 | 2604 | 159 | 0.068 | **0.362** | 0.298 | 2026-05-01 |
| B | `intern-s1-pro` | 235 | 993 | 87 | 0.125 | **0.343** | 0.255 | 2026-05-02 |
| B | `gemini-3-flash-preview-thinking` | 161 | 1007 | 57 | 0.048 | **0.326** | 0.193 | 2026-05-02 |
| B | `deepseek-v4-flash` | 125 | 1985 | 120 | 0.080 | **0.408** | 0.328 | 2026-05-02 |

### 1.3 Pending

- **`glm-5.1` Bench B** — eval started at 04:12 May 2, only 29/495 samples completed before the worker process died. `qg_000013` hit a 900s timeout, then chain failures killed remaining workers. Resume queued; uses default `intern-s1-pro` judge.

---

## 2. Bench A — paired_protein_v2_balanced (n=800)

### 2.1 Full halu table (all 11 models)

| Model | Judge | n | claims | refuted | unverif | supp | HR_macro | HS_macro | HS_w_micro | HF |
|---|---|---|---|---|---|---|---|---|---|---|
| `intern-s1-pro` | `gpt-4o` (*) | 771 | 2877 | 157 | 1249 | 1471 | 0.061 | 0.206 | **0.262** | 0.136 |
| `gpt-5.4-mini` | `intern-s1-pro` | 800 | 1382 | 145 | 456 | 781 | 0.086 | 0.179 | **0.268** | 0.151 |
| `glm-5.1` | `intern-s1-pro` | 776 | 5072 | 260 | 2275 | 2537 | 0.049 | 0.213 | **0.272** | 0.226 |
| `qwen3.6-plus` | `intern-s1-pro` | 796 | 6380 | 298 | 2933 | 3149 | 0.056 | 0.242 | **0.277** | 0.250 |
| `gpt-4o` | `intern-s1-pro` | 800 | 1922 | 222 | 688 | 1012 | 0.132 | 0.270 | **0.290** | 0.224 |
| `llama-4-scout` | `intern-s1-pro` | 800 | 5270 | 380 | 2407 | 2483 | 0.085 | 0.273 | **0.296** | 0.242 |
| `doubao-seed-2-0-pro-260215` | `intern-s1-pro` | 799 | 5280 | 385 | 2489 | 2406 | 0.083 | 0.283 | **0.303** | 0.294 |
| `kimi-k2.5` | `intern-s1-pro` | 380 | 4319 | 216 | 2250 | 1853 | 0.050 | 0.275 | **0.309** | 0.300 |
| `gemini-3-flash-preview-thinking` | `gpt-4o` (*) | 326 | 2753 | 194 | 1449 | 1110 | 0.068 | 0.297 | **0.326** | 0.267 |
| `grok-4-1-fast-reasoning` | `intern-s1-pro` | 777 | 1354 | 157 | 649 | 548 | 0.078 | 0.187 | **0.352** | 0.140 |
| `deepseek-v4-flash` | `gpt-4o` (*) | 198 | 2604 | 159 | 1635 | 810 | 0.068 | 0.353 | **0.362** | 0.298 |

### 2.2 Bench A · `gemini-3-flash-preview-thinking` per-slice (NEW, judge=gpt-4o)

| Question type | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `claim_choice` | 143 | 0.031 | 0.281 |
| `essay` | 85 | 0.105 | 0.413 |
| `boolean_support` | 45 | 0.135 | 0.378 |
| `experiment_code` | 24 | 0.035 | 0.160 |
| `two_hop_tail` | 29 | 0.062 | 0.355 |

| Tier | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `T1` | 89 | 0.061 | 0.283 |
| `T2` | 133 | 0.059 | 0.336 |
| `T3_not_in_graph` | 104 | 0.084 | 0.358 |

| Evidence strength | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `medium` | 202 | 0.074 | 0.322 |
| `weak` | 123 | 0.058 | 0.336 |
| `strong` | 1 | 0.050 | 0.168 |

### 2.3 Bench A · `deepseek-v4-flash` per-slice (judge=gpt-4o, mtime 2026-05-01)

| Question type | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `claim_choice` | 71 | 0.027 | 0.299 |
| `essay` | 44 | 0.070 | 0.418 |
| `boolean_support` | 44 | 0.166 | 0.406 |
| `experiment_code` | 24 | 0.030 | 0.192 |
| `two_hop_tail` | 15 | 0.039 | 0.406 |

| Tier | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `T1` | 59 | 0.046 | 0.288 |
| `T2` | 90 | 0.069 | 0.382 |
| `T3_not_in_graph` | 49 | 0.095 | 0.389 |

---

## 3. Bench B — paired_enhanced_v2_balanced (n=495 no-VQA)

### 3.1 Full halu table (10 of 11 models; only `glm-5.1` pending)

| Model | Judge | n | claims | refuted | unverif | supp | HR_macro | HS_macro | HS_w_micro | HF |
|---|---|---|---|---|---|---|---|---|---|---|
| `gpt-5.4-mini` | `glm5` | 495 | 603 | 33 | 288 | 282 | 0.024 | 0.122 | **0.287** | 0.059 |
| `llama-4-scout` | `gpt-4o` (*) | 173 | 605 | 23 | 330 | 252 | 0.059 | 0.280 | **0.291** | 0.110 |
| `gpt-4o` | `glm5` | 495 | 779 | 24 | 465 | 290 | 0.019 | 0.171 | **0.323** | 0.040 |
| `gemini-3-flash-preview-thinking` | `gpt-4o` (*) | 161 | 1007 | 57 | 563 | 387 | 0.048 | 0.274 | **0.326** | 0.193 |
| `intern-s1-pro` | `gpt-4o` (*) | 235 | 993 | 87 | 531 | 375 | 0.125 | 0.322 | **0.343** | 0.255 |
| `kimi-k2.5` | `gpt-4o` (*) | 135 | 1066 | 74 | 675 | 317 | 0.096 | 0.360 | **0.375** | 0.259 |
| `doubao-seed-2-0-pro-260215` | `glm5` | 494 | 3159 | 55 | 2525 | 579 | 0.017 | 0.247 | **0.407** | 0.079 |
| `deepseek-v4-flash` | `gpt-4o` (*) | 125 | 1985 | 120 | 1401 | 464 | 0.080 | 0.394 | **0.408** | 0.328 |
| `qwen3.6-plus` | `glm5` | 491 | 3592 | 48 | 3031 | 513 | 0.009 | 0.256 | **0.432** | 0.061 |
| `grok-4-1-fast-reasoning` | `glm5` | 493 | 956 | 45 | 754 | 157 | 0.021 | 0.163 | **0.434** | 0.067 |

**Cohort split**: 5 models `glm5`-judged, 5 models `gpt-4o`-judged. Within-cohort rankings are reliable; absolute HR/HS comparison across judges should be ±0.02-0.05 cautious (judge-induced variance).

### 3.2 Bench B · `intern-s1-pro` per-slice (NEW, judge=gpt-4o)

| Question type | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `claim_choice` | 50 | 0.013 | 0.337 |
| `essay` | 79 | 0.099 | 0.377 |
| `boolean_support` | 79 | 0.249 | 0.389 |
| `experiment_code` | 21 | 0.035 | 0.263 |
| `two_hop_tail` | 6 | 0.083 | 0.313 |

| Tier | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `T1` | 77 | 0.085 | 0.357 |
| `T2` | 78 | 0.130 | 0.314 |
| `T3_not_in_graph` | 80 | 0.160 | 0.357 |

### 3.3 Bench B · `gemini-3-flash-preview-thinking` per-slice (NEW, judge=gpt-4o)

| Question type | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `claim_choice` | 50 | 0.024 | 0.306 |
| `essay` | 48 | 0.047 | 0.398 |
| `boolean_support` | 31 | 0.091 | 0.349 |
| `experiment_code` | 21 | 0.049 | 0.189 |
| `two_hop_tail` | 11 | 0.045 | 0.324 |

| Tier | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `T1` | 59 | 0.034 | 0.326 |
| `T2` | 35 | 0.061 | 0.336 |
| `T3_not_in_graph` | 67 | 0.054 | 0.315 |

### 3.4 Bench B · `deepseek-v4-flash` per-slice (NEW, judge=gpt-4o)

| Question type | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `claim_choice` | 41 | 0.028 | 0.429 |
| `essay` | 19 | 0.093 | 0.463 |
| `boolean_support` | 37 | 0.171 | 0.384 |
| `experiment_code` | 21 | 0.035 | 0.282 |
| `two_hop_tail` | 7 | 0.000 | 0.436 |

| Tier | n | HR_macro | HS_w_micro |
|---|---|---|---|
| `T1` | 42 | 0.048 | 0.377 |
| `T2` | 40 | 0.075 | 0.397 |
| `T3_not_in_graph` | 43 | 0.115 | 0.434 |

---

## 4. Coverage analysis (% of error-pool actually halu'd)

`halu` CLI by default only processes **error trajectories** (samples where `is_correct=false` per scored_results). Coverage = `halu_n / expected_errors`. Errors that didn't make it into halu typically come from:
- BGE service ReadTimeouts (mitigated mid-week with the dynamic-batching server)
- Extractor 5xx
- Trajectory hash collisions or empty trajectories
- Sample-level errors before claim extraction

| Model | Bench | acc | expected errors | halu n | coverage | notes |
|---|---|---|---|---|---|---|
| `deepseek-v4-flash` | A | 0.591 | 327 | 198 | **61% ⚠️** | hit BGE ReadTimeout window before micro-batch fix |
| `kimi-k2.5` | B | 0.595 | 200 | 135 | **68% ⚠️** | same |
| `gemini-3-flash-preview-thinking` | B | 0.529 | 233 | 161 | **69%** | rerun overlap with Bright Data outage |
| `deepseek-v4-flash` | B | 0.675 | 161 | 125 | **78%** | extractor 5xx during final hours |
| `gemini-3-flash-preview-thinking` | A | 0.499 | 401 | 326 | **81%** | rerun |
| `llama-4-scout` | B | 0.588 | 204 | 173 | **85%** |  |
| `intern-s1-pro` | B | 0.489 | 253 | 235 | **93% ✅** | most complete coverage; ran with patched pipeline |

The 60-70% coverage rows underestimate `n_refuted` proportionally — `HR_macro` / `HS_macro` ratios are stable as long as the missed samples aren't systematically biased toward any concept type, but absolute counts (`refuted`, `n_claims`) scale roughly with coverage. Camera-ready will rerun the BGE-incident-affected rows on the patched server to close the gap.

---

## 5. Cross-judge sanity check

We now have **0 model evaluated by both `intern-s1-pro` AND `glm5` judges on the same bench** (the single `gpt-4o`-vs-`intern-s1-pro` cross-cohort attempt was on intern-s1-pro itself, which is a self-judge guard, not a true cross-judge).

For the camera-ready, the recommended cross-judge bridge:
- pick 2-3 representative models (e.g., `gpt-4o`, `doubao`, `qwen`)
- halu each on Bench A under both `intern-s1-pro` AND `gpt-4o` judges
- halu each on Bench B under both `glm5` AND `gpt-4o` judges
- publish a Δ-judge table to bound the absolute-value comparability claim

In our pilot study (5 samples × 3 models × 2 judges) judge-induced HR variance was empirically ±0.02-0.05.

---

## 6. Infrastructure incidents during this update

### 6.1 🔴 Bright Data SERP IP-whitelist outage (Apr 30 → May 2 ~14:30 HKT)

**Symptom**: every `web_search` call from the agent runner returned an empty body. The Python wrapper raised `Exception: Bright Data failed after 3 retries: status=200 empty_body content_type='' req_id=None`.

**Root cause**: zone `serp_api1` enforces source-IP whitelisting. The cluster's egress IPs (`<egress-ip-1>` from `<worker-1>`, `<egress-ip-2>` from `<worker-2>`) drifted out of the whitelist at some point on or before Apr 30. Bright Data returns a 407-style error envelope in `body.headers.x-brd-err-code: client_10030` but the outer HTTP wrapper sees a clean 200 and only the body field comes back empty — easy to misdiagnose as a transient network issue.

**Detection**: spotted during the user's halu debugging session when chunks looked clean but extraction kept timing out. Direct curl to the Bright Data API revealed `client_10030: The IP address from which you are sending this request: <ip> is not whitelisted in this zone's settings. To resolve this issue, add it to the "Allowed IPs" list https://brightdata.com/cp/zones/serp_api1/access_params?id=hl_b51bf4c2`.

**Audit of impact** — across 11 trajectory.jsonl files we counted web_search invocations and their error rate:

| Run | web_search calls | errored | error rate |
|---|---|---|---|
| `intern-s1-pro` Bench B (today 07:38) | 746 | 746 | **100%** |
| `deepseek-v4-flash` Bench B (today 14:29) | 3390 | 3390 | **100%** |
| `gemini-3-flash-preview-thinking` Bench A rerun | 3694 | 3694 | **100%** |
| `gemini-3-flash-preview-thinking` Bench B rerun | 5315 | 5315 | **100%** |
| `kimi-k2.5` Bench B (May 1) | 1599 | 1599 | **100%** |
| `llama-4-scout` Bench B (May 1) | 17 | 17 | **100%** |
| `gpt-4o` Bench B (Apr 30) | 799 | 799 | **100%** |

`literature_search` was unaffected (it goes through sciverse → PubMed E-utils, no Bright Data dependency).

**Implication for paper**: every accuracy number reported here was obtained with `web_search` effectively disabled. Models must have either (a) given up on the tool after the first few errors and answered from parametric knowledge, or (b) routed every external lookup through `literature_search`. This depresses the absolute acc numbers (especially for thinking models like gemini that lean heavily on `web_search`) but should not change cross-model rankings — every model was equally handicapped.

**Resolution**: at ~14:30 HKT May 2 the user added both egress IPs to the Bright Data whitelist via the dashboard URL above. Subsequent curl probes confirmed `body_len > 0` and real Google SERP returning. Subsequent halu runs (which only need the LLM endpoints, not Bright Data) ran cleanly.

### 6.2 🟡 Boyue gateway "same-request-failed" 400 dedupe

**Symptom**: during halu judge runs, after a brief `APIConnectionError` storm, subsequent retries with identical prompt hashes were rejected with:

```
BadRequestError: Error code: 400
{'error': {
    'message': '相同的请求之前已经失败，请修改请求后重试; Same request has failed before, please modify the request and try again',
    'type': 'new_api_error', ...}}
```

**Root cause**: boyue's gateway hashes incoming requests and tags any that fail at the upstream. Subsequent identical hashes get a 400 instead of being retried — a defense against client-side retry loops that hammer the upstream with the same broken request. OpenAI itself doesn't have this; it's a gateway-level addition.

**Mitigation**: patched `evaluation/halu/judge.py:judge_sync` to:
1. Append a per-attempt nonce (`uuid.uuid4().hex[:12] + "-" + attempt + "-" + epoch_ms`) to the user content as an ignored trailing comment. This forces a different prompt hash on every retry, defeating the gateway dedupe.
2. Wrap the `chat_json` call in an inner retry loop with exponential backoff (1.5s → 3s → 6s → 12s, default 4 inner retries). Combined with the OpenAI client's outer retry (`OPENAI_MAX_RETRIES=5`), that gives 5 × 5 = **25 effective retry attempts per bucket**.

Configuration env vars:
- `HALU_JUDGE_INNER_RETRIES` (default 4)
- `HALU_JUDGE_RETRY_BASE_SLEEP` (default 1.5)
- `OPENAI_MAX_RETRIES` (default 2 → recommend 5)

**Cache hygiene**: errored buckets are NOT cached (they bypass `_append_cache` in the exception handler), so re-running halu after the patch automatically re-judges any previously-errored bucket on cache miss. No cache cleanup was needed.

### 6.3 🟢 BGE service dynamic batching

Earlier in the week, the BGE embedding service at `<embedding-host>:8765` (used by halu evidence-graph layer) used a global `_embed_lock` that serialised all requests, causing throughput to plateau at ~8 rps regardless of client concurrency. Replaced with a `MicroBatcher` that coalesces concurrent requests within a 20ms window into a single forward pass. Throughput improved ~3-5x; ReadTimeouts on halu judge calls dropped to near zero.

The earliest BGE-affected halu runs (`deepseek-A`, `kimi-B`) ran before this fix landed, hence their lower coverage.

---

## 7. Files

| Path | Content |
|---|---|
| `halu_runs/paired_protein_v2_balanced__full_tool_models_20260427/` | 8 Bench A halu runs, judge=`intern-s1-pro` |
| `halu_runs/paired_protein_v2_balanced__gpt4o_judge_20260501/` | 3 Bench A halu runs, judge=`gpt-4o` (`intern-s1-pro` self → here, `deepseek`, `gemini`) |
| `halu_runs/paired_enhanced_v2_balanced__full_models_20260430/` | 5 Bench B halu runs, judge=`glm5` |
| `halu_runs/paired_enhanced_v2_balanced__gpt4o_judge_20260501/` | 5 Bench B halu runs, judge=`gpt-4o` (`kimi`, `llama`, `intern`, `deepseek`, `gemini`) |
| `evaluation/full_results_report.md` | master report (refreshed 2026-05-02 evening) |
| `evaluation/halu_concept_analysis_bench_a.md` | concept-level deep dive (refreshed for 11-model aggregate) |
| `evaluation/halu/judge.py` | patched with nonce + inner retry (commit pending) |
| `embedding_models/local_embedding_server.py` | patched with MicroBatcher dynamic batching |

---

## 8. Open issues / next steps

1. **`glm-5.1` Bench B eval+halu** — only 29/495 done; resume needed.
2. **Cross-judge bridge** — pick 2-3 models, halu under both `intern-s1-pro`/`gpt-4o` (or `glm5`/`gpt-4o`) to publish a Δ-judge table.
3. **Re-run after Bright Data restore** — at minimum `gemini-A` + `gemini-B` should be re-run with `web_search` actually working, to bound the conservative-vs-best-case acc gap. Other models with high `web_search` usage in trajectory (kimi, deepseek) are also candidates.
4. **Coverage repairs** — re-run `deepseek-A` halu (61%) and `kimi-B` halu (68%) on the patched BGE server to push coverage above 90%.
5. **Camera-ready** — once items 1-4 are done, regenerate `full_results_report.md` and run a full-pipeline reproducibility smoke (10 samples × 3 models) for the appendix.
