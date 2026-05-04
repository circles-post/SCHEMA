# Bench B Hallucination — Concept-Level Deep Analysis

_Aggregated across **10 of 11 models** on Bench B balanced (n=495 no-VQA), **566 refuted claim instances** total. Halu judges: `glm5` for 5 models (`doubao`, `gpt-4o`, `gpt-5.4-mini`, `grok`, `qwen`), `gpt-4o` for 5 models (`kimi-k2.5`, `llama-4-scout`, `intern-s1-pro`, `deepseek-v4-flash`, `gemini-3-flash-preview-thinking`). `glm-5.1` Bench B halu pending. Sister analysis to `halu_concept_analysis_bench_a.md`. Last refresh: 2026-05-03._

> **Critical caveat upfront**: Unlike Bench A where 11 models share a single judge (intern-s1-pro for 8/11), Bench B is split 5/5 between two judges. The judges have **systematically different aggressiveness** (§5 below). This means concept-level "universal hallucinations" don't replicate cleanly across the two cohorts the way they do on Bench A — the data structure here is fundamentally noisier as a single-bench analysis.

---

## 1. Executive comparison vs Bench A

| Metric | Bench A (n=800) | Bench B (n=495) | ratio |
|---|---|---|---|
| Models analysed | 11 | 10 (glm-5.1 pending) | — |
| Total refuted instances | 2573 | 566 | 4.5× lower |
| Unique concepts hit | 866 | 250 | 3.5× fewer |
| Max-model-overlap concept | 10/11 (`ptpn22`) | 6/10 (`hiden`, `cenpa`) | substantially weaker consensus |
| Long-tail (1-model concepts) | 612 / 866 (70.7%) | 172 / 250 (68.8%) | similar shape |
| Universal threshold (≥7 models) | 34 concepts | 0 concepts | qualitatively different |
| Universal threshold (≥4 models) | ~120 concepts | 10 concepts | 12× fewer |
| Refuted per sample | 3.22 | 1.14 | 2.8× lower density |

Bench B is far less concept-dense in hallucinations than Bench A, partly because:
1. **Smaller sample size** (495 vs 800), naturally fewer total claims
2. **Different question distribution** — Bench B has more `claim_choice` (40%) which produces shorter trajectories than Bench A's `essay`-heavy setup
3. **Judge mixture** — `glm5` judges 5 models conservatively (low refuted rate); `gpt-4o` judges 5 models aggressively (high refuted rate). Bench A by contrast uses `intern-s1-pro` for 8 of 11 (more uniform).

---

## 2. Concept-type distribution of hallucinations

Aggregating refuted claims across all 10 models. Note: absolute counts; not normalised by total claims of that type.

| Concept type | refuted insts | % of total | observation |
|---|---|---|---|
| **Protein** | 132 | 23.3% | Top category, same as Bench A. Mostly from gpt-4o-judged models (gemini, intern, deepseek, kimi each contribute 21-26). |
| **Drug** | 95 | 16.8% | Lower share than Bench A (22.5%) — Bench B questions are more biology-focused. |
| **Gene** | 75 | 13.3% | Substantially higher share than Bench A (6.1%) — Bench B has more gene-name questions in `claim_choice` format. |
| **CellLine** | 58 | 10.2% | Lower than Bench A (17.7%) — fewer mechanism-heavy questions. |
| **MolecularEntity** | 49 | 8.7% | Slightly lower than Bench A (14.0%). |
| **RNA** | 33 | 5.8% | Higher than Bench A (1.6%) — driven by `hiden` toxic sample (19 refuted instances, see §4). |
| **Disease** | 28 | 4.9% | Higher than Bench A (0.9%) — Bench B includes some clinical / disease-specific questions. |
| **TissueRegion** | 24 | 4.2% | Bench B specific — emerges from PathVQA-derived questions about tissue locations / staining patterns. |
| **Biomarker** | 14 | 2.5% | Higher than Bench A (0.7%). |
| **Complex** | 13 | 2.3% | Lower than Bench A (5.0%). |
| **ClinicalEndpoint** | 10 | 1.8% | Higher than Bench A (0.6%). |
| **CellType** | 10 | 1.8% | Similar to Bench A. |
| **BiologicalProcess** | 9 | 1.6% | Substantially lower than Bench A (5.1%). |
| **Pathway** | 3 | 0.5% | Similar to Bench A. |
| **StainingMethod** | 3 | 0.5% | Bench B specific (PathVQA legacy). |
| (untyped) | 10 | 1.8% | |

**Comparison to Bench A**: top-3 categories are stable (Protein, Drug, then either CellLine or Gene), but the **mix shifts toward Gene + Disease + TissueRegion** on Bench B. The pathology-derived questions surface a different concept-type distribution than the protein-mechanism questions of Bench A — consistent with the dataset construction (PathVQA-augmented).

---

## 3. Universal hallucinations: concepts where ≥4 of 10 models all err

There are NO concepts refuted by 7 or more models on Bench B. The strongest universal pattern caps at 6 models:

| Concept | # models | refuted insts | concept type | example refuted claim | Notes |
|---|---|---|---|---|---|
| `hiden` | 6 | 19 | RNA | HIDEN downregulates beta Catenin. | Toxic sample qg_999019; same hiden concept also seen 9-models on Bench A. |
| `cenpa` | 6 | 12 | Protein | CENPA has reduced centromeric localization and expression in systemic sclerosis patient cells. | Toxic sample qg_001253. |
| `cyld` | 5 | 14 | Protein | CYLD expression is decreased in keratinocytes in psoriasis. | Toxic sample qg_000999. |
| `hapln1` | 5 | 8 | Gene | HAPLN1's relationship with TNFAIP6 is not routinely established in literature. | `false_negation` pattern. |
| `neuromodulin-1β` | 5 | 7 | Drug | The evidence results do not mention Neuromodulin-1β. | `false_negation` (drug taxonomy issue). |
| `follistatin` | 4 | 10 | Protein | Follistatin is not directly reported as activating PI3K-Akt-mTOR. | Toxic sample qg_002278. |
| `lhx1a` | 4 | 7 | Gene | lhx1a downregulates gsc. | Direction error. |
| `parenteral nutrition` | 4 | 6 | Drug | PN is used to provide nutrition including amino acids. | Generic textbook fact mismatched with paper context. |
| `prominent hooding` | 4 | 5 | (untyped) | Prominent hooding is associated with prolapse of the posterior mitral leaflet. | Cardiology term, paper uses different definition. |
| `enzalutamide` | 4 | 4 | Drug | Enzalutamide activating CYP3A4 is not directly supported in the cited study. | `false_negation` of metabolic pathway claim. |

**Total ≥4-model concepts: 10** (vs 34 ≥7-model concepts on Bench A — qualitatively much weaker consensus).

### Observation: zero judge-robust universals

A "judge-robust universal" would be a concept refuted by ≥3 `glm5`-judged models AND ≥3 `gpt-4o`-judged models. The intersection on Bench B is **0 concepts**. Every "universal" listed above is either majority-glm5 (low recall) or majority-gpt-4o (high recall but in models with lower coverage). This is the strongest single signal that **Bench B halu numbers should not be aggregated naïvely across the two judge cohorts**.

---

## 4. Toxic samples — questions that break multiple models

### `qg_000481` (essay) — 23 refuted across 5 models, 4 concepts
**About**: rs3865444 C allele effect on CD33 expression.
**Why toxic**: A specific GWAS variant with a published effect direction; models confuse the C vs T allele direction. Concepts hit: `c allele`, `cd33`, `rs3865444`, `rs3865444 c allele`. All 5 models that attempted it gave the wrong directional account.

### `qg_999019` (experiment_code) — 19 refuted across 5 models, 3 concepts
**About**: HIDEN regulation of Wnt pathway activity.
**Why toxic**: Same HIDEN universal-misconception that broke 9 models on Bench A, but attached to a different (Wnt-axis) question. Models default to "downregulates" when paper specifies upregulates or context-dependent.

### `qg_002290` (essay) — 16 refuted across 4 models, 2 concepts
**About**: KHDC1 as negative regulator of cysteine endopeptidases (cathepsins).
**Why toxic**: Multi-claim chains about KHDC1's regulatory mechanism; models substitute the cathepsin family details and over-extrapolate the negative-regulator claim.

### `qg_000999` (boolean_support) — 14 refuted across 5 models, 1 concept
**About**: CYLD expression in psoriatic keratinocytes.
**Why toxic**: 5 models all repeat the same wrong directional claim ("CYLD expression is decreased"). The cited paper actually shows increased / context-dependent expression. Concentration of 14 refuted instances on a single concept (CYLD) is distinctive — most B-bench toxics spread across multiple concepts.

### `qg_001978` (essay) — 13 refuted across 2 models, 1 concept
**About**: VEGFC-C152S as dominant-negative VEGFC mutant.
**Why interesting**: Only 2 models triggered (both verbose: `intern-s1-pro` and `deepseek-v4-flash`). Demonstrates the verbose-model claim-density confound — a single sample × 2 models can rack up 13 refuted instances when the models elaborate mechanism details.

### `qg_001253` (essay) — 12 refuted across 6 models, 1 concept
**About**: CENPA centromeric localization in systemic sclerosis.
**Why toxic**: All 6 models give the textbook "reduced expression in disease" account; the cited paper has a more nuanced finding. Strong example of "training-data prior overrides paper-specific evidence" — same pattern as Bench A's universals but at smaller scale.

### `qg_000948` (essay) — 11 refuted across 4 models, 4 concepts
**About**: Syntaxin 17 ↔ SERCA2 calcium ATPase interaction.
**Why toxic**: Models hedge with "available data do not establish" / `false_negation` when the paper actually does establish the interaction.

### `qg_001343` (boolean_support) — 10 refuted across 4 models, 4 concepts
**About**: Glutamate-induced calcium signalling in pericytes.
**Why toxic**: Models invoke generic "glutamate increases calcium in most cells" priors; paper shows pericyte-specific exception.

### `qg_002278` (boolean_support) — 10 refuted across 4 models, 1 concept
**About**: Follistatin's role in PI3K-Akt-mTOR signalling.
**Why toxic**: All 4 models say "not directly reported" — but the paper does report indirect activation. Classic `false_negation`.

### `qg_000109` (essay) — 10 refuted across 4 models, 4 concepts
**About**: Age above 10 years and chronic otitis media correlation.
**Why toxic**: Models hedge against the specific clinical correlation, calling it `false_negation` ("no evidence of specific interaction").

### `qg_001994` (essay) — 10 refuted across 3 models, 1 concept
**About**: Resolvin E1 mechanism via ChemR23.
**Why interesting**: 3 models hallucinate the same mechanism direction (E1 blocks neutrophil apoptosis via ChemR23) — actually the paper says a different mechanism.

### `qg_001678` (essay) — 9 refuted across 4 models, 2 concepts
**About**: EGR1 ↔ GNAT domain physical interaction.
**Why toxic**: Models claim "no direct physical interaction" → `false_negation`.

---

## 5. The judge-induced split: glm5 vs gpt-4o on Bench B

This is the **single biggest paper caveat** for Bench B halu numbers.

### 5.1 Cohort-level summary

| Cohort | Models | Total refuted | Avg refuted per model | n_halu (avg) |
|---|---|---|---|---|
| `glm5` judge | 5 (`doubao`, `gpt-4o`, `gpt-5.4-mini`, `grok`, `qwen`) | 205 | 41 | 494 |
| `gpt-4o` judge | 5 (`kimi-k2.5`, `llama-4-scout`, `intern-s1-pro`, `deepseek-v4-flash`, `gemini-3-flash-preview-thinking`) | 361 | 72 | 166 |

The `gpt-4o`-judged cohort produces **75% more refuted claims per model**, despite having ~70% smaller halu coverage (n_halu averages 166 vs 494). Per-sample ratios:

| Cohort | refuted per sample (n_halu denominator) |
|---|---|
| `glm5` cohort | 0.083 |
| `gpt-4o` cohort | 0.434 |

That's **5.2× more refuted-per-sample under the `gpt-4o` judge**.

### 5.2 HF_rate split is even more dramatic

`HF_rate` = fraction of samples with at least one refuted claim. Aggregate:

| Cohort | HF_rate range |
|---|---|
| `glm5`-judged | **0.040 – 0.079** (4–8%) |
| `gpt-4o`-judged | **0.110 – 0.328** (11–33%) |

The `gpt-4o` judge is **3–5× more aggressive** at marking samples as containing refuted claims. The most extreme example: `deepseek-v4-flash` HF=0.328 (gpt-4o judge) vs `qwen3.6-plus` HF=0.061 (glm5 judge), despite similar Bench B accuracy (0.675 vs 0.555).

### 5.3 Implications for paper

1. **Within-cohort rankings remain reliable**: e.g., among gpt-4o-judged models the order `llama (0.291) → gemini (0.326) → intern (0.343) → kimi (0.375) → deepseek (0.408)` reflects real differences.
2. **Cross-cohort comparisons are fraught**: comparing `qwen` (HS_w=0.432, glm5 judge) directly to `kimi` (HS_w=0.375, gpt-4o judge) is misleading — the absolute values aren't on the same scale.
3. **Paper-grade fix**: pick 2-3 representative models, halu under both judges, publish a Δ-judge offset table. Without this, any cross-cohort claim should carry an explicit caveat.

This is qualitatively different from Bench A, where 8 of 11 models share a single judge and the cross-cohort offset is bounded to ±0.05 in our pilot.

---

## 6. Per-model failure profile by concept type

| Model | Protein | Drug | CellLine | MolecularEntity | Gene | Complex | BiologicalProcess | RNA | Disease | TissueRegion | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek-v4-flash` (*) | 23 | 18 | 14 | 16 | 20 | 0 | 2 | 14 | 3 | 4 | **120** |
| `intern-s1-pro` (*) | 26 | 16 | 5 | 10 | 8 | 3 | 1 | 1 | 4 | 3 | **87** |
| `kimi-k2.5` (*) | 21 | 14 | 15 | 5 | 5 | 0 | 1 | 6 | 1 | 4 | **74** |
| `gemini-3-flash-preview-thinking` (*) | 23 | 9 | 5 | 2 | 9 | 3 | 1 | 3 | 0 | 1 | **57** |
| `doubao-seed-2-0-pro-260215` | 17 | 12 | 2 | 3 | 6 | 3 | 0 | 2 | 6 | 0 | **55** |
| `qwen3.6-plus` | 3 | 7 | 2 | 4 | 7 | 2 | 0 | 3 | 4 | 3 | **48** |
| `grok-4-1-fast-reasoning` | 7 | 5 | 8 | 4 | 7 | 1 | 0 | 1 | 2 | 6 | **45** |
| `gpt-5.4-mini` | 8 | 1 | 2 | 2 | 4 | 1 | 1 | 1 | 6 | 2 | **33** |
| `gpt-4o` | 2 | 7 | 1 | 3 | 5 | 0 | 1 | 0 | 1 | 1 | **24** |
| `llama-4-scout` (*) | 2 | 6 | 4 | 0 | 4 | 0 | 2 | 2 | 1 | 0 | **23** |

`(*)` = `gpt-4o` judge.

**Reading the matrix**:
- The top 4 rows (`deepseek`, `intern`, `kimi`, `gemini`) are all `gpt-4o`-judged and dominate the refuted count (338/566 = 60% of all refuted on Bench B). Two of these (`deepseek`, `intern`) are also the **highest absolute counts** of any model on either bench.
- `llama-4-scout` is judged by `gpt-4o` but has the LOWEST count (23) — interesting because llama also has high absolute count on Bench A (380). This suggests llama on Bench B is genuinely making fewer claims, not just being judged differently. Likely tied to llama's terser Bench B response style.
- The 5 `glm5`-judged models cluster in the 24-55 range — the judge cap effect.
- **Concept profile difference vs Bench A**: deepseek's RNA count (14) is unusually high for Bench B due to the `hiden` and similar RNA-modulator questions; intern-s1-pro's protein count (26) is its highest on either bench.

---

## 7. Long-tail: distribution of hallucination across concepts

| # models that refuted concept | # distinct concepts | % | interpretation |
|---|---|---|---|
| 1 model | 172 | 68.8% | Idiosyncratic — slightly higher fraction than Bench A (70.7%) |
| 2-3 models | 68 | 27.2% | Spotty agreement |
| 4-6 models | 10 | 4.0% | Common pitfall |
| 7-8 models | 0 | 0.0% | No strong consensus errors |
| 9-10 models | 0 | 0.0% | No universal hallucinations |

Total unique concepts hit: **250** across 566 refuted instances.

Striking fact: **77% of distinct hallucinated concepts are 1- or 2-model concepts**, vs Bench A's 75%. Despite the smaller dataset, the long-tail is similarly heavy-tailed — consistent with the "judge cohort masks signal" hypothesis (the 5/5 split fragments any agreement that would surface with a uniform judge).

---

## 8. Cross-bench concept overlap

Concepts that hallucinated on BOTH benches:

| Concept | A models | B models | A-instances | B-instances | Note |
|---|---|---|---|---|---|
| `hiden` | 9 | 6 | 22 | 19 | The clearest cross-bench universal. Same RNA-pathway misconception (knockout effect direction) crops up in both contexts. |
| `cyld` | 4 | 5 | 7 | 14 | Tumor suppressor / inflammation contradictions. |
| `cenpa` | 3 | 6 | 6 | 12 | Centromere protein expression-direction errors. |

Overlapping concepts are a small set (≈3-5 strong cases). This corroborates that the two benches sample fairly different concept-spaces — the toxic-question phenomenon is mostly bench-specific, not cross-portable.

---

## 9. Failure pattern taxonomy (Bench B)

Re-aggregated for the 566-claim corpus:

| Pattern | Count | % | Bench A comparison |
|---|---|---|---|
| `false_negation` | 256 | 45.2% | A=40.5% — slightly higher on B (more "evidence does not mention" hedging on PathVQA-derived questions) |
| `wrong_direction` | 64 | 11.3% | A=12.9% — similar |
| `fabricated_target` | 58 | 10.2% | A=9.8% — similar |
| `wrong_mechanism` | 41 | 7.2% | A=7.7% — similar |
| `over_extrapolation` | 32 | 5.7% | A=5.6% — similar |
| `paper_misattribution` | 18 | 3.2% | A=3.4% — similar |
| other / mixed | 97 | 17.1% | A=20.1% |

**Falsifiable-pattern share** (`false_negation` + `wrong_direction` + `fabricated_target`): **66.7% on Bench B vs 63.2% on Bench A**.

Both benches independently confirm: **about 2/3 of biomedical agent hallucinations are textually-detectable patterns** that don't require domain-knowledge dispute. Robust paper claim — across two different benchmarks, two judge architectures, and 11 different models.

---

## 10. Implications for the paper

### 10.1 Bench B is a noisier instrument for concept-level halu analysis

The judge-cohort split (§5) means cross-model HR/HS comparisons within Bench B carry an irreducible ±0.05+ uncertainty band. For concept-level analysis specifically, the split eliminates the "universal hallucination" structure that we found on Bench A — there are no concepts where ≥7 of 10 models agree on the wrong answer.

**Recommendation**: in the paper's main hallucination analysis, **lean on Bench A for the concept-level deep-dive** (where 8/11 models share a judge), and use Bench B primarily for cross-domain robustness checks at the **per-question-type and per-tier level** (which are less judge-sensitive than concept-level claims).

### 10.2 The cross-bench concept overlap (§8) is a paper-grade finding

`hiden` is universal on Bench A (9/11) and substantially shared on Bench B (6/10) → it represents a true cross-bench training-data prior issue. Using `hiden` and the 2-3 other shared concepts as **cross-bench validation anchors** strengthens the universal-hallucination claim more than any single-bench analysis.

### 10.3 Falsifiable-pattern result replicates across benches

§9 shows ~66% of refuted claims on B and ~63% on A fall into the same 3 textually-detectable patterns. The paper's "lightweight verifier could intercept ~2/3 of hallucinations" claim is confirmed twice independently — by far the strongest empirical finding here.

### 10.4 The PathVQA legacy adds new concept types

Bench B's TissueRegion / StainingMethod / ClinicalEndpoint refuted categories are absent or near-zero on Bench A. This is real signal that **multimodal-derived questions stress models on different concept types** — and motivates a multimodal follow-up.

### 10.5 Verbose-model effect is more pronounced on Bench B

Top 4 models by refuted count on Bench B (`deepseek` 120, `intern` 87, `kimi` 74, `gemini` 57) are all gpt-4o-judged AND verbose. They produce 3-5× more refuted claims per sample than the glm5-judged cohort, even after coverage normalisation. The §9.5 finding from Bench A (HF_rate is more robust than HR_macro for cross-model comparison when verbosity differs) replicates here.

### 10.6 Bench B's gemini results reflect a Bright-Data-broken rerun

As with Bench A, gemini's Bench B halu was run on a force-rerun trajectory where 5315/5315 `web_search` calls failed (Bright Data IP-whitelist outage; details in `halu_results_update_20260502.md` §6.1). Gemini's profile here may over-represent "parametric knowledge fallback" rather than full-tool behavior. A re-run after Bright Data restoration is recommended for camera-ready.

### 10.7 No-glm-5.1-Bench-B blind spot

`glm-5.1` Bench B (eval + halu) is still pending. On Bench A, glm-5.1's hallucination profile is non-trivial (260 refuted, HS_w_micro=0.272). Without the Bench B counterpart we can't bound glm-5.1's cross-bench consistency. Resume queued.
