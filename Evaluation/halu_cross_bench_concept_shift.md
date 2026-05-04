# Cross-Bench Concept Shift Analysis: Bench A ↔ Bench B

_Quantifies the concept-space dissociation between `paired_protein_v2_balanced` (Bench A, 800 samples, 11 models) and `paired_enhanced_v2_balanced` (Bench B, 495 no-VQA samples, 10 models). Last refresh: 2026-05-03._

> **TL;DR** — The two benchmarks test almost entirely disjoint concept spaces (Jaccard IoU = **0.005**, only **5 shared concepts** out of 1111 union). Concept-type distribution shifts systematically toward Gene/RNA/Disease/TissueRegion on B, away from CellLine/Drug/MolecularEntity/BiologicalProcess. Per-model concept-type profile cross-bench correlation ranges from r=0.16 (qwen) to r=0.83 (kimi). Three structural findings replicate across this dissociation: **falsifiable-pattern share (~65%), hub-concept concentration, and verbosity-driven claim-density variation.** Universal hallucinations identified on Bench A do **not** generalize — only `hiden` (RNA pathway) survives as a robust cross-bench universal.

---

## 1. Headline numbers

| Metric | Bench A | Bench B | Comment |
|---|---|---|---|
| Total samples | 800 | 495 | A is 1.6× larger |
| Models with halu | 11 | 10 (`glm-5.1` pending) | — |
| Total refuted instances | 2573 | 566 | A is **4.5×** more dense |
| Unique concepts hit | **866** | **250** | A is **3.5×** more diverse |
| **Intersection size (A ∩ B)** | **5** | **5** | five concepts only |
| **Union size (A ∪ B)** | **1111** | **1111** | — |
| **Jaccard IoU(A, B)** | **0.0045** | (= 0.5%) | near-disjoint |
| Refuted claims per sample | 3.22 | 1.14 | A 2.8× denser |
| Mean halu coverage | 81% | 58% | B has more BGE-incident-affected runs |

The Jaccard IoU **0.0045** is the most striking number in this analysis. By comparison, two random subsets of papers from the same domain typically share 5–15% of named-entity vocabulary. The two benchmarks are operating on near-orthogonal concept lexicons.

---

## 2. Concept-type distribution shift

Comparing the share of refuted instances across each `entity_type`:

| Concept type | A_count | A% | B_count | B% | Δ (B − A) | shift signal |
|---|---|---|---|---|---|---|
| **Protein** | 620 | 24.1% | 132 | 23.3% | -0.8% | stable (the only invariant) |
| **CellLine** | 456 | 17.7% | 58 | 10.2% | **-7.5%** | A-skewed |
| **Drug** | 579 | 22.5% | 95 | 16.8% | -5.7% | A-skewed |
| **MolecularEntity** | 360 | 14.0% | 49 | 8.7% | -5.3% | A-skewed |
| **BiologicalProcess** | 130 | 5.1% | 9 | 1.6% | -3.5% | A-skewed |
| **Complex** | 129 | 5.0% | 13 | 2.3% | -2.7% | A-skewed |
| **Pathway** | 19 | 0.7% | 3 | 0.5% | -0.2% | rare on both |
| **Gene** | 158 | 6.1% | 75 | 13.3% | **+7.1%** | B-skewed |
| **RNA** | 40 | 1.6% | 33 | 5.8% | **+4.3%** | B-skewed |
| **Disease** | 24 | 0.9% | 28 | 4.9% | **+4.0%** | B-skewed |
| **TissueRegion** | 6 | 0.2% | 24 | 4.2% | **+4.0%** | B-emergent (PathVQA legacy) |
| **Biomarker** | 17 | 0.7% | 14 | 2.5% | +1.8% | B-skewed |
| **CellType** | 18 | 0.7% | 10 | 1.8% | +1.1% | mild B-skew |
| **ClinicalEndpoint** | 16 | 0.6% | 10 | 1.8% | +1.1% | mild B-skew |
| **StainingMethod** | 0 | 0.0% | 3 | 0.5% | +0.5% | B-emergent |
| **(unknown)** | 0 | 0.0% | 10 | 1.8% | +1.8% | B-only |
| **Assay** | 1 | 0.0% | 0 | 0.0% | -0.0% | rare |

### Distribution divergence (information-theoretic)

Treating the type distributions as discrete probability distributions:

| Quantity | Value | Interpretation |
|---|---|---|
| KL(A ‖ B) | **0.231** | "expected extra bits to encode A using B's distribution" |
| KL(B ‖ A) | **0.636** | asymmetric — B has B-emergent types (TissueRegion, StainingMethod) absent on A |
| Jensen-Shannon divergence | **0.063** | symmetric distance; modest but non-trivial (random uniform = 0) |

The asymmetric KL (0.636 vs 0.231) confirms B has its own emergent type signature (PathVQA-derived TissueRegion + StainingMethod + Disease + ClinicalEndpoint) that's mostly absent on A.

### Concrete examples per type (top concepts on each bench)

| Type | Bench A top-5 (refuted insts) | Bench B top-5 (refuted insts) |
|---|---|---|
| **Gene** | `ace2` (21), `ubash3b` (15), `tp53` (8), `ubash3a` (7), `taf13` (7) | `khdc1` (15), `hapln1` (8), `lhx1a` (7), `twist2` (4), `clec2d` (4) |
| **RNA** | `hiden` (22), `circdym` (6), `hiden lncrna` (4), `circrps19` (2), `dhav-1 3d polymerase` (2) | **`hiden` (19)**, `nqo1 mrna` (5), `hiden rna` (3), `lncrna edch1` (3), `lnc-mg` (1) |
| **Disease** | `h1n1` (2), `zika virus` (2), `thalidomide` (2), `actinic keratosis` (1), `hip dysplasia` (1) | `spiradenomas` (3), `chronic otitis media` (3), `classical schwannomas` (2), `alcohol dependence` (2), `vitamin c deficiency` (2) |
| **TissueRegion** | `integrin` (1), `colorectal cancer` (1), `colon` (1) | `pro-inflammatory cytokines` (5), `capillary pericytes` (4), `pericyte ca2+` (3), `renal medullary carcinoma` (2), `thymic atrophy` (1) |
| **CellLine** | `sars-cov-2` (72), `ripk3` (29), `usp22` (26), `nse4-hth` (20), `ptpn22` (17) | `vegfc-c152s` (13), `syntaxin 17` (8), `basigin-2` (6), `opa-1` (5), `hexokinase-3` (3) |
| **Drug** | `pcbp2` (29), `chonglouoside sl-8` (29), `sodium` (20), `tm4sf5` (19), `ml283` (15) | `neuromodulin-1β` (7), `lgr5` (7), `parenteral nutrition` (6), `glutamate` (5), `enzalutamide` (4) |
| **MolecularEntity** | `electronegative clusters` (29), `zinc protoporphyrin ix` (20), `c-di-gmp` (19), `thr412` (15), `h-ns` (15) | `rs3865444 c allele` (15), `resolvin e1` (10), `the study` (6), `p38 mapk` (5), `c allele` (3) |
| **BiologicalProcess** | `ezh2` (27), `hnrnpk` (18), `fmrp` (8), `hnrnp e1` (6), `dhx9` (5) | `neuraminidase` (5), `ras` (1), `dementia` (1), `hif-1α` (1), `reactive oxygen species` (1) |

The per-type top-5 lists are **completely disjoint except for `hiden` in RNA**. Even within stable categories like Protein and CellLine, top concepts don't overlap: Bench A is dominated by `sars-cov-2`, `ripk3`, `usp22`; Bench B has `vegfc-c152s`, `syntaxin 17`, `basigin-2`. Different paper pools, different mechanisms, different vocabulary.

---

## 3. Concrete concept overlap

The **complete list of cross-bench shared concepts** (only 5 concepts in `A ∩ B`):

| Concept | A models / 11 | A insts | B models / 10 | B insts | Type | Strength |
|---|---|---|---|---|---|---|
| **`hiden`** | **9/11** | 22 | **6/10** | 19 | RNA | 🟢 Strong (only true cross-bench universal) |
| `stat3` | 7/11 | 13 | 1/10 | 1 | Complex | 🟡 A-strong, B-incidental |
| `nkx6-1` | 2/11 | 2 | 3/10 | 3 | CellLine | ⚪ Both weak |
| `dbet6` | 1/11 | 2 | 1/10 | 2 | Drug | ⚪ Both weak |
| `ns5` | 2/11 | 2 | 1/10 | 1 | CellLine | ⚪ Both weak |

**Interpretation**: Only `hiden` clears the bar of "≥3 models on both benches consistently agree on the wrong answer". The other 4 are statistical accidents — concepts that happen to have a borderline number of refuted claims on one or both benches.

`hiden` is a long-noncoding RNA whose role in Wnt-pathway / β-catenin regulation is widely cited but paper-specifically nuanced. Models default to "downregulates Wnt" / "knockout upregulates pluripotency genes" textbook claims regardless of which bench the question comes from. **Single best paper-grade example of a cross-bench universal hallucination.**

For the Bench A universals (`ptpn22` 10/11, `electronegative clusters` 9/11, `ace2` 9/11, `ripk3` 8/11, `grp78` 8/11, `usp22` 8/11, `ezh2` 8/11, …), **none** have a meaningful Bench B counterpart — they're all bench-specific.

---

## 4. Per-model concept-type profile correlation

For each model, we compute the Pearson correlation between its Bench-A and Bench-B refuted-claim counts across the 8 main concept types (Protein, Drug, CellLine, MolecularEntity, Gene, Complex, BiologicalProcess, RNA). High r means "this model has a stable concept-type 'profile' regardless of which bench it's tested on"; low r means "this model's failure mode shifts substantially with the bench".

| Model | Pearson r | A_total | B_total | Stability |
|---|---|---|---|---|
| `kimi-k2.5` | **0.828** | 216 | 74 | Highest stability |
| `intern-s1-pro` | **0.760** | 157 | 87 | High stability |
| `gemini-3-flash-preview-thinking` | 0.696 | 194 | 57 | Moderate-high |
| `doubao-seed-2-0-pro-260215` | 0.663 | 385 | 55 | Moderate-high |
| `grok-4-1-fast-reasoning` | 0.656 | 157 | 45 | Moderate-high |
| `deepseek-v4-flash` | 0.601 | 159 | 120 | Moderate |
| `gpt-4o` | 0.499 | 222 | 24 | Moderate |
| `llama-4-scout` | 0.445 | 380 | 23 | Moderate-low |
| `gpt-5.4-mini` | **0.262** | 145 | 33 | Low |
| `qwen3.6-plus` | **0.159** | 298 | 48 | Lowest |

### What the spread tells us

- **kimi-k2.5 and intern-s1-pro** have model-intrinsic concept-type biases (heavy on Protein/CellLine/Drug) that are stable regardless of bench composition. These models bring their own "favourite hallucination concept types" with them.
- **qwen3.6-plus and gpt-5.4-mini** are highly bench-sensitive: their concept-type profile on A doesn't predict their profile on B at all (r < 0.3). For these models, "the bench shapes which concept types they fail on" more than "the model has a fixed weakness".
- **Caveat**: `qwen` and `llama` have very small B totals (48 and 23 refuted respectively) — small-sample noise inflates their apparent profile dissimilarity. For paper-grade claims, only models with B_total ≥ 70 (kimi, intern, deepseek, gemini, doubao) should be used to argue stability.

### Verbosity confound

Per-sample claim density on Bench B varies dramatically:

| Model | judge | claims/sample | refuted/sample | profile r vs A |
|---|---|---|---|---|
| `deepseek-v4-flash` | gpt-4o | **15.88** | **0.96** | 0.601 |
| `kimi-k2.5` | gpt-4o | 7.90 | 0.548 | 0.828 |
| `qwen3.6-plus` | glm5 | 7.32 | 0.098 | 0.159 |
| `doubao-seed-2-0-pro-260215` | glm5 | 6.39 | 0.111 | 0.663 |
| `gemini-3-flash-preview-thinking` | gpt-4o | 6.25 | 0.354 | 0.696 |
| `intern-s1-pro` | gpt-4o | 4.23 | 0.370 | 0.760 |
| `llama-4-scout` | gpt-4o | 3.50 | 0.133 | 0.445 |
| `grok-4-1-fast-reasoning` | glm5 | 1.94 | 0.091 | 0.656 |
| `gpt-4o` | glm5 | 1.57 | 0.048 | 0.499 |
| `gpt-5.4-mini` | glm5 | 1.22 | 0.067 | 0.262 |

Spread of claims-per-sample is **13×** (deepseek 15.88 vs gpt-5.4-mini 1.22). This is the hidden engine behind the per-model totals: `deepseek-v4-flash` produces the **most** refuted claims on Bench B (120) despite having only 125 sample halu — its 0.96 refuted/sample beats every other model by a wide margin. Cross-bench profile correlation is driven partly by claim-volume similarity, not just concept-type preference.

---

## 5. Why the shift exists — three causal contributors

### 5.1 Different paper pools (root cause)

- **Bench A** (`paired_protein_v2`) is built on `proteinlmbench_full_graphbench`: a 1412-sample question generation over a curated protein-mechanism literature set (necroptosis, ubiquitination, protein structural biology, drug-target binding).
- **Bench B** (`paired_enhanced_v2`) is built on `protein_plus_pathvqa_500_v3`: protein papers PLUS a PathVQA-augmented sub-corpus that adds histology / pathology / clinical-trial questions.

The papers themselves describe different concept neighborhoods. **The concept shift is by construction.** The PathVQA legacy explains the +4.0pp TissueRegion + 4.0pp Disease + 0.5pp StainingMethod emergence on B.

### 5.2 Different question type composition

| Question type | Bench A count | Bench A % | Bench B count | Bench B % |
|---|---|---|---|---|
| `claim_choice` | 382 | 47.8% | 198 | 40.0% |
| `essay` | 239 | 29.9% | 169 | 34.1% |
| `boolean_support` | 108 | 13.5% | 87 | 17.6% |
| `two_hop_tail` | 47 | 5.9% | 20 | 4.0% |
| `experiment_code` | 24 | 3.0% | 21 | 4.2% |

`claim_choice` (which produces shorter, more bounded responses with fewer mechanism claims) drops 7.8pp on B in favor of `essay` (+4.2pp) and `boolean_support` (+4.1pp). More open-ended formats → more diverse concept invocation → contributes to the per-bench concept divergence even before paper-pool effects.

### 5.3 Mixed judge cohorts (amplifier, not cause)

- **Bench A**: 8/11 models judged by `intern-s1-pro` (uniform), 3/11 by `gpt-4o`. Within-cohort Δ-judge offset bounded to ±0.05 in pilot.
- **Bench B**: 5/10 models judged by `glm5`, 5/10 by `gpt-4o`. Refuted-per-sample on `gpt-4o` cohort is **5.2×** the `glm5` cohort. A single concept might be flagged by `gpt-4o` but not by `glm5` (or vice versa), inflating apparent concept divergence.

Judge effects don't change the underlying concept-type distribution **within a single bench** materially, but they amplify cross-bench concept-list dissimilarity by adding judge-specific verdict noise.

---

## 6. What replicates across the shift (paper-grade structural findings)

Despite the near-disjoint concept space, three claims survive cross-bench verification:

### 6.1 Falsifiable-pattern share (~65%)

| Pattern | Bench A % | Bench B % | Δ |
|---|---|---|---|
| `false_negation` | 40.5% | 45.2% | +4.7% (B slightly higher) |
| `wrong_direction` | 12.9% | 11.3% | -1.6% |
| `fabricated_target` | 9.8% | 10.2% | +0.4% |
| **Sum (falsifiable)** | **63.2%** | **66.7%** | +3.5% |

About 2/3 of biomedical agent hallucinations are textually-detectable patterns regardless of bench. **Strongest empirical claim in this study** — replicates across 11 models, 2 benchmarks, 2 judge architectures.

### 6.2 Hub-concept concentration (the "Pareto" pattern)

Long-tail distribution of how many models refuted each concept:

| # models per concept | Bench A | Bench A % | Bench B | Bench B % |
|---|---|---|---|---|
| 1 model only | 612 | 70.7% | 172 | 68.8% |
| 2-3 models | 220 | 25.4% | 68 | 27.2% |
| 4+ models | 34 | 3.9% | 10 | 4.0% |

The shape is **identical**. ~70% of distinct hallucinated concepts are 1-model idiosyncratic; ~25% are 2-3 model spotty agreement; ~4% are universal hub concepts. **Bench composition affects WHICH concepts populate each bucket, not how the buckets are sized.**

### 6.3 Verbosity-driven HR / HF dissociation

On both benches:
- Bench A: `deepseek-v4-flash` produces 13+ refuted claims on a single toxic sample (`qg_000913`) while terse models give 1-2 claims for the same sample.
- Bench B: `deepseek-v4-flash` averages 15.88 claims/sample (top 1) vs `gpt-5.4-mini` 1.22 (bottom 1) — a 13× gap.

Verbose models inflate `HR_macro` (= refuted / total claims) AND total refuted instances, but the underlying behavior (mechanism-elaboration → exposes more surface area to fact-check) is the same. **HF_rate (sample-level) is more bench-portable than HR_macro for cross-model comparison.**

### 6.4 The `hiden` exception (one true cross-bench universal)

Only one concept ranks as "universal hallucination" on both benches:

| Bench | # models | refuted insts | example claim |
|---|---|---|---|
| A | 9/11 | 22 | "Knockout of HIDEN upregulates pluripotency genes (SOX2, OCT4)." |
| B | 6/10 | 19 | "HIDEN downregulates beta Catenin." / "HIDEN downregulates the Wnt signaling pathway." |

Different specific claims on different benches — but the same training-data prior leak: HIDEN as a "universal regulator" with simple direction-of-effect, rather than the paper-specific contextual nuance.

---

## 7. What does NOT replicate

### 7.1 Specific universal hallucinations don't transfer

The Bench A universals (`ptpn22` 10/11, `electronegative clusters` 9/11, `ace2` 9/11, `ripk3` 8/11, `grp78` 8/11, `usp22` 8/11) are **not present on Bench B**. The Bench B "universals" (`hiden`, `cenpa`, `cyld`) are not strong on Bench A (except hiden).

→ When the paper says "universal hallucination", it must specify "universal across the 11 models in Bench A" rather than "universal in biomedical agent QA generally".

### 7.2 Per-type concept dominance

On Bench A, `Drug` is dominated by `pcbp2` and `chonglouoside sl-8` (29 each). On Bench B, top Drug is `neuromodulin-1β` (7) and `lgr5` (7). The shift in absolute volume per concept (29 → 7) is bigger than the shift in proportional importance — partly because Bench B has 4.5× fewer total refuted instances, partly because the specific drugs studied in the two paper pools don't overlap.

### 7.3 Some models' bench-specific profiles

- `qwen3.6-plus` (r=0.159): Bench A profile (Protein/Drug/CellLine heavy) shifts to Drug/Gene heavy on Bench B. Likely tied to qwen's response style varying with question format.
- `gpt-5.4-mini` (r=0.262): Similar bench sensitivity. On A it skews to Drug + BiologicalProcess; on B it skews to Protein + Disease.

For these models, cross-bench inferences should be limited to aggregate metrics (HR, HS), not concept-type profiles.

---

## 8. Implications for the paper

### 8.1 Reframe "universal hallucination" as bench-conditional

Don't claim "models universally hallucinate on `ptpn22`". Do claim "On Bench A, 10 of 11 models refute `ptpn22` with the same incorrect mechanistic claim, demonstrating a paper-vs-prior conflict at the corpus level."

For cross-bench universality, only `hiden` clears the bar.

### 8.2 The dissociation is itself a finding

A 0.5% Jaccard between concept spaces of two ostensibly similar biomedical-QA benchmarks is a substantive empirical observation. **It argues against using a single benchmark to characterize "biomedical agent hallucination behavior" in general.** Camera-ready should:

1. Report the IoU number explicitly
2. Use it to motivate the dual-bench design
3. Restrict universal-pattern claims to those that survive Bench A AND Bench B (the 3 in §6)

### 8.3 Stability metrics for paper-grade ranking

When ranking models by hallucination, report two cuts:
- **Bench-A-only ranking** (most uniform judge cohort, largest N) — primary
- **Cross-bench correlation r** of the model's concept-type profile — diagnostic of "is this model's behavior bench-stable enough to generalize from one of these benches to the other?"

Models with r > 0.7 (kimi, intern, gemini) can be treated as having a stable hallucination signature. Models with r < 0.3 (qwen, gpt-5.4-mini) carry a "high bench-sensitivity" caveat.

### 8.4 The concept-type shift suggests a third bench

Bench B's TissueRegion + StainingMethod + Disease emergence (vs A's CellLine + MolecularEntity + BiologicalProcess focus) suggests biomedical agents have at least two distinct failure regimes:

- **A-style**: protein-mechanism, biochemistry, drug-target → concentrated on hub proteins / drugs
- **B-style**: tissue-level, clinical, gene-name → concentrated on specific patient / tissue contexts

A future "Bench C" focused on, e.g., epidemiology / patient-level outcomes might surface a third regime distinct from both. The 3-regime hypothesis predicts the IoU between any two of A/B/C would remain at ~0.5%.

---

## 9. Reproducibility data

All numbers come from these halu_results.jsonl files:

```
halu_runs/paired_protein_v2_balanced__full_tool_models_20260427/{model}/halu_results.jsonl   # 8 intern-judge models on A
halu_runs/paired_protein_v2_balanced__gpt4o_judge_20260501/{model}/halu_results.jsonl        # 3 gpt-4o-judge models on A (intern-self, deepseek, gemini)
halu_runs/paired_enhanced_v2_balanced__full_models_20260430/{model}/halu_results.jsonl       # 5 glm5-judge models on B
halu_runs/paired_enhanced_v2_balanced__gpt4o_judge_20260501/{model}/halu_results.jsonl       # 5 gpt-4o-judge models on B
```

Aggregation key: `(canonical_concept).lower().strip()` from the `claim` substructure of each refuted `judged_claim`. Concept-type is the `concept_type` field of the same `judged_claim`.

Pearson r computed over the canonical 8 concept types: Protein, Drug, CellLine, MolecularEntity, Gene, Complex, BiologicalProcess, RNA.

KL / JS divergences computed treating type-frequency vectors as discrete distributions over 16 type categories with ε=1e-9 smoothing.

### Companion files

- `full_results_report.md` — master accuracy + halu tables
- `halu_concept_analysis_bench_a.md` — Bench A concept deep-dive (single-bench)
- `halu_concept_analysis_bench_b.md` — Bench B concept deep-dive (single-bench)
- `halu_results_update_20260502.md` — changelog + infrastructure incidents
- **`halu_cross_bench_concept_shift.md`** ← _this file_

The 5 documents together are paper-ready supplementary material for hallucination claims. This document is the cross-bench validation layer; without it, single-bench claims are exposed to "bench-specific artifact" objections.
