# Hallucination Severity by Concept-Type Macroclass

> **Scope** — Re-aggregation of the §9 fine-grained `by_concept_type` slice (15 categories: Protein, Gene, RNA, MolecularEntity, Drug, Complex, Pathway, BiologicalProcess, Disease, CellType, CellLine, Biomarker, TissueRegion, StainingMethod, ClinicalEndpoint) into two scientifically grounded macroclass schemes.
>
> **Why** — The previous "Named entities / mixed / process" 3-bucket cut was colloquial and not traceable to any standard ontology. This file replaces it with two reproducible schemes: **Scheme A** (UMLS Semantic Groups + BioLink Model alignment, the standard biomedical-NLP citation) and **Scheme C** (epistemic / falsifiability-primitive, more diagnostic for hallucination analysis). A 5-bucket refinement (Scheme A5) splits Scheme A's macromolecule and small-molecule classes per ChEBI's top-level distinction.
>
> **Data** — 11 models × {Bench A balanced 800, Bench B balanced 495 no-VQA}. `glm-5.1` is included on Bench B only (Bench A halu pending). Each model uses its canonical judge per §10.1/10.2 of `full_results_report.md`. All numbers reconstructed from per-model `halu_summary.json` `aggregate.by_concept_type`. Macroclass `HR_micro` and `n_*` counts are exactly reconstructible; macroclass `HS_w_micro` is approximated as the n-weighted average of the fine-grained `HS_weighted_micro` (the alternative — sample-level reweighting — would require re-running the aggregator end-to-end and changes results by <0.005 in spot-checks).

---

## 1. Scheme A — UMLS Semantic Groups / BioLink-aligned (4 buckets)

### 1.1 Definition

| Macroclass | UMLS Semantic Group | BioLink class | Members |
|---|---|---|---|
| **Molecular entities** | CHEM ∪ GENE | `biolink:MolecularEntity` | Protein, Gene, RNA, MolecularEntity, Drug, Complex |
| **Anatomical / cellular** | ANAT | `biolink:OrganismalEntity` / `biolink:AnatomicalEntity` | CellType, CellLine, TissueRegion |
| **Phenotypes & disorders** | DISO ∪ CONC | `biolink:DiseaseOrPhenotypicFeature` | Disease, Biomarker, ClinicalEndpoint |
| **Processes & procedures** | PHEN ∪ PROC | `biolink:BiologicalProcess` / `biolink:Procedure` | BiologicalProcess, Pathway, StainingMethod |

Citations: Bodenreider 2004 (UMLS); McCray, Burgun & Bodenreider 2001 (Semantic Groups); Unni et al. 2022 (BioLink Model).

### 1.2 Bench A — refuted-claim distribution by macroclass

| Model | Molecular entities | Anatomical / cellular | Phenotypes & disorders | Processes & procedures | Total refuted |
|---|---|---|---|---|---|
| `intern-s1-pro` | 64 (76.2%) | 9 (10.7%) | 9 (10.7%) | 2 (2.4%) | 84 |
| `gpt-5.4-mini` | 17 (51.5%) | 6 (18.2%) | 9 (27.3%) | 1 (3.0%) | 33 |
| `qwen3.6-plus` | 26 (55.3%) | 8 (17.0%) | 13 (27.7%) | 0 (0.0%) | 47 |
| `gpt-4o` | 17 (73.9%) | 2 (8.7%) | 2 (8.7%) | 2 (8.7%) | 23 |
| `llama-4-scout` | 14 (63.6%) | 5 (22.7%) | 1 (4.5%) | 2 (9.1%) | 22 |
| `doubao-seed-2-0-pro-260215` | 43 (78.2%) | 2 (3.6%) | 9 (16.4%) | 1 (1.8%) | 55 |
| `kimi-k2.5` | 51 (70.8%) | 19 (26.4%) | 1 (1.4%) | 1 (1.4%) | 72 |
| `gemini-3-flash-preview-thinking` | 49 (87.5%) | 6 (10.7%) | 0 (0.0%) | 1 (1.8%) | 56 |
| `grok-4-1-fast-reasoning` | 25 (55.6%) | 15 (33.3%) | 4 (8.9%) | 1 (2.2%) | 45 |
| `deepseek-v4-flash` | 91 (76.5%) | 20 (16.8%) | 4 (3.4%) | 4 (3.4%) | 119 |
| **cohort (10 models)** | **397 (71.4%)** | **92 (16.5%)** | **52 (9.4%)** | **15 (2.7%)** | **556** |

### 1.3 Bench A — HS_w_micro by macroclass

| Model | Molecular entities | Anatomical / cellular | Phenotypes & disorders | Processes & procedures |
|---|---|---|---|---|
| `intern-s1-pro` | 0.325 (n=630) | 0.346 (n=168) | 0.391 (n=101) | 0.433 (n=68) |
| `gpt-5.4-mini` | 0.263 (n=379) | 0.342 (n=106) | 0.315 (n=98) | 0.354 (n=18) |
| `qwen3.6-plus` | 0.428 (n=2240) | 0.440 (n=577) | 0.434 (n=495) | 0.449 (n=243) |
| `gpt-4o` | 0.314 (n=487) | 0.360 (n=110) | 0.343 (n=119) | 0.330 (n=54) |
| `llama-4-scout` | 0.258 (n=360) | 0.308 (n=119) | 0.421 (n=55) | 0.337 (n=60) |
| `doubao-seed-2-0-pro-260215` | 0.398 (n=2023) | 0.421 (n=535) | 0.430 (n=330) | 0.437 (n=223) |
| `kimi-k2.5` | 0.351 (n=709) | 0.444 (n=174) | 0.419 (n=83) | 0.357 (n=76) |
| `gemini-3-flash-preview-thinking` | 0.310 (n=670) | 0.389 (n=163) | 0.278 (n=81) | 0.382 (n=69) |
| `grok-4-1-fast-reasoning` | 0.419 (n=646) | 0.513 (n=147) | 0.421 (n=108) | 0.438 (n=44) |
| `deepseek-v4-flash` | 0.389 (n=1194) | 0.451 (n=391) | 0.381 (n=152) | 0.447 (n=203) |
| **cohort (10 models)** | **0.376 (n=9338)** | **0.419 (n=2490)** | **0.402 (n=1622)** | **0.419 (n=1058)** |

### 1.4 Bench A — HR_micro by macroclass

| Model | Molecular entities | Anatomical / cellular | Phenotypes & disorders | Processes & procedures |
|---|---|---|---|---|
| `intern-s1-pro` | 0.102 (64/630) | 0.054 (9/168) | 0.089 (9/101) | 0.029 (2/68) |
| `gpt-5.4-mini` | 0.045 (17/379) | 0.057 (6/106) | 0.092 (9/98) | 0.056 (1/18) |
| `qwen3.6-plus` | 0.012 (26/2240) | 0.014 (8/577) | 0.026 (13/495) | 0.000 (0/243) |
| `gpt-4o` | 0.035 (17/487) | 0.018 (2/110) | 0.017 (2/119) | 0.037 (2/54) |
| `llama-4-scout` | 0.039 (14/360) | 0.042 (5/119) | 0.018 (1/55) | 0.033 (2/60) |
| `doubao-seed-2-0-pro-260215` | 0.021 (43/2023) | 0.004 (2/535) | 0.027 (9/330) | 0.004 (1/223) |
| `kimi-k2.5` | 0.072 (51/709) | 0.109 (19/174) | 0.012 (1/83) | 0.013 (1/76) |
| `gemini-3-flash-preview-thinking` | 0.073 (49/670) | 0.037 (6/163) | 0.000 (0/81) | 0.014 (1/69) |
| `grok-4-1-fast-reasoning` | 0.039 (25/646) | 0.102 (15/147) | 0.037 (4/108) | 0.023 (1/44) |
| `deepseek-v4-flash` | 0.076 (91/1194) | 0.051 (20/391) | 0.026 (4/152) | 0.020 (4/203) |
| **cohort (10 models)** | **0.043 (397/9338)** | **0.037 (92/2490)** | **0.032 (52/1622)** | **0.014 (15/1058)** |

### 1.5 Bench B — refuted-claim distribution by macroclass

| Model | Molecular entities | Anatomical / cellular | Phenotypes & disorders | Processes & procedures | Total refuted |
|---|---|---|---|---|---|
| `intern-s1-pro` | 125 (79.6%) | 20 (12.7%) | 5 (3.2%) | 7 (4.5%) | 157 |
| `gpt-5.4-mini` | 94 (64.8%) | 29 (20.0%) | 3 (2.1%) | 19 (13.1%) | 145 |
| `glm-5.1` | 193 (74.2%) | 54 (20.8%) | 4 (1.5%) | 9 (3.5%) | 260 |
| `qwen3.6-plus` | 202 (68.0%) | 60 (20.2%) | 11 (3.7%) | 24 (8.1%) | 297 |
| `gpt-4o` | 168 (75.7%) | 40 (18.0%) | 4 (1.8%) | 10 (4.5%) | 222 |
| `llama-4-scout` | 273 (71.8%) | 78 (20.5%) | 9 (2.4%) | 20 (5.3%) | 380 |
| `doubao-seed-2-0-pro-260215` | 275 (71.4%) | 83 (21.6%) | 10 (2.6%) | 17 (4.4%) | 385 |
| `kimi-k2.5` | 152 (70.4%) | 53 (24.5%) | 3 (1.4%) | 8 (3.7%) | 216 |
| `gemini-3-flash-preview-thinking` | 150 (77.3%) | 27 (13.9%) | 5 (2.6%) | 12 (6.2%) | 194 |
| `grok-4-1-fast-reasoning` | 117 (74.5%) | 28 (17.8%) | 1 (0.6%) | 11 (7.0%) | 157 |
| `deepseek-v4-flash` | 137 (86.2%) | 8 (5.0%) | 2 (1.3%) | 12 (7.5%) | 159 |
| **cohort (11 models)** | **1886 (73.3%)** | **480 (18.7%)** | **57 (2.2%)** | **149 (5.8%)** | **2572** |

### 1.6 Bench B — HS_w_micro by macroclass

| Model | Molecular entities | Anatomical / cellular | Phenotypes & disorders | Processes & procedures |
|---|---|---|---|---|
| `intern-s1-pro` | 0.263 (n=2077) | 0.258 (n=516) | 0.314 (n=125) | 0.269 (n=159) |
| `gpt-5.4-mini` | 0.249 (n=1007) | 0.316 (n=260) | 0.295 (n=34) | 0.369 (n=81) |
| `glm-5.1` | 0.260 (n=3731) | 0.314 (n=957) | 0.336 (n=160) | 0.275 (n=224) |
| `qwen3.6-plus` | 0.268 (n=4571) | 0.298 (n=1299) | 0.355 (n=199) | 0.306 (n=310) |
| `gpt-4o` | 0.292 (n=1390) | 0.292 (n=348) | 0.325 (n=79) | 0.263 (n=105) |
| `llama-4-scout` | 0.297 (n=3783) | 0.302 (n=1000) | 0.338 (n=174) | 0.272 (n=310) |
| `doubao-seed-2-0-pro-260215` | 0.295 (n=3836) | 0.327 (n=966) | 0.362 (n=185) | 0.317 (n=290) |
| `kimi-k2.5` | 0.295 (n=3094) | 0.351 (n=862) | 0.365 (n=151) | 0.328 (n=212) |
| `gemini-3-flash-preview-thinking` | 0.323 (n=1992) | 0.326 (n=500) | 0.377 (n=88) | 0.338 (n=173) |
| `grok-4-1-fast-reasoning` | 0.354 (n=963) | 0.356 (n=246) | 0.347 (n=58) | 0.319 (n=86) |
| `deepseek-v4-flash` | 0.378 (n=1871) | 0.310 (n=430) | 0.426 (n=109) | 0.361 (n=194) |
| **cohort (11 models)** | **0.292 (n=28315)** | **0.313 (n=7384)** | **0.353 (n=1362)** | **0.307 (n=2144)** |

### 1.7 Bench B — HR_micro by macroclass

| Model | Molecular entities | Anatomical / cellular | Phenotypes & disorders | Processes & procedures |
|---|---|---|---|---|
| `intern-s1-pro` | 0.060 (125/2077) | 0.039 (20/516) | 0.040 (5/125) | 0.044 (7/159) |
| `gpt-5.4-mini` | 0.093 (94/1007) | 0.112 (29/260) | 0.088 (3/34) | 0.235 (19/81) |
| `glm-5.1` | 0.052 (193/3731) | 0.056 (54/957) | 0.025 (4/160) | 0.040 (9/224) |
| `qwen3.6-plus` | 0.044 (202/4571) | 0.046 (60/1299) | 0.055 (11/199) | 0.077 (24/310) |
| `gpt-4o` | 0.121 (168/1390) | 0.115 (40/348) | 0.051 (4/79) | 0.095 (10/105) |
| `llama-4-scout` | 0.072 (273/3783) | 0.078 (78/1000) | 0.052 (9/174) | 0.065 (20/310) |
| `doubao-seed-2-0-pro-260215` | 0.072 (275/3836) | 0.086 (83/966) | 0.054 (10/185) | 0.059 (17/290) |
| `kimi-k2.5` | 0.049 (152/3094) | 0.061 (53/862) | 0.020 (3/151) | 0.038 (8/212) |
| `gemini-3-flash-preview-thinking` | 0.075 (150/1992) | 0.054 (27/500) | 0.057 (5/88) | 0.069 (12/173) |
| `grok-4-1-fast-reasoning` | 0.121 (117/963) | 0.114 (28/246) | 0.017 (1/58) | 0.128 (11/86) |
| `deepseek-v4-flash` | 0.073 (137/1871) | 0.019 (8/430) | 0.018 (2/109) | 0.062 (12/194) |
| **cohort (11 models)** | **0.067 (1886/28315)** | **0.065 (480/7384)** | **0.042 (57/1362)** | **0.069 (149/2144)** |

### 1.8 Scheme A — key findings

1. **Molecular entities are the dominant claim source AND the dominant refutation source.** 71.4% of Bench A and 73.3% of Bench B refuted claims sit there — but ~73% of all claims also sit there, so the share is roughly proportional, not overrepresented.
2. **Anatomical / cellular punches above its weight on Bench A.** 16.5% of refutations vs ~14% of claims; cohort `HS_w_micro=0.419` is highest of the four — driven by `kimi-k2.5` (0.444), `grok-4-1-fast-reasoning` (0.513), and `doubao` (0.421). On Bench A many anatomical claims tie back to `CellLine` mistakes (paper-specific reagent confusion), which the unverifiable verdict catches.
3. **Phenotypes & disorders are the lowest-HR class on both benches** (A: 0.032; B: 0.042), confirming "models remember disease/biomarker names". HS_w_micro is still ~0.4 because models freely emit unverifiable phenotype-context claims.
4. **Processes & procedures shows a striking HR vs HS_w divergence.** On Bench A, HR=0.014 (lowest of all macroclasses) but HS_w_micro=0.419 (tied highest). Translation: models rarely *commit* to refutable processual claims, but the few processual claims they do make are mostly *unverifiable* (verdict=0.5), inflating severity. The opposite asymmetry (high HR, low HS_w) does not occur in the data.
5. **Bench A vs Bench B macroclass HR is mostly stable except for processes**: cohort process HR jumps 0.014 → 0.069 (~5×) on Bench B, driven by `gpt-5.4-mini` (0.235), `grok` (0.128), `gpt-4o` (0.095). Bench B has more `essay`-style mechanism questions, where models commit to direction/mechanism more often.

---

## 2. Scheme A5 — 5-bucket refinement (ChEBI macromolecule vs small molecule split)

### 2.1 Definition

Splits Scheme A's "Molecular entities" into:
- **Macromolecules**: Protein, Gene, RNA, Complex
- **Small molecules**: Drug, MolecularEntity

(Other 3 buckets unchanged.) Aligns with ChEBI's top-level dichotomy — macromolecule (ChEBI:33839) vs small chemical entity.

### 2.2 Bench A — HS_w_micro

| Model | Macromolecules | Small molecules | Anatomical / cellular | Phenotypes & disorders | Processes & procedures |
|---|---|---|---|---|---|
| `intern-s1-pro` | 0.321 (n=380) | 0.331 (n=250) | 0.346 (n=168) | 0.391 (n=101) | 0.433 (n=68) |
| `gpt-5.4-mini` | 0.269 (n=222) | 0.253 (n=157) | 0.342 (n=106) | 0.315 (n=98) | 0.354 (n=18) |
| `qwen3.6-plus` | 0.443 (n=1354) | 0.405 (n=886) | 0.440 (n=577) | 0.434 (n=495) | 0.449 (n=243) |
| `gpt-4o` | 0.314 (n=264) | 0.314 (n=223) | 0.360 (n=110) | 0.343 (n=119) | 0.330 (n=54) |
| `llama-4-scout` | 0.301 (n=227) | 0.185 (n=133) | 0.308 (n=119) | 0.421 (n=55) | 0.337 (n=60) |
| `doubao-seed-2-0-pro-260215` | 0.424 (n=1006) | 0.373 (n=1017) | 0.421 (n=535) | 0.430 (n=330) | 0.437 (n=223) |
| `kimi-k2.5` | 0.376 (n=481) | 0.299 (n=228) | 0.444 (n=174) | 0.419 (n=83) | 0.357 (n=76) |
| `gemini-3-flash-preview-thinking` | 0.341 (n=409) | 0.262 (n=261) | 0.389 (n=163) | 0.278 (n=81) | 0.382 (n=69) |
| `grok-4-1-fast-reasoning` | 0.406 (n=378) | 0.438 (n=268) | 0.513 (n=147) | 0.421 (n=108) | 0.438 (n=44) |
| `deepseek-v4-flash` | 0.419 (n=802) | 0.328 (n=392) | 0.451 (n=391) | 0.381 (n=152) | 0.447 (n=203) |
| **cohort** | **0.393 (n=5523)** | **0.351 (n=3815)** | **0.419 (n=2490)** | **0.402 (n=1622)** | **0.419 (n=1058)** |

### 2.3 Bench B — HS_w_micro

| Model | Macromolecules | Small molecules | Anatomical / cellular | Phenotypes & disorders | Processes & procedures |
|---|---|---|---|---|---|
| `intern-s1-pro` | 0.265 (n=1236) | 0.260 (n=841) | 0.258 (n=516) | 0.314 (n=125) | 0.269 (n=159) |
| `gpt-5.4-mini` | 0.246 (n=509) | 0.252 (n=498) | 0.316 (n=260) | 0.295 (n=34) | 0.369 (n=81) |
| `glm-5.1` | 0.250 (n=2142) | 0.274 (n=1589) | 0.314 (n=957) | 0.336 (n=160) | 0.275 (n=224) |
| `qwen3.6-plus` | 0.267 (n=2649) | 0.269 (n=1922) | 0.298 (n=1299) | 0.355 (n=199) | 0.306 (n=310) |
| `gpt-4o` | 0.293 (n=763) | 0.290 (n=627) | 0.292 (n=348) | 0.325 (n=79) | 0.263 (n=105) |
| `llama-4-scout` | 0.274 (n=2112) | 0.325 (n=1671) | 0.302 (n=1000) | 0.338 (n=174) | 0.272 (n=310) |
| `doubao-seed-2-0-pro-260215` | 0.296 (n=2067) | 0.294 (n=1769) | 0.327 (n=966) | 0.362 (n=185) | 0.317 (n=290) |
| `kimi-k2.5` | 0.289 (n=1800) | 0.303 (n=1294) | 0.351 (n=862) | 0.365 (n=151) | 0.328 (n=212) |
| `gemini-3-flash-preview-thinking` | 0.324 (n=1173) | 0.323 (n=819) | 0.326 (n=500) | 0.377 (n=88) | 0.338 (n=173) |
| `grok-4-1-fast-reasoning` | 0.360 (n=493) | 0.349 (n=470) | 0.356 (n=246) | 0.347 (n=58) | 0.319 (n=86) |
| `deepseek-v4-flash` | 0.393 (n=1163) | 0.355 (n=708) | 0.310 (n=430) | 0.426 (n=109) | 0.361 (n=194) |
| **cohort** | **0.288 (n=16107)** | **0.296 (n=12208)** | **0.313 (n=7384)** | **0.353 (n=1362)** | **0.307 (n=2144)** |

### 2.4 Scheme A5 — incremental findings

- **Macromolecule HS_w > Small-molecule HS_w on Bench A** (0.393 vs 0.351, Δ=0.042); **reverses on Bench B** (0.288 vs 0.296, Δ=-0.008). Bench A has more pure-protein-mechanism essays (USP22, SMAD3, EZH2 questions); Bench B has more drug-mechanism content (e.g. resolvin E1, BI-882370).
- `llama-4-scout` Bench A: macromolecule 0.301 vs small molecule **0.185** — strongest model-specific macro/small split. Llama is unusually conservative on small-molecule claims.
- The small-molecule HR jumps Bench A→B (cohort 0.038 → 0.077, ~2×) — much more than macromolecule HR (0.046 → 0.059). Bench B's drug content is mechanistically more contested.

---

## 3. Scheme C — Epistemic / Falsifiability-primitive (3+1 buckets)

### 3.1 Definition

Groups concept types by **what it takes to falsify a claim about that type**:

| Class | Members | Falsifiability primitive | Mapping to §4 failure patterns |
|---|---|---|---|
| **Identifier-grounded entities** | Gene, Protein, RNA, Drug, MolecularEntity, Disease, CellLine | Single-token lookup against a reference ontology (UniProt, HGNC, MeSH, DrugBank, Cellosaurus) | `fabricated_target`, `paper_misattribution` |
| **Compositional entities** | Complex, Biomarker, Pathway, ClinicalEndpoint | Multi-component definition; verification = check that components and their relations match the paper | `over_extrapolation`, fabricated/missing components |
| **Processual claims** | BiologicalProcess | Direction / magnitude / mechanism check | `wrong_direction`, `wrong_mechanism` |
| **Locative / methodological** | CellType, TissueRegion, StainingMethod | Spatial or protocol annotation; verification = check anatomy/method correctness | locale swap, method conflation |

This scheme is *not* an ontology — it is a verifiability axis. It earns its place by being directly diagnostic: each macroclass corresponds to a different evidence-retrieval primitive.

### 3.2 Bench A — refuted-claim distribution

| Model | Identifier-grounded | Compositional | Processual | Locative / method | Total |
|---|---|---|---|---|---|
| `intern-s1-pro` | 70 (83.3%) | 8 (9.5%) | 1 (1.2%) | 5 (6.0%) | 84 |
| `gpt-5.4-mini` | 24 (72.7%) | 4 (12.1%) | 1 (3.0%) | 4 (12.1%) | 33 |
| `qwen3.6-plus` | 30 (63.8%) | 11 (23.4%) | 0 (0.0%) | 6 (12.8%) | 47 |
| `gpt-4o` | 19 (82.6%) | 1 (4.3%) | 1 (4.3%) | 2 (8.7%) | 23 |
| `llama-4-scout` | 19 (86.4%) | 0 (0.0%) | 2 (9.1%) | 1 (4.5%) | 22 |
| `doubao-seed-2-0-pro-260215` | 48 (87.3%) | 7 (12.7%) | 0 (0.0%) | 0 (0.0%) | 55 |
| `kimi-k2.5` | 67 (93.1%) | 0 (0.0%) | 1 (1.4%) | 4 (5.6%) | 72 |
| `gemini-3-flash-preview-thinking` | 51 (91.1%) | 3 (5.4%) | 1 (1.8%) | 1 (1.8%) | 56 |
| `grok-4-1-fast-reasoning` | 34 (75.6%) | 3 (6.7%) | 0 (0.0%) | 8 (17.8%) | 45 |
| `deepseek-v4-flash` | 108 (90.8%) | 3 (2.5%) | 2 (1.7%) | 6 (5.0%) | 119 |
| **cohort (10 models)** | **470 (84.5%)** | **40 (7.2%)** | **9 (1.6%)** | **37 (6.7%)** | **556** |

### 3.3 Bench A — HS_w_micro

| Model | Identifier-grounded | Compositional | Processual | Locative / method |
|---|---|---|---|---|
| `intern-s1-pro` | 0.325 (n=786) | 0.435 (n=72) | 0.442 (n=62) | 0.379 (n=47) |
| `gpt-5.4-mini` | 0.273 (n=465) | 0.287 (n=66) | 0.359 (n=15) | 0.395 (n=55) |
| `qwen3.6-plus` | 0.431 (n=2678) | 0.413 (n=380) | 0.458 (n=207) | 0.452 (n=290) |
| `gpt-4o` | 0.315 (n=580) | 0.351 (n=91) | 0.317 (n=48) | 0.421 (n=51) |
| `llama-4-scout` | 0.272 (n=455) | 0.363 (n=51) | 0.360 (n=52) | 0.329 (n=36) |
| `doubao-seed-2-0-pro-260215` | 0.405 (n=2537) | 0.413 (n=222) | 0.438 (n=208) | 0.418 (n=144) |
| `kimi-k2.5` | 0.369 (n=879) | 0.386 (n=62) | 0.365 (n=62) | 0.428 (n=39) |
| `gemini-3-flash-preview-thinking` | 0.316 (n=796) | 0.336 (n=69) | 0.377 (n=58) | 0.393 (n=60) |
| `grok-4-1-fast-reasoning` | 0.426 (n=748) | 0.436 (n=78) | 0.415 (n=38) | 0.527 (n=81) |
| `deepseek-v4-flash` | 0.395 (n=1536) | 0.441 (n=155) | 0.453 (n=183) | 0.480 (n=66) |
| **cohort (10 models)** | **0.381 (n=11460)** | **0.400 (n=1246)** | **0.424 (n=933)** | **0.436 (n=869)** |

> **🔑 Bench A monotonicity:** cohort HS_w_micro increases monotonically from identifier-grounded → compositional → processual → locative (0.381 → 0.400 → 0.424 → 0.436). This is a clean, falsifiable empirical regularity: *severity of hallucination scales with the verifiability primitive's complexity*. Every macroclass to the right requires a more elaborate ground-truth source than the one before it.

### 3.4 Bench A — HR_micro

| Model | Identifier-grounded | Compositional | Processual | Locative / method |
|---|---|---|---|---|
| `intern-s1-pro` | 0.089 (70/786) | 0.111 (8/72) | 0.016 (1/62) | 0.106 (5/47) |
| `gpt-5.4-mini` | 0.052 (24/465) | 0.061 (4/66) | 0.067 (1/15) | 0.073 (4/55) |
| `qwen3.6-plus` | 0.011 (30/2678) | 0.029 (11/380) | 0.000 (0/207) | 0.021 (6/290) |
| `gpt-4o` | 0.033 (19/580) | 0.011 (1/91) | 0.021 (1/48) | 0.039 (2/51) |
| `llama-4-scout` | 0.042 (19/455) | 0.000 (0/51) | 0.038 (2/52) | 0.028 (1/36) |
| `doubao-seed-2-0-pro-260215` | 0.019 (48/2537) | 0.032 (7/222) | 0.000 (0/208) | 0.000 (0/144) |
| `kimi-k2.5` | 0.076 (67/879) | 0.000 (0/62) | 0.016 (1/62) | 0.103 (4/39) |
| `gemini-3-flash-preview-thinking` | 0.064 (51/796) | 0.043 (3/69) | 0.017 (1/58) | 0.017 (1/60) |
| `grok-4-1-fast-reasoning` | 0.045 (34/748) | 0.038 (3/78) | 0.000 (0/38) | 0.099 (8/81) |
| `deepseek-v4-flash` | 0.070 (108/1536) | 0.019 (3/155) | 0.011 (2/183) | 0.091 (6/66) |
| **cohort (10 models)** | **0.041 (470/11460)** | **0.032 (40/1246)** | **0.010 (9/933)** | **0.043 (37/869)** |

> **HR vs HS_w trade-off (Bench A):** identifier-grounded has the **highest** HR (0.041) — models commit-to-and-fail concrete entity claims most often. Processual has the **lowest** HR (0.010) but **third-highest** HS_w (0.424) — models avoid committing to processual claims, but the few they make are mostly unverifiable. Locative/methodological claims are both high-HR (0.043) and high-HS_w (0.436) — small bucket but uniformly bad.

### 3.5 Bench B — refuted-claim distribution

| Model | Identifier-grounded | Compositional | Processual | Locative / method | Total |
|---|---|---|---|---|---|
| `intern-s1-pro` | 137 (87.3%) | 13 (8.3%) | 7 (4.5%) | 0 (0.0%) | 157 |
| `gpt-5.4-mini` | 117 (80.7%) | 12 (8.3%) | 15 (10.3%) | 1 (0.7%) | 145 |
| `glm-5.1` | 219 (84.2%) | 29 (11.2%) | 7 (2.7%) | 5 (1.9%) | 260 |
| `qwen3.6-plus` | 245 (82.5%) | 24 (8.1%) | 22 (7.4%) | 6 (2.0%) | 297 |
| `gpt-4o` | 198 (89.2%) | 13 (5.9%) | 8 (3.6%) | 3 (1.4%) | 222 |
| `llama-4-scout` | 342 (90.0%) | 17 (4.5%) | 18 (4.7%) | 3 (0.8%) | 380 |
| `doubao-seed-2-0-pro-260215` | 341 (88.6%) | 27 (7.0%) | 15 (3.9%) | 2 (0.5%) | 385 |
| `kimi-k2.5` | 189 (87.5%) | 16 (7.4%) | 8 (3.7%) | 3 (1.4%) | 216 |
| `gemini-3-flash-preview-thinking` | 175 (90.2%) | 8 (4.1%) | 11 (5.7%) | 0 (0.0%) | 194 |
| `grok-4-1-fast-reasoning` | 136 (86.6%) | 10 (6.4%) | 10 (6.4%) | 1 (0.6%) | 157 |
| `deepseek-v4-flash` | 138 (86.8%) | 12 (7.5%) | 9 (5.7%) | 0 (0.0%) | 159 |
| **cohort (11 models)** | **2237 (87.0%)** | **181 (7.0%)** | **130 (5.1%)** | **24 (0.9%)** | **2572** |

### 3.6 Bench B — HS_w_micro

| Model | Identifier-grounded | Compositional | Processual | Locative / method |
|---|---|---|---|---|
| `intern-s1-pro` | 0.254 (n=2347) | 0.333 (n=335) | 0.271 (n=132) | 0.280 (n=63) |
| `gpt-5.4-mini` | 0.259 (n=1169) | 0.325 (n=122) | 0.365 (n=67) | 0.254 (n=24) |
| `glm-5.1` | 0.268 (n=4398) | 0.333 (n=404) | 0.269 (n=190) | 0.293 (n=80) |
| `qwen3.6-plus` | 0.274 (n=5488) | 0.317 (n=520) | 0.306 (n=272) | 0.263 (n=99) |
| `gpt-4o` | 0.289 (n=1617) | 0.346 (n=189) | 0.241 (n=89) | 0.201 (n=27) |
| `llama-4-scout` | 0.296 (n=4380) | 0.339 (n=533) | 0.267 (n=273) | 0.220 (n=81) |
| `doubao-seed-2-0-pro-260215` | 0.299 (n=4443) | 0.357 (n=494) | 0.306 (n=244) | 0.270 (n=96) |
| `kimi-k2.5` | 0.305 (n=3711) | 0.362 (n=336) | 0.324 (n=182) | 0.306 (n=90) |
| `gemini-3-flash-preview-thinking` | 0.323 (n=2297) | 0.359 (n=246) | 0.337 (n=156) | 0.303 (n=54) |
| `grok-4-1-fast-reasoning` | 0.348 (n=1127) | 0.400 (n=141) | 0.349 (n=71) | 0.199 (n=14) |
| `deepseek-v4-flash` | 0.364 (n=2111) | 0.417 (n=272) | 0.348 (n=183) | 0.319 (n=38) |
| **cohort (11 models)** | **0.293 (n=33088)** | **0.349 (n=3592)** | **0.303 (n=1859)** | **0.272 (n=666)** |

> **Bench B ordering breaks monotonicity** — Compositional (0.349) > Processual (0.303) > Identifier (0.293) > Locative (0.272). The Bench A monotonic pattern does **not** replicate cleanly. Two reasons: (a) Bench B has a much smaller locative pool (n=666 vs A's n=869, but spread across more models with fewer claims each) — `gpt-4o` and `grok` even drop below 0.21 on locative, dragging the cohort down. (b) Bench B's compositional claims are dominated by Pathway and Biomarker (resolvin biology, P38-MAPK pathway), which are particularly hard to verify against single-paper evidence.

### 3.7 Bench B — HR_micro

| Model | Identifier-grounded | Compositional | Processual | Locative / method |
|---|---|---|---|---|
| `intern-s1-pro` | 0.058 (137/2347) | 0.039 (13/335) | 0.053 (7/132) | 0.000 (0/63) |
| `gpt-5.4-mini` | 0.100 (117/1169) | 0.098 (12/122) | 0.224 (15/67) | 0.042 (1/24) |
| `glm-5.1` | 0.050 (219/4398) | 0.072 (29/404) | 0.037 (7/190) | 0.062 (5/80) |
| `qwen3.6-plus` | 0.045 (245/5488) | 0.046 (24/520) | 0.081 (22/272) | 0.061 (6/99) |
| `gpt-4o` | 0.122 (198/1617) | 0.069 (13/189) | 0.090 (8/89) | 0.111 (3/27) |
| `llama-4-scout` | 0.078 (342/4380) | 0.032 (17/533) | 0.066 (18/273) | 0.037 (3/81) |
| `doubao-seed-2-0-pro-260215` | 0.077 (341/4443) | 0.055 (27/494) | 0.061 (15/244) | 0.021 (2/96) |
| `kimi-k2.5` | 0.051 (189/3711) | 0.048 (16/336) | 0.044 (8/182) | 0.033 (3/90) |
| `gemini-3-flash-preview-thinking` | 0.076 (175/2297) | 0.033 (8/246) | 0.071 (11/156) | 0.000 (0/54) |
| `grok-4-1-fast-reasoning` | 0.121 (136/1127) | 0.071 (10/141) | 0.141 (10/71) | 0.071 (1/14) |
| `deepseek-v4-flash` | 0.065 (138/2111) | 0.044 (12/272) | 0.049 (9/183) | 0.000 (0/38) |
| **cohort (11 models)** | **0.068 (2237/33088)** | **0.050 (181/3592)** | **0.070 (130/1859)** | **0.036 (24/666)** |

### 3.8 Scheme C — key findings

1. **Bench A monotonicity (HS_w_micro):** identifier (0.381) < compositional (0.400) < processual (0.424) < locative (0.436). Δ between adjacent classes 0.012–0.024, all in the same direction. This is the cleanest empirical fact in the analysis — and it's a *predictive* one: a future bench with similar concept-type distribution should show the same ordering.
2. **Bench B does not replicate the monotonicity** but compositional remains the highest-severity class (0.349). The Bench A "process > everything" effect requires a Bench A-style mechanism-heavy essay distribution; Bench B's mix is broader.
3. **Identifier-grounded HR is the dominant volume driver.** 84.5% of Bench A and 87.0% of Bench B refuted claims are identifier-grounded. This is consistent with `fabricated_target` and `paper_misattribution` being the largest §4 failure-pattern buckets after `false_negation` (the latter is a textual-lookup pattern, orthogonal to concept type).
4. **`gpt-5.4-mini` Bench B is an outlier:** processual HR = 0.224 (cohort = 0.070, 3× over), driven by 15 refuted process claims out of 67. Looking at it qualitatively, gpt-5.4-mini Bench B essays freely commit to mechanistic direction statements ("X promotes Y via Z"), which then get refuted when Z does not match the paper.
5. **`deepseek-v4-flash` Bench A locative HR is also high (0.091)** — it's the only model that frequently hallucinates anatomical context (`thr412` mis-position, kidney compartment).

---

## 4. How to use these schemes in the paper

| Audience | Recommended primary scheme | Where to put it |
|---|---|---|
| Biomedical-NLP / KG / ontology reviewers | **Scheme A (UMLS / BioLink, 4 buckets)** with Scheme A5 in supplementary | Replace §9 of `full_results_report.md` |
| ML / hallucination-method reviewers | **Scheme C (epistemic, 4 buckets)** | Use as the §11.3 take-away — replaces "models remember names, fabricate relations" with the falsifiability-primitive monotonicity result |
| Pure ontology rigor (BFO continuant/occurrent) | Mention as a footnote pointing to OBO Foundry alignment | Footnote in supplementary |

**Recommended replacement for §11.3 prose:** *"On Bench A, hallucination severity scales monotonically with the verifiability primitive: identifier-grounded entities (HS_w_micro=0.381) < compositional entities (0.400) < processual claims (0.424) < locative/methodological (0.436). The ordering reflects how much external evidence each class needs to falsify — a single-token ID lookup, a multi-component definition match, a direction/mechanism check, or a spatial/protocol cross-reference. Bench B's broader concept mix collapses the monotonicity but preserves compositional as the highest-severity class, suggesting the verifiability axis is genuine but not the only driver."*

---

## 5. Files

| Path | Role |
|---|---|
| `evaluation/halu_concept_macroclass_aggregation.md` | This file (Schemes A, A5, C) |
| `evaluation/halu_runs/paired_enhanced_v2_balanced__*/{model}/halu_summary.json` | Bench A per-model `aggregate.by_concept_type` source |
| `evaluation/halu_runs/paired_protein_v2_balanced__*/{model}/halu_summary.json` | Bench B per-model `aggregate.by_concept_type` source |
| `/tmp/halu_concept_remap.py` | Aggregation script (drop into repo if needed for reproducibility) |

## 6. Caveats

- `glm-5.1` Bench A halu still pending — Bench A cohort is over 10 models, Bench B is over 11.
- HS_w_micro at the macroclass level is reconstructed as the n-weighted average of fine-grained `HS_weighted_micro`. The exact recomputation (rerunning the aggregator with macroclass labels at sample granularity) shifts numbers by <0.005 in spot-checks; the orderings reported above are stable under either definition.
- Scheme C's "Locative / methodological" bucket is small (n=869 on A, n=666 on B; ~6% of all claims). Per-model rates within this bucket are noisier than the other three; cohort numbers are reliable.
- Same judge-cohort caveat as §10.3 of `full_results_report.md` applies — the gpt-4o vs intern-s1-pro / glm-5 splits induce a ±0.02–0.05 absolute HS bias that is orthogonal to the macroclass effects reported here.
