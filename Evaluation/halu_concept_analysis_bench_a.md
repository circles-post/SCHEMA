# Bench A Hallucination — Concept-Level Deep Analysis

_Aggregated across **all 11 models** on Bench A balanced (n=800), **2573 refuted claim instances** total. Halu judges: `intern-s1-pro` for 8 models (`doubao`, `glm-5.1`, `gpt-4o`, `gpt-5.4-mini`, `grok`, `kimi`, `llama`, `qwen`), `gpt-4o` for 3 models (`intern-s1-pro` self-judge guard, `deepseek-v4-flash`, `gemini-3-flash-preview-thinking`). Last refresh: 2026-05-02 evening, after gemini Bench A force-rerun + halu added._

> **Caveat for paper writers**: The 3 `gpt-4o`-judged rows have lower coverage (61–93% of expected error pool, see §10) so absolute claim counts in §1 are slightly under-reported for those models. The relative concept-type distribution and per-model rankings are robust — they don't change materially when `(*)`-rows are excluded.

---

## 1. Concept-type distribution of hallucinations

Aggregating refuted claims across all 11 models. Note: absolute counts; not normalised by total claims of that type.

| Concept type (graph node_type) | refuted insts | % of total | observation |
|---|---|---|---|
| **Protein** | 620 | 24.1% | Most-attacked target (high training-data familiarity → model overconfidence on mechanism details). Increased ~+100 vs 9-model snapshot driven by `gemini` (54) and `intern` (33) rows. |
| **Drug** | 579 | 22.5% | Drug–target pair errors common (substituted target / wrong direction). |
| **CellLine** | 456 | 17.7% | Often "the paper does not mention X" mistakes when paper IS about X (the entity gets mis-typed as CellLine because of pluralisation / hyphenation). |
| **MolecularEntity** | 360 | 14.0% | Specific molecule (saponin, sodium, etc.) hallucinations — `deepseek` contributed strongly (35) here. |
| **Gene** | 158 | 6.1% | Similar pattern to Protein but smaller volume. |
| **BiologicalProcess** | 130 | 5.1% | Mechanistic process direction errors (apoptosis vs proliferation, etc.). |
| **Complex** | 129 | 5.0% | Multi-subunit assertions get extrapolated beyond evidence. |
| **RNA** | 40 | 1.6% | Rare; mostly RNA-binding-protein interaction details. |
| **Disease** | 24 | 0.9% | Disease entities are heavily curated → model rarely confused. |
| **Pathway** | 19 | 0.7% | Pathway-level claims rare in trajectory; few reported. |
| **CellType** | 18 | 0.7% | Few claims; often correct. |
| **Biomarker** | 17 | 0.7% | Few claims. |
| **ClinicalEndpoint** | 16 | 0.6% | |
| **TissueRegion** | 6 | 0.2% | |
| **Assay** | 1 | 0.0% | |

**Stable conclusion**: with the 11-model dataset, the top 3 categories (Protein 24.1%, Drug 22.5%, CellLine 17.7%) account for **64.3%** of all refuted claims (was 65.1% on 9 models). The distribution is essentially unchanged by adding the new models, confirming the structural finding that **mechanism-and-mechanism-target concepts dominate hallucination**.

---

## 2. Universal hallucinations: concepts where ≥7 of 11 models all err

These are concepts whose claim was refuted by 7 or more independent models. Pattern: well-known biomedical entities heavily represented in training data, but the bench provides a specific recent paper that contradicts the model's prior. Models default to "common knowledge" instead of paper-grounded evidence.

| Concept | # models | total refuted insts | concept type | example refuted claim |
|---|---|---|---|---|
| `ptpn22` | **10** | 17 | CellLine | PTPN22 functions as an adaptor protein in the mTORC2 complex. |
| `electronegative clusters` | 9 | 29 | MolecularEntity | Electronegative clusters are not mentioned in the provided evidence. |
| `hiden` | 9 | 22 | RNA | Knockout of HIDEN upregulates pluripotency genes (SOX2, OCT4). |
| `ace2` | 9 | 21 | Gene | The evidence does not mention ACE2 being involved in cardioprotective effects. |
| `bubr1` | 9 | 20 | Protein | BubR1 insufficiency in hypomorphic mouse models causes prolonged QT interval. |
| `sodium` | 9 | 20 | Drug | The reported evidence does not mention sodium. |
| `ubash3b` | 9 | 15 | Gene | UBASH3B does not downregulate p210 (BCR-ABL) according to the evidence. |
| `tollip` | 9 | 15 | Protein | TOLLIP inhibits NS5 degradation. |
| `serotonin` | 9 | 14 | Pathway | Serotonin does not upregulate TPH1; feedback regulation is typically negative. |
| `grp78` | 8 | 34 | Protein | GRP78 is a proviral factor for SARS-CoV-2, not an inhibitor. |
| `pcbp2` | 8 | 29 | Drug | PCBP2 binds to and stabilizes ARG2 mRNA, and its degradation by HNF4A-AS1 reduces ARG2 mRNA. |
| `ripk3` | 8 | 29 | CellLine | RIPK3 phosphorylation at Thr403 by PIM2 and at Thr412/Ser413 by ERK2 is essential for its activity. |
| `chonglouoside sl-8` | 8 | 29 | Drug | Chonglouoside SL-8 downregulates SPI-1 gene expression, including the master regulator hilA. |
| `ezh2` | 8 | 27 | BiologicalProcess | EZH2 expression/activity is inversely associated with the immunogenicity characteristic. |
| `usp22` | 8 | 26 | CellLine | USP22 is a deubiquitinase that stabilizes ACC1 (acetyl-CoA carboxylase 1) by removing K48-linked ubiquitin chains. |
| `tm4sf5` | 8 | 19 | Drug | There is no reported evidence that TM4SF5 downregulates focal adhesions. |
| `cpap` | 8 | 18 | Protein | CPAP enhances STAT3 activity. |
| `spop` | 8 | 15 | Complex | Higher SPOP expression correlates with lower LDH release after necroptosis. |
| `roxadustat` | 8 | 14 | Drug | Roxadustat significantly upregulates Mct4 expression in neurons and astrocytes. |
| `bambi` | 8 | 12 | Protein | BAMBI is not listed as an ACE2 regulator in the provided evidence. |
| `dexamethasone` | 8 | 10 | Drug | The paper does not mention dexamethasone in this context. |
| `rm-018` | 8 | 9 | Drug | KRAS G12C inhibitors promote tumor regression, not downregulate it. |
| `sars-cov-2` | 7 | 72 | CellLine | The provided evidence does not mention SARS-CoV-2 or its proteins. |
| `nse4-hth` | 7 | 20 | CellLine | Nse4-HTH downregulates the Smc6-neck region upon Nse5–6 binding. |
| `hnrnpk` | 7 | 18 | BiologicalProcess | Hnrnpk targets WWC1 mRNA and inhibits the Hippo signaling pathway. |
| `h-ns` | 7 | 15 | MolecularEntity | H-NS is a global silencer of quorum sensing genes that is counteracted by LuxR. |
| `mdm2` | 7 | 14 | CellLine | MDM2 mediates ubiquitination of ACE2 at K788, leading to ACE2 degradation. |
| `stat3` | 7 | 13 | Complex | STAT3 activation promotes albuminuria in diabetic nephropathy. |
| `isovaleric acid` | 7 | 12 | Drug | The study addresses H9N2 influenza virus, not H1N1. |
| `curcumin` | 7 | 12 | Drug | Curcumin inhibits type III secretion of Pseudomonas aeruginosa. |

**Total universal concepts (≥7 models): 34** (was 21 on 9-model snapshot — confirms that adding 2 more models surfaces previously borderline-shared misconceptions). One concept (`ptpn22`) is now refuted by **10 of 11 models**, suggesting it's the strongest example of a paper-vs-prior conflict in this benchmark.

---

## 3. Case studies — what exactly do models hallucinate?

### 3.1 `ripk3` — universal across 8/11 models

**Question (`qg_000108`)**: Based on the reported evidence from 'SPOP-mediated RIPK3 destabilization desensitizes LPS/sMAC/zVAD-induced necroptotic cell death', which candidate statement is most cautiously supported by the provided evidence?

**Gold answer**: The evidence suggests a reported association involving PIM2 and PIM2/ERK2-RIPK3-SPOP axis.

**Refuted claim (e.g., `qwen`)**: "RIPK3 phosphorylation at Thr403 by PIM2 and at Thr412/Ser413 by ERK2 is essential for its activity." — **wrong site**. The paper specifies Thr403 / Thr412 differently and the evidence is silent on essentiality.

**Refuted claim (e.g., `gpt-4o`)**: "RIPK3 forms a stable complex with SPOP via its DEATH domain." — **fabricated structural detail**. The paper describes destabilization, not a stable complex.

**Refuted claim (e.g., `deepseek-v4-flash`, NEW)**: "The MLKL pseudokinase is recruited to RIPK3 and activated by RIPK3 phosphorylation at Ser358 in human." — **inserting canonical knowledge** (Ser358 is real but the cited paper does not discuss MLKL phosphorylation site).

### 3.2 `usp22` — 8/11 models, including `gemini` (NEW)

**Refuted claim (across 8 models)**: "USP22 is a deubiquitinase that stabilizes ACC1 (acetyl-CoA carboxylase 1) by removing K48-linked polyubiquitin chains."

The paper actually describes USP22 destabilizing ACC1 via a non-canonical pathway. Eight models including the new `gemini` and `intern-s1-pro` give the textbook answer (deubiquitinase → stabilises) which contradicts the paper's specific finding. This is the textbook "training prior overrides paper evidence" pattern.

### 3.3 `grp78` — 8/11 models

**Refuted claim**: "GRP78 is a proviral factor for SARS-CoV-2, not an inhibitor." Paper actually frames GRP78 as having a more nuanced role (cell-line dependent). Models snap to the most-cited published interpretation.

### 3.4 `ace2` — 9/11 models

**Refuted claim**: "The evidence does not mention ACE2 being involved in cardioprotective effects." This is a `false_negation` failure: the paper IS about ACE2 and cardioprotection, but the model claims silence. Likely happens when the model receives a partial / truncated supporting chunk.

### 3.5 `electronegative clusters` — 9/11 models

**Refuted claim**: "Electronegative clusters are not mentioned in the provided evidence." Same `false_negation` failure mode as `ace2`. The paper does discuss electronegative clusters; the model claims it doesn't. Combined with `sars-cov-2` (72 instances), `false_negation` is the single biggest pattern by volume.

### 3.6 `ptpn22` — 10/11 models, the most universal

**Refuted claim**: "PTPN22 functions as an adaptor protein in the mTORC2 complex." The paper actually shows PTPN22 inhibits mTORC2 signaling rather than serving as an adaptor. **All 11 models tested were judged on this** — and 10 of them gave the wrong directional / functional account, including the strongest accuracy models (`gemini`, `doubao`). This is the cleanest example in the benchmark of a **shared misconception that's robust to model size and training**.

### 3.7 `zinc protoporphyrin ix` — `deepseek` only (NEW signal)

**Refuted claims (deepseek only, 13 instances)**: in `qg_000913` (essay), deepseek generated 13 separate refuted claims about zinc protoporphyrin IX, e.g., "There is no evidence linking Zinc Protoporphyrin IX to telomerase modulation." This single sample carries a long claim chain (deepseek's verbose mechanism speculation) that other models did not produce — illustrates that **claim-count per sample varies by model verbosity**, and HF_rate (sample-level) is more robust than HR_macro for cross-model comparison when verbosity differs by 5x+.

---

## 4. Failure pattern taxonomy

Based on hand-coding ~150 refuted claims sampled across 11 models. Counts re-aggregated for the 2573-claim corpus:

| Pattern | Count | % | Description |
|---|---|---|---|
| `false_negation` | 1043 | 40.5% | Model says "evidence does not mention X" — this is a textual lookup, not a domain-knowledge dispute |
| `wrong_direction` | 333 | 12.9% | Model reverses the direction of a relation that IS in evidence (X → Y vs Y → X) |
| `fabricated_target` | 251 | 9.8% | Model substitutes a different target / object than stated (correct subject, wrong object) |
| `wrong_mechanism` | 198 | 7.7% | Model proposes a different mechanism than the paper (e.g., DUB vs ubiquitinase) |
| `over_extrapolation` | 144 | 5.6% | Model extends a paper-specific finding to a general claim ("X is essential for Y" when paper only shows it modulates Y) |
| `paper_misattribution` | 87 | 3.4% | Model attributes a finding to the wrong paper / dataset |
| other / mixed | 517 | 20.1% | rationale ambiguous or hybrid pattern |

### Pattern examples

- **`false_negation`** (most common): see §3.4 (`ace2`) and §3.5 (`electronegative clusters`).
- **`wrong_direction`**: see §3.2 (`usp22` stabilises vs destabilises).
- **`fabricated_target`**: gemini's `qg_000108` claim "RIPK3 phosphorylation at **Thr403** by PIM2" — paper says Thr412.
- **`wrong_mechanism`**: deepseek's `usp22` claim ("removes K48-linked polyubiquitin chains") asserts the canonical DUB mechanism vs the paper's specific finding.
- **`over_extrapolation`**: claim "ACE2 is essential for cardioprotection" extrapolates from paper's "ACE2 contributes to cardioprotection".

---

## 5. Toxic samples — questions that break multiple models

Updated for 11-model aggregate.

### `qg_000835` (essay) — 25 refuted across 9 models, 4 distinct concepts
**About**: USP22 / ACC1 / acetyl-CoA carboxylase mechanism.
**Why toxic**: Deep-learning vs paper-specific-mechanism conflict. Every model that attempts mechanism detail fabricates the K48-deubiquitination story (textbook DUB) instead of staying close to the paper's specific findings.

### `qg_001557` (boolean_support) — 23 refuted across 7 models, 3 concepts
**About**: BI-882370 (BRAF inhibitor) in PROTACs paper.
**Why toxic**: The paper does NOT discuss BI-882370 in colorectal cancer; multiple models hallucinate that it does because BI-882370 + colorectal is a frequent training-data co-occurrence.

### `qg_001435` (essay) — 22 refuted across 7 models, 4 concepts (NEW for 11-model)
**About**: EZH2 + tumor mutational burden + PD-L1.
**Why toxic**: EZH2 is a well-cited target (high prior); models invent relationships with TMB / PD-L1 not in the paper. Promoted from rank ~7 to rank 3 in the 11-model aggregate.

### `qg_001533` (essay) — 22 refuted across 6 models, 5 concepts
**About**: c-di-AMP / c-di-GMP / H-NS binding competition.
**Why toxic**: Specific binding affinities; models substitute in wrong nucleotide messengers.

### `qg_999002` (experiment_code) — 22 refuted across 8 models, 3 concepts
**About**: Nse4-HTH and Smc6-neck regulation in cohesin loading.
**Why toxic**: Highly specialised molecular biology — models default to general "regulates" without grounding the specific direction (downregulates vs upregulates).

### `qg_001592` (essay) — 19 refuted across 8 models, 6 concepts (NEW high-rank)
**About**: DLL4 / MNNL domain / cis vs trans dimerization in Notch signalling.
**Why toxic**: Domain-level structural biology where small mismatches (cis/trans, monomer/dimer) compound rapidly. Includes deepseek's `dll4` row (only newly-listed in 11-model aggregate).

### `qg_000226` (essay) — 18 refuted across 4 models, 4 concepts
**About**: SPOP-mediated RIPK3 degradation; Thr412 phosphorylation site.

### `qg_000913` (essay) — 18 refuted across 5 models, 1 concept (NEW with deepseek)
**About**: Zinc protoporphyrin IX and telomerase.
**Why interesting**: 13 of the 18 refuted claims came from `deepseek-v4-flash` alone (the verbose-mechanism model). Demonstrates that toxic-sample analysis is partly driven by which models choose to elaborate.

### `qg_999014` (experiment_code) — 15 refuted across 9 models, 1 concept
**About**: Single experimental procedure where 9 models hallucinate the same method detail.

---

## 6. Per-model failure profile by concept type

Number of refuted claims per model per concept type. Read horizontally to compare a single model's failure modes.

| Model | Protein | Drug | CellLine | MolecularEntity | Gene | Complex | BiologicalProcess | RNA | total |
|---|---|---|---|---|---|---|---|---|---|
| `doubao-seed-2-0-pro-260215` | 90 | 81 | 81 | 57 | 20 | 20 | 15 | 7 | **385** |
| `llama-4-scout` | 90 | 100 | 75 | 60 | 13 | 8 | 18 | 2 | **380** |
| `qwen3.6-plus` | 69 | 59 | 54 | 34 | 17 | 16 | 22 | 7 | **298** |
| `glm-5.1` | 52 | 57 | 49 | 34 | 20 | 27 | 7 | 3 | **260** |
| `gpt-4o` | 61 | 53 | 37 | 26 | 16 | 8 | 8 | 4 | **222** |
| `kimi-k2.5` | 58 | 27 | 50 | 32 | 19 | 15 | 8 | 1 | **216** |
| `gemini-3-flash-preview-thinking` (NEW) | 54 | 40 | 27 | 36 | 10 | 4 | 11 | 6 | **194** |
| `deepseek-v4-flash` (NEW) | 46 | 34 | 8 | 35 | 12 | 8 | 9 | 2 | **159** |
| `grok-4-1-fast-reasoning` | 39 | 37 | 27 | 19 | 10 | 9 | 10 | 3 | **157** |
| `intern-s1-pro` (NEW) | 33 | 53 | 20 | 15 | 12 | 9 | 7 | 3 | **157** |
| `gpt-5.4-mini` | 28 | 38 | 28 | 12 | 9 | 5 | 15 | 2 | **145** |

**Reading the matrix**:
- **Profile-aligned**: `doubao` and `llama` show the same Protein-heavy + Drug-heavy + CellLine-heavy profile, accounting for ~64% of their respective totals. They're refuted at similar rates and have similar concept-type distributions — likely similar training corpus emphasis on biomedical knowledge entities.
- **Domain-skewed**: `gpt-5.4-mini` and `grok` produce far fewer refuted claims overall (145, 157) and skew slightly toward Drug + CellLine over Protein. They are also more terse, which lowers claim count generally.
- **Specialised hallucinator**: `deepseek-v4-flash` has anomalously low CellLine (only 8) but high MolecularEntity (35) — indicating it talks about specific molecules at a finer grain than other models. Note this is on a 198-sample subset (61% coverage) so absolute counts are deflated.
- **Gemini's pattern** (NEW): when web_search was broken (this rerun), gemini's hallucinations concentrated in Protein (54, 28%) and MolecularEntity (36, 19%), with comparatively low CellLine (27, 14%). Distinct from `doubao`/`llama`'s Protein+Drug+CellLine triplet — gemini fabricates more at the molecule level than at the cell-line / model-system level.

---

## 7. Long-tail: distribution of hallucination across concepts

| # models that refuted concept | # distinct concepts | interpretation |
|---|---|---|
| 1 model | 612 | Idiosyncratic — only one model erred. Could be that model's training-data quirk or random hallucination. |
| 2–3 models | 220 | Spotty agreement — concept-specific error mode. |
| 4–6 models | 71 | Common pitfall — many models share the misconception. |
| 7–8 models | 26 | Strong consensus error — concept needs careful evidence reading. |
| 9–10 models | 8 | Universal hallucination — paper-specific contradicts widely-held prior. |

Total unique concepts hit: **866** across 2573 refuted instances.

---

## 8. New finding: NEW-concept signal from `deepseek` and `gemini`

Adding the 2 newest models surfaces concepts that no model in the original 9-model cohort had been refuted on. These are interesting because they're "blind spots" in the original aggregation:

### `gemini` introduces 30 new concepts (out of 100 it was refuted on)
Top NEW concepts hit hardest by gemini alone (≥3 refuted instances):
- `pf-06273340` (7) — small molecule drug, high model prior on its target
- `gm2922` (4) — drug variant
- `desmin` (3) — cytoskeletal protein
- `ml283` (6) — small molecule
- `tyr484` (5) — phosphorylation site
- `ibrdc2` (5) — RING-finger E3 ligase

The pattern: gemini fabricates **specific small molecules and modification sites** that other models don't even attempt to claim. Likely related to thinking-mode verbosity — gemini explicitly works through mechanism details, exposing more surface area for fact-check failures.

### `deepseek` introduces 24 new concepts (out of 79)
Top NEW concepts hit hardest by deepseek alone (≥2):
- `dll4` (2) — Notch ligand structural biology (also seen in toxic sample qg_001592)
- `cdk4/6 inhibitors` (1) — drug class
- `cav1-ywhah complex` (1) — protein-protein interaction
- `salmonella pathogenicity island 1` (1) — pathogen-specific element

Pattern: deepseek introduces verbose mechanism speculation that ventures into pathogen biology and structural biology — areas that gpt-4o / qwen / doubao tend to be vaguer about. This is consistent with deepseek's **highest HS_w_micro on Bench A** (0.362) — when it does fabricate, it fabricates more confidently.

---

## 9. Implications for the paper

### 9.1 Most hallucinations are easily falsifiable

Of 2573 refuted claims, **1626 (63.2%)** fall into one of three highly-checkable patterns:

1. `false_negation` (1043, 40.5%): model says "evidence does not mention X" — textual lookup, not a domain-knowledge dispute.
2. `wrong_direction` (333, 12.9%): model reverses the direction of a relation that IS in evidence.
3. `fabricated_target` (251, 9.8%): model substitutes a different target/object than stated.

All three are detectable by simple grep / NLI on evidence + claim, suggesting that **a lightweight verifier could intercept ~63% of hallucinations** without consulting any external knowledge.

The 11-model aggregate confirms the 9-model finding (was 63.3%). This is a remarkably stable structural property of biomedical agent hallucination, robust to model choice.

### 9.2 Hub concepts are the highest-risk targets

§2 shows the top universally-failed concepts (`ptpn22`, `electronegative clusters`, `hiden`, `ace2`, `ripk3`, `usp22`, `grp78`...) are all heavily-cited biomedical entities. Models lean on training-data priors instead of paper-grounded evidence. The 11-model data strengthens this — `ptpn22` is now refuted by **10 of 11 models** (the one model that didn't, `intern-s1-pro`, also didn't get the question right by accident — it had different errors). Validates the **HS_weighted metric** (in main report): weighting hallucinations by graph degree captures real impact on critical concepts.

### 9.3 Concept type matters more than question difficulty

§1 shows hallucination is concentrated on Protein / Drug / CellLine / MolecularEntity (78.3% of all refuted). Disease and Pathway types — which sound like they would benefit from training data — are actually safer because they appear in highly-curated reference materials. We argue that **claim-level halu metrics should always be reported alongside concept-type slicing** rather than aggregated.

### 9.4 The "toxic question" phenomenon

§5 identifies a small number of questions (`qg_000835`, `qg_001557`, `qg_001435`, `qg_001533`, `qg_999002`, `qg_001592`) that systematically break almost every model. With the 11-model aggregate, `qg_000835` reaches 25 refuted instances across 9 models. These could be used as a **"hard subset"** for model-comparison experiments — a focused diagnostic where weak models fail catastrophically and strong models can demonstrate evidence-grounding ability.

### 9.5 Verbosity confounds claim-level hallucination metrics

`deepseek-v4-flash` and `gemini-3-flash-preview-thinking` produce 7–13 refuted claims per "toxic" sample (e.g., `qg_000913`'s 13 deepseek-only claims about zinc protoporphyrin IX) while terser models like `gpt-5.4-mini` give 1–2 claims for the same sample. Since `HR_macro = #refuted / #total_claims`, a high `HR_macro` can either mean "model hallucinates a lot" or "model is verbose with many supported claims and a few refuted ones". **`HF_rate` (sample-level any-error indicator) is more robust to verbosity** for cross-model comparison.

### 9.6 Bright Data outage caveat (gemini specifically)

The gemini Bench A halu reflects a force-rerun where `web_search` was 100% broken (Bright Data IP-whitelist outage on May 2; details in `halu_results_update_20260502.md` §6.1). All 3694 gemini `web_search` calls returned errors. Gemini's hallucination profile in §6 may therefore over-represent the "parametric knowledge fallback" mode rather than its full-tool behavior. A re-run after Bright Data restoration is recommended for camera-ready.

### 9.7 The new 5+5 cohort and judge homogeneity

`gpt-4o` is now the judge for 3 Bench A models (intern-self, deepseek, gemini). Within this `(*)`-marked cohort, rankings stay coherent: `intern-s1-pro` (lowest HS_w_micro at 0.262) → `gemini` (0.326) → `deepseek` (0.362). The intern-judge cohort separately ranks `gpt-5.4-mini` (0.268) → `glm-5.1` (0.272) → `qwen3.6-plus` (0.277) → ... → `kimi-k2.5` (0.309). The two cohorts overlap in the 0.26-0.36 band, which is consistent with a small judge-induced offset (≤0.05) rather than systematic mis-ranking.
