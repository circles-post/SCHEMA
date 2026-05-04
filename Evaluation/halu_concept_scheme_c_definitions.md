# Scheme C — Epistemic / Falsifiability-Primitive 概念分类定义

> 用于 hallucination-by-concept-type 分析的四桶分类法。不按"实体属于哪个生物学科目"切分(那是 Scheme A 的 UMLS 思路),而是按 **"要证伪一条关于该实体的陈述,需要什么类型的证据"** 来切分。
>
> 理论根基:Popper (1959) 可证伪性原则 + 当代生物医学本体论的 referent identifiability 概念 + BFO 2.0 的 continuant / occurrent 二分。

---

## 总览

| 桶号 | 名称 | Falsifiability primitive | 成员 fine-grained types |
|---|---|---|---|
| 1 | Identifier-grounded entities | Identifier lookup (1 步) | Gene, Protein, RNA, Drug, MolecularEntity, Disease, CellLine |
| 2 | Compositional entities | Part-decomposition (多步组合) | Complex, Biomarker, Pathway, ClinicalEndpoint |
| 3 | Processual claims | Causal/process tracing (多步因果) | BiologicalProcess |
| 4 | Locative / methodological | Locative annotation (查 figure/methods) | CellType, TissueRegion, StainingMethod |

---

## 1. Identifier-grounded entities — 标识符可锚定实体

**成员**: `Gene, Protein, RNA, Drug, MolecularEntity, Disease, CellLine`

### 定义

在国际公认的权威数据库里有 **唯一稳定标识符 (stable accession ID)** 的实体。一条关于它的陈述,理论上可以用 *单次数据库查找* 完成证伪。

### 对应的权威 ID 系统

| Concept | Authority | Identifier 例子 |
|---|---|---|
| Gene | HGNC (Tweedie et al. 2021), NCBI Gene | `HGNC:3236` (EGFR) |
| Protein | UniProtKB (UniProt Consortium 2023) | `P00533` |
| RNA | RNAcentral, miRBase | `URS0000759CF4` |
| Drug | DrugBank (Wishart et al. 2018), ChEMBL, RxNorm | `DB00530` (erlotinib) |
| MolecularEntity | ChEBI (Hastings et al. 2016) | `CHEBI:15377` (water) |
| Disease | MONDO (Vasilevsky et al. 2022), MeSH, OMIM | `MONDO:0007254` |
| CellLine | Cellosaurus (Bairoch 2018) | `CVCL_0023` (HeLa) |

### Falsifiability primitive

**Identifier lookup**。"EGFR is on chromosome 7" 的真假可以一步到 HGNC/Ensembl 验证,不需要语义理解,也不需要拼装多个事实。

### 为什么单独成桶

这是 LLM **最有可能出错** 也 **最容易被 retrieval 修复** 的一类。模型常 hallucinate 的失败模式是 *fabricate-an-ID-shaped-string* (例如编造一个不存在的 UniProt accession),而这恰恰是判别器最容易抓的。

---

## 2. Compositional entities — 组合性实体

**成员**: `Complex, Biomarker, Pathway, ClinicalEndpoint`

### 定义

**没有单一权威标识符**,实体的身份由 *它的成分 (part-whole 关系) + 其成分之间的关系* 共同定义。要证伪一条关于它的陈述,必须先 **拆解** 出组成部分再分别比对。

### 为什么必然是 compositional

- **Complex** (蛋白复合物,如 *PRC2*): 由 EZH2 + SUZ12 + EED + RbAp46/48 组成。同一个名字可以指代不同的亚基组合 (PRC2.1 vs PRC2.2)。Reactome / CORUM 给出的是 *组合定义* 而非单一 ID。
- **Biomarker** ("PD-L1 high tumor"): 是 protein + threshold + assay + scoring system 的复合体。FDA-NIH BEST (2016) 明确 biomarker 是 a *defined characteristic*,不是单一分子。
- **Pathway** ("MAPK signaling"): KEGG / Reactome 给出的是 *节点和边的图*,不是 ID。两个数据库对同一通路的边界划法都不一致。
- **ClinicalEndpoint** ("progression-free survival at 12 months"): outcome variable + measurement instrument + time horizon + statistical definition 的组合。CDISC / CONSORT 标准明确是 "operationally defined."

### Authority

GO Cellular Component (Ashburner et al. 2000) 对 complex 的定义、Reactome (Jassal et al. 2020) 对 pathway 的层级、FDA-NIH BEST glossary 对 biomarker 和 endpoint 的形式化定义。

### Falsifiability primitive

**Compositional verification** — 必须先把声称拆成 "X 包含 Y" + "Y 在 Z 条件下满足 W",每一项再单独查证。比 identifier lookup 多一层结构。

---

## 3. Processual claims — 过程性陈述

**成员**: `BiologicalProcess`

### 定义

不是关于"是什么"的陈述,而是 **关于"如何变化、向哪个方向变化、在什么机制下变化"** 的陈述。本体论上属于 Basic Formal Ontology (Arp, Smith & Spear 2015) 的 **occurrent** (与 continuant 相对)。

### Authority

- **BFO 2.0** (ISO/IEC 21838-2:2021): continuant (entity that persists through time) vs. **occurrent** (entity that *happens* — process, event, state change)。BiologicalProcess 是典型 occurrent。
- **GO Biological Process branch** (Ashburner et al. 2000; Gene Ontology Consortium 2023): 形式化定义 ~30k 个 biological process 节点,每个都带 *directionality* (e.g., `GO:0006915` "apoptotic process" 是单向的细胞死亡过程)。

### 为什么这一桶最难证伪

一个过程性陈述 (例如 "LPS activates NF-κB which upregulates TNF-α") 至少包含三个可质疑维度:

1. **方向性**: activates 还是 inhibits?
2. **机制路径**: 直接还是通过中介?中介是谁?
3. **条件依赖**: 在哪些细胞类型/刺激/时间点?

不存在单一数据库能给出"是/否"判定。需要的是 *primary literature 综述*,而综述本身可能矛盾。

### Falsifiability primitive

**Process tracing** — 必须重建因果链条并逐步比对文献证据。这正好对应 Bench A 上 `BiologicalProcess` 桶的 HS_w_micro=0.424 (并列最高) 但 HR_micro=0.014 (最低) 的奇特模式 — 模型对过程性陈述的 unverifiable 比例特别高,因为判别器自己也很难拍板。

---

## 4. Locative / methodological claims — 位置/方法学陈述

**成员**: `CellType, TissueRegion, StainingMethod`

### 定义

陈述的不是实体本身的身份,而是 **"在哪里"** 或 **"用什么方法看到"** — 即关于 *观察上下文* 的元数据。

### 为什么独立成桶

- **CellType** (Cell Ontology, Diehl et al. 2016): `CL:0000236` "B cell" 是有 ID 的,但科研陈述里出现的形式通常是 "CD19+ B cells in germinal centers" — 这里 cell type 实际是 *anatomical location + marker panel* 的复合定位词。
- **TissueRegion** (UBERON, Mungall et al. 2012): "hippocampal CA1" 同样是 anatomical-location locator,不是被研究的实体本身,而是 "实体 X 在哪里被观察到"。
- **StainingMethod** ("DAB IHC", "RNAscope", "H&E"): 完全是方法学元数据 — 在 *什么实验技术下* 观察到陈述里的现象。OBI (Ontology for Biomedical Investigations, Bandrowski et al. 2016) 把这些归类为 `assay`。

### Authority

Cell Ontology, UBERON, OBI — 三者都是 OBO Foundry 成员,与 BFO 对齐 (它们把这些归为 *role* 或 *spatial region*,不是被陈述的物理实体)。

### Falsifiability primitive

**Locative annotation** — 通常需要去原始 figure legend / methods section 才能验证,而不是去 entity database。

### 实证观察

Bench A 上 locative 桶 HS_w_micro=0.436 (cohort 最高);refuted 占比却很小 (6.7%),因为大量陈述被判 unverifiable — 判别器不知道作者用的是哪种染色、哪个亚区,从证据上无法证伪。

---

## 关键学术依据汇总

| 思想来源 | 关键文献 | Scheme C 哪部分用到 |
|---|---|---|
| Falsifiability 原则 | Popper (1959) *The Logic of Scientific Discovery* | 整体框架 |
| Continuant vs occurrent | Arp, Smith & Spear (2015) *Building Ontologies with BFO*; ISO/IEC 21838-2:2021 | 桶 3 vs 桶 1/2 的根本切分 |
| Stable accession 是 entity 身份的充分必要条件 | Bodenreider (2004) *NAR* — UMLS Metathesaurus | 桶 1 的 identifier-grounded 判据 |
| Compositional definition (part-of, has-part) | Smith et al. (2005) *Genome Biology* — Relation Ontology | 桶 2 的成分依赖性 |
| Occurrent / process semantics | Gene Ontology Consortium (2023) GO BP 分支 | 桶 3 的 directionality / mechanism |
| Locative role vs. independent continuant | Diehl et al. (2016) Cell Ontology; OBI (Bandrowski et al. 2016) | 桶 4 的"上下文 vs 实体"切分 |

---

## 一句话核心

> 桶号 = **要证伪一条陈述需要走的认知步数**
> - 1 步 = identifier-grounded
> - 多步 part-decomposition = compositional
> - 多步 causal-tracing = processual
> - 看 figure / methods 元数据 = locative

这正是为什么 Bench A 上四桶的 HS_w_micro 是单调上升的 (0.381 → 0.400 → 0.424 → 0.436) — 它直接编码了"这条陈述有多难被外部证据反驳"。

---

## 与本仓库的对应关系

- **聚合脚本**: `/tmp/halu_concept_remap.py` 中 `SCHEME_C` 字典定义。
- **结果文档**: `evaluation/halu_concept_macroclass_aggregation.md` §3 (Scheme C 全部表格)。
- **建议落点**: `full_results_report.md` §11.3 take-away 句替换为 Scheme C 单调性结论;主表 §9 仍用 Scheme A (UMLS/BioLink),Scheme C 作为 supplementary。
