# Question Generation Workflow

## 1. 目标

基于现有科学知识图谱产物，生成事实可回溯的评测问题，用于评测 LLM 的科学事实理解能力。

当前实现是一个 **evidence-first scientific benchmark generator + hybrid model validation framework**。

当前支持 **4 类题型**：
- `claim_choice`（选择题）
- `boolean_support`（支持性判断题）
- `two_hop_tail`（两跳链式补全题）
- `essay`（论述题，由 LLM-as-Judge 评判）

当前还支持 **1 类 GitHub-assisted 实验代码题**：
- `experiment_code`（实验代码补全题，先搜索 GitHub 参考项目，再生成 `data_code + incomplete_main_code + unit_tests`）

核心原则：
- 以 evidence 驱动 claim 构造与题型选择，而非仅依赖 relation label
- 先选答案，再生成问题
- 规则做 hard guardrails，模型做语义判决
- essay 题型的参考答案正确性由 LLM-as-Judge 判定
- 模型不可用时保守降级

---

## 2. 当前工作流总览

```text
normalized_triples.jsonl + chunks.jsonl
        ↓
load triples / chunks
        ↓
build index
        ↓
sample subgraph candidates
        ↓
evidence profiling (strength / hedge / claim)
        ↓
evidence-gated question type selection
        ↓
evidence-first claim construction
        ↓
generate benchmark-style questions
  ├─ claim_choice / boolean_support / two_hop_tail → 选择题/判断题
  ├─ essay → 开放论述题 + 参考答案
  └─ experiment_code → GitHub-assisted 实验代码题
        ↓
rule-based hard validation
        ↓
literature retrieval (local + PubMed)
        ↓
model-based verdict (intern-s1-pro)
  ├─ claim_choice / boolean_support / two_hop_tail → judge_claim (三态判决)
  └─ essay → judge_essay (LLM-as-Judge 评分 + rationale)
        ↓
deduplicate
        ↓
export question_samples.jsonl + summary.json
```

---

## 3. 输入数据

### 3.1 `normalized_triples.jsonl`
每条 triple 包含：`doc_id`, `chunk_id`, `head`, `head_type`, `normalized_relation`, `tail`, `tail_type`, `confidence`, `evidence`

### 3.2 `chunks.jsonl`
每条 chunk 包含：`doc_id`, `chunk_id`, `title`, `section`, `text`, `start_offset`, `end_offset`

---

## 4. 模块结构

### 4.1 真值构造与采样层
- `config.py` / `models.py` / `io.py` / `indexing.py` / `subgraph_builder.py` / `sampler.py`

### 4.2 Evidence-first 层
- `evidence_profiler.py`：计算 evidence strength / hedge score / relation strength
- `evidence_claims.py`：基于 evidence profile 生成 claim 文本

### 4.3 问题生成层
- `templates.py`：claim-strength 驱动的题面模板（含 essay 开放式问题模板）
- `generator.py`：组装 question / answer / options / metadata

### 4.4 校验层
- `validator.py`：rule-based hard checks + hybrid model orchestration

### 4.5 Hybrid validation 层
- `retrieval_validator.py`：本地 + PubMed 证据检索
- `model_validator.py`：
  - `judge_claim()`：对选择题/判断题做三态判决
  - `judge_essay()`：对论述题做 LLM-as-Judge 评分
- `validation_prompts.py`：claim verification prompt + essay judge prompt
- `validation_cache.py` / `validation_types.py`

### 4.6 导出层
- `dedup.py` / `exporter.py` / `cli.py`

### 4.7 GitHub-assisted 实验代码层
- `github_tools.py`：搜索 GitHub 仓库和代码片段
- `experiment_templates.py`：relation → life-science 实验模板映射
- `experiment_generator.py`：生成 `data_code`、`main_code`、挖空函数、unit tests、GitHub 参考结果

---

## 5. Evidence-first 出题逻辑

### 5.1 为什么不能只用 relation label 出题

像 `TAAR6 associated_with mTOR pathway` 这种边，`associated_with` 本身语义很弱。而每条边上保留的 `evidence` 往往更具体。因此当前系统以 evidence 驱动 claim 构造与题型选择。

### 5.2 Evidence Profiler

由 `evidence_profiler.py` 完成，对每个 sampled subgraph 计算：

**Relation strength**：
- strong：`activates / inhibits / upregulates / downregulates / overexpressed_in / promotes / improves`
- weak：`associated_with / part_of`
- medium：其他

**Hedge score**：扫描 evidence + chunk 文本中的 hedge 词（suggest / may / might / potential / possibly / correlat / linked），出现越多分越高。

**Evidence strength**（综合判定）：
- `strong`：relation 强 + support >= 2 + hedge < 0.34
- `weak`：relation 弱 或 hedge >= 0.67
- `medium`：其他

### 5.3 Evidence-gated 题型选择

| evidence_strength | 允许的题型 |
|---|---|
| strong | `claim_choice`, `boolean_support`, `essay`, `two_hop_tail` |
| medium | `claim_choice`, `boolean_support`, `essay` |
| weak | `claim_choice`, `essay` |

注意：`essay` 在所有 evidence strength 下都允许生成，因为论述题本身可以适应不同证据强度。

### 5.4 Evidence-first Claim Construction

由 `evidence_claims.py` 完成。Claim 文本根据 evidence strength 生成不同层次：

- **weak**：`The evidence suggests a reported association involving X and Y.`
- **medium**：`The evidence supports a contextual relationship in which X relation Y.`
- **strong**：`The available evidence supports that X relation Y.`

并附带 evidence snippet 作为上下文。

---

## 6. 当前支持的问题类型

### 6.1 `claim_choice`（选择题）
给定一组候选 claim，问哪一个最被当前证据支持。

- 不假设图是完整的，不要求全局唯一答案
- 选项文本也是 evidence-aware 的
- 弱证据时题面用 `which candidate statement is most cautiously supported`
- 强证据时用 `which candidate claim is most directly supported`
- 通常 4 个选项（1 正确 + 3 干扰项）

### 6.2 `boolean_support`（支持性判断题）
给定一个 claim，判断是否被证据支持。

- 选项：`Supported / Not supported`
- 只有 evidence_strength >= medium 时才允许生成
- 题面使用 `claim_text_conservative`

### 6.3 `two_hop_tail`（两跳链式补全题）
对两跳链 `A -> B -> C`，问中间实体 B。

- 只有 evidence_strength = strong 时才允许生成
- 每一跳都必须有独立证据支持

### 6.4 `essay`（论述题）
开放式科学论述题，没有固定选项，答案是自由文本。

**生成逻辑**：
- `options = []`（无选项）
- 参考答案由 evidence claim + evidence snippets 自动组装
- `answer_type = "Essay"`

**题面根据 evidence_strength 自动调整**：
- **weak evidence**：
  > Based on the reported evidence from '...', describe the reported relationship between X and Y, and explain what the available evidence indicates about their interaction.
- **medium evidence**：
  > Based on the reported evidence from '...', based on the provided evidence, explain the relationship between X and Y, and discuss the strength of the supporting findings.
- **strong evidence**：
  > Based on the reported evidence from '...', explain the mechanism by which X [relation] Y, citing the supporting evidence.

**正确性判定**：
- 不使用精确匹配
- 由 **LLM-as-Judge**（intern-s1-pro）评估参考答案的：
  - 科学准确性
  - 证据支持度
  - 完整性
- 输出 `essay_score`（0.0-1.0）和 `essay_rationale`（评分理由）

### 6.5 `experiment_code`（GitHub-assisted 实验代码题）
给定 grounded scientific claim，把事实关系改写成一个小型科研实验编程任务。

**Agent workflow**：
1. 从 triple + evidence 生成 experiment spec
2. 调用 `github_tools.py` 搜索相关 GitHub 仓库与代码实现
3. 生成 `data_code`
4. 生成完整 `main_code`
5. 挖去 1-2 个关键函数形成 `incomplete_main_code`
6. 导出 unit tests 与预期结果

**当前模板族**：
- biomarker screening
- differential expression
- drug response
- pathway activity

**输出特点**：
- `question` 中包含：
  - scientific claim
  - research direction
  - agent workflow
  - GitHub repository / code search results
  - `data_code`
  - `main_code`（实际导出的是挖空后的 incomplete 版本）
- `answer.text` 存储参考完整 `main_code`
- `metadata` 中额外存储：
  - `data_code`
  - `main_code`
  - `incomplete_main_code`
  - `incomplete_functions`
  - `unit_tests`
  - `github_references`

---

## 7. Rule-based Hard Validation

由 `validator.py` 中 `validate_sample_rule_based(...)` 完成。

当前硬规则：
1. **Evidence alignment**：triple.evidence 必须能在 chunk.text 中找到
2. **最低多源支持**：至少 2 条 supporting evidence / 2 个 doc / 2 个 chunk
3. **答案泄漏**：question 中不能直接包含答案文本（essay / boolean / claim_choice 跳过此检查）
4. **两跳唯一性**：two_hop_tail 的中间节点必须唯一
5. **Evidence-gated 题型合法性**：question_type 必须在 evidence profile 允许的范围内

注意：type-relation 白名单已移除（全放行）。

---

## 8. Hybrid Model Validation（已接入 intern-s1-pro）

### 8.1 架构
规则预检通过后，进入 hybrid model validation。根据题型走不同判决路径：

#### 对 claim_choice / boolean_support / two_hop_tail
调用 `judge_claim()`：
1. 构建 evidence bundle（本地 + PubMed）
2. intern-s1-pro 做三态判决：`supported / insufficient_evidence / contradicted`
3. 输出 `support_score` 和 `confidence_band`

#### 对 essay
调用 `judge_essay()`：
1. 构建 evidence bundle（本地 + PubMed）
2. intern-s1-pro 作为 LLM-as-Judge 评估参考答案
3. 输出：
   - `verdict`：三态判决
   - `essay_score`：0.0-1.0 质量分
   - `essay_rationale`：评分理由
   - `issue_tags`：发现的问题标签

### 8.2 LLM-as-Judge 评估维度
essay 题的 LLM-as-Judge prompt 要求模型评估：
1. **Scientific accuracy**：参考答案是否基于证据在科学上正确
2. **Evidence support**：答案中的每个 claim 是否都有证据支持
3. **Completeness**：答案是否充分回答了问题

### 8.3 当前已验证的模型配置

```
model:    intern-s1-pro
base_url: https://chat.intern-ai.org.cn/api/v1/
api_key:  (已配置)
```

### 8.4 实际运行结果（proteinlmbench_full_graph_v1）

#### claim_choice 结果

| 样本 | evidence_strength | model_verdict | support_score |
|---|---|---|---|
| Ena/VASP - T cell actin remodeling | medium | supported | 1.0 |
| MDM2 inhibits TP53 | strong | supported | 0.95 |
| HP1 - H3K9me association | weak | supported | 0.9 |

#### essay 结果

| 样本 | evidence_strength | essay_score | verdict | rationale 摘要 |
|---|---|---|---|---|
| Ena/VASP - T cell actin remodeling | medium | 0.9 | supported | 正确描述了 Ena/VASP 在 T cell actin remodeling 中的角色 |
| MDM2 inhibits TP53 | strong | 0.95 | supported | 准确描述了三个抑制机制（转录抑制、核输出、泛素化降解） |
| HP1 - H3K9me | weak | 0.8 | supported | 正确但有重复内容，描述了 HP1 结合 H3K9me 抑制转录 |

### 8.5 Degraded 模式
如果 `base_url / api_key / model` 为空或模型调用失败：
- 不报错，不中断主流程
- 保留 rule-based 通过结果
- `validation_mode = degraded`
- `double_checked = False`

---

## 9. CLI 使用方式

### 9.1 Rule-only 模式

```bash
PYTHONPATH="/path/to/datasetsa" \
python -m question_generation.cli \
  --triples "/path/to/normalized_triples.jsonl" \
  --chunks "/path/to/chunks.jsonl" \
  --output "/path/to/question_samples.jsonl" \
  --summary-output "/path/to/question_summary.json" \
  --question-types claim_choice boolean_support two_hop_tail essay \
  --validation-mode rule_only
```

### 9.2 Hybrid model 模式（含 essay LLM-as-Judge）

```bash
PYTHONPATH="/path/to/datasetsa" \
python -m question_generation.cli \
  --triples "/path/to/normalized_triples.jsonl" \
  --chunks "/path/to/chunks.jsonl" \
  --output "/path/to/question_samples.jsonl" \
  --summary-output "/path/to/question_summary.json" \
  --question-types claim_choice boolean_support two_hop_tail essay \
  --validation-mode hybrid_model \
  --retrieval-top-k 3 \
  --validator-enabled \
  --validator-model "intern-s1-pro" \
  --validator-base-url "https://chat.intern-ai.org.cn/api/v1/" \
  --validator-api-key "YOUR_API_KEY"
```

### 9.3 仅 essay 题型

```bash
PYTHONPATH="/path/to/datasetsa" \
python -m question_generation.cli \
  --triples "/path/to/normalized_triples.jsonl" \
  --chunks "/path/to/chunks.jsonl" \
  --output "/path/to/question_samples.jsonl" \
  --question-types essay \
  --validation-mode hybrid_model \
  --validator-enabled \
  --validator-model "intern-s1-pro" \
  --validator-base-url "https://chat.intern-ai.org.cn/api/v1/" \
  --validator-api-key "YOUR_API_KEY"
```

### 9.4 CLI 参数一览

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--triples` | normalized_triples.jsonl 路径 | 必填 |
| `--chunks` | chunks.jsonl 路径 | 必填 |
| `--output` | 输出 question_samples.jsonl 路径 | 必填 |
| `--summary-output` | 输出 summary json 路径 | 可选 |
| `--max-samples` | 最大采样数 | 100 |
| `--min-confidence` | 最低置信度 | 0.7 |
| `--question-types` | 题型列表 | claim_choice boolean_support two_hop_tail essay |
| `--validation-mode` | rule_only 或 hybrid_model | rule_only |
| `--retrieval-top-k` | PubMed 检索条数 | 3 |
| `--validation-cache-dir` | 验证缓存目录 | 可选 |
| `--validator-enabled` | 启用模型验证 | 默认关闭 |
| `--validator-model` | 模型名 | 空 |
| `--validator-base-url` | API base URL | 空 |
| `--validator-api-key` | API key | 空 |
| `--github-search-language` | GitHub 检索语言过滤 | Python |
| `--github-search-per-page` | 每次 GitHub 检索返回条数 | 3 |

### 9.5 生成实验代码题

```bash
PYTHONPATH="/path/to/datasetsa" \
python -m question_generation.cli \
  --triples "/path/to/normalized_triples.jsonl" \
  --chunks "/path/to/chunks.jsonl" \
  --output "/path/to/question_samples.jsonl" \
  --question-types experiment_code \
  --validation-mode rule_only \
  --github-search-language Python \
  --github-search-per-page 3
```

---

## 10. 输出数据格式

### 10.1 `question_samples.jsonl`
每条样本包含：

- `sample_id`
- `question_type`：`claim_choice / boolean_support / two_hop_tail / essay / experiment_code`
- `question`
- `answer`：`text / canonical_text / answer_type`
  - 选择题：`answer_type = "Claim" / "Boolean" / 实体类型`
  - 论述题：`answer_type = "Essay"`，`text` 为参考答案全文
- `options`
  - 选择题：2-4 个 Option
  - 论述题 / 实验代码题：`[]`（空）
- `subgraph`：`nodes / edges`
- `provenance`：`supporting_triples / supporting_chunks / source_docs / retrieved_evidence_items`
- `grounding`：
  - `validation_mode`：`rule_only / hybrid_model / degraded`
  - `validation_status_detail`：`supported / insufficient_evidence / contradicted`
  - `double_checked`
  - `evidence_strength / claim_strength`
  - `support_score / model_confidence`
- `quality`：
  - `validation_status`
  - `model_verdict / model_rejection_reasons`
  - `essay_score`：论述题 LLM 评分（0.0-1.0）
  - `essay_rationale`：论述题 LLM 评分理由
  - `validator_version`
- `metadata`：
  - `claim_text / claim_text_conservative / query_text`
  - `evidence_strength / claim_strength / relation_strength / hedge_score`
  - `preferred_question_type`
  - `task_family / research_direction / task_objective`
  - `data_code / main_code / incomplete_main_code / incomplete_functions`
  - `unit_tests`
  - `github_references`

### 10.2 `summary.json`
运行统计：triples / chunks / sampled_subgraphs / accepted_questions / validation / question_types / validation_mode

---

## 11. 题型对比总结

| 维度 | claim_choice | boolean_support | two_hop_tail | essay |
|---|---|---|---|---|
| 选项数 | 4 | 2 | 4 | 0（开放式） |
| 答案形式 | claim 文本 | Supported/Not supported | 中间实体 | 自由文本参考答案 |
| evidence_strength 要求 | 所有 | >= medium | strong only | 所有 |
| 正确性判定 | 精确匹配 | 精确匹配 | 精确匹配 | LLM-as-Judge |
| model validation | judge_claim | judge_claim | judge_claim | judge_essay |
| 输出特有字段 | — | — | — | essay_score, essay_rationale |
| 适合测什么 | 证据支持判别 | 事实验证 | 多跳推理 | 综合论述与证据引用 |

---

## 12. 当前限制与后续方向

### 当前限制
1. evidence profiler 仍是规则启发式，对 evidence 语义理解深度有限
2. `boolean_support` 目前只有正例，还没有系统构造 `Not supported` 负样本
3. essay 参考答案由模板组装，还没接 LLM 生成更自然的参考答案
4. PubMed 检索 query 已 evidence-aware，但还可继续精细化
5. essay 题的 LLM-as-Judge 目前只评估参考答案，还不能评估被测 LLM 的回答

### 后续方向
1. 接入 LLM rewrite 层，让题面和参考答案更自然
2. 构造 hard negatives 用于 `boolean_support` 和 `claim_choice`
3. 让 evidence profiler 与 model validation 联动
4. 扩展 essay 题的 LLM-as-Judge 能力，支持评估被测模型的实际回答
5. 扩展检索源（Crossref / preprint）
6. 增加更复杂题型（evidence chain completion / mechanism selection）

---

## 13. 结论

当前系统已从最初的 relation-first 模板生成器，演进为：

> **evidence-first scientific benchmark generator + hybrid model validation framework + LLM-as-Judge essay evaluation**

核心能力链路：

```
evidence profiling → evidence-gated type selection → evidence-first claim construction
    → question generation (choice / boolean / two_hop / essay)
    → rule-based hard validation
    → literature retrieval
    → intern-s1-pro model verdict / LLM-as-Judge
```

已在 `proteinlmbench_full_graph_v1`（434 triples / 8297 chunks）上验证通过，包括 claim_choice 和 essay 两类题型。
