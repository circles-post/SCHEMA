# 当前 5 种题型的完整出题流程

> 目的：把每种题型的生成路径拆到"决策点"粒度，方便决定哪些点要插 agent、agent 的职责是什么。
>
> 所有引用的 `file:line` 对应当前 repo 状态（`experiments/` 包重构 + difficulty 三档之后的版本）。

---

## 0. 共享前后置（所有题型一定走）

```
load_triples / load_chunks                     io.py:9 / io.py:13
        ↓
build_index                                    indexing.py:25
   生成: chunks_by_id
         triples_by_key       (head,rel,tail) → list[TripleRecord]
         outgoing_by_head     entity → outgoing triples
         incoming_by_tail
         entities_by_type     type → set(entity_name)
        ↓
sample_single_hop_subgraphs / sample_two_hop_subgraphs   sampler.py:11 / 60
   过滤: avg_confidence ≥ min_confidence (默认 0.7)
         support_count   ≥ min_support    (默认 2)
   遍历每个 (head,rel,tail) 组,按请求的 question_types 各造一份候选 SampledSubgraph
        ↓
profile_subgraph_evidence                      evidence_profiler.py:53
   计算: relation_strength   ∈ {weak, medium, strong}   (relation 字面分类)
         hedge_score         (扫 evidence 里 "may/suggest/associate..." 等 hedge marker)
         evidence_strength   ∈ {weak, medium, strong}
         allowed_question_types  (强证据→所有题型,中等→去掉 two_hop_tail,弱→只剩 claim_choice/essay)
        ↓
                ────── 进入 build_question_sample ──────         generator.py:78
        ↓                                                       (5 个题型在这里分叉)
        ↓
validate_sample                                validator.py:213
   rule guardrails:
     • _evidence_supported       triple.evidence 必须是 chunk.text 子串      validator.py:35
     • _supports_minimum_double_check  ≥2 evidence / 2 docs / 2 chunks      validator.py:45
     • _answer_unique            (only two_hop_tail)                       validator.py:57
     • _answer_not_leaked        question 不能包含 answer 字面量             validator.py:80
     • _question_type_allowed_by_evidence                                  validator.py:86
     • _experiment_metadata_complete (only experiment_code)                validator.py:90
   if hybrid_model and 非 experiment_code:
     • model_validator.judge_claim / judge_essay                           model_validator.py:21 / 76
        ↓
deduplicate_by_question        dedup.py:10        (按 question 文本字面去重)
        ↓
export_samples                 exporter.py:11
```

---

## 1. `claim_choice` —— 选择题

```
SampledSubgraph (single edge)
        │
        ▼
[A] question 文本                                templates.py:42 / 66-69
    prefix     ← 第一条 supporting_chunk.title    templates.py:33
    模板     ← 二选一 (强证据 vs 弱证据 写死的 f-string):
                "...which candidate claim is most directly supported..."
                "...which candidate statement is most cautiously supported..."
        │
        ▼
[B] 正确答案                                    generator.py:113
    target_answer ← edge.tail                    subgraph_builder.py:71
    correct_text  ← _claim_option_text(...)      generator.py:19
                    再按 evidence_strength 三档套 claim 模板:
                      strong:  "The available evidence supports that {h} {rel} {t}."
                      medium:  "The evidence supports a contextual relationship..."
                      weak:    "The evidence suggests a reported association..."
        │
        ▼
[C] distractors                                 generator.py:33-61
    1) 同 head + 同 relation 但 tail≠正确 tail    (查 outgoing_by_head)
    2) 不够再从全图 同 relation 的其他 (head,tail) 拿
    每个 distractor 都被套进 [B] 的同一个模板,只换 head/tail
        │
        ▼
[D] Answer 对象                                 generator.py:115
    Answer(text=correct_text, answer_type="Claim")
        │
        ▼
[E] options = [correct] + distractors[:3]
        │
        ▼
[F] grounding/quality 字段(全是从 supporting_triples 数 出来的)
    difficulty = "medium" if support_count>1 else "easy"
        │
        ▼
[G] validate_sample
    • evidence 必须是 chunk 子串
    • multi-source 检查
    • options ≥ 2 (cli.py:128 后置过滤)
```

**LLM 调用次数: 0**(rule_only) / 1 次 `judge_claim` 复核(hybrid_model)

---

## 2. `boolean_support` —— 支持性判断题

```
SampledSubgraph (single edge)
        │
        ▼
[A] question 文本                                templates.py:70-72
    stem ← claim_text_conservative              evidence_claims.py:16-34
            (按 evidence_strength 三档写死的句子,见上面 [B] 模板)
    模板 ← 写死:
        "...is the following scientific claim supported by the provided
         evidence set: {stem}?"
        │
        ▼
[B] 正确答案                                    generator.py:103
    硬编码 Answer(text="Supported", answer_type="Boolean")
        │
        ▼
[C] options                                     generator.py:103
    硬编码 [Option("Supported", True), Option("Not supported", False)]
    ⚠ 当前只产正例,没有反例(不存在生成 "Not supported" 标签的样本)
        │
        ▼
[D] grounding/quality (同上)
        │
        ▼
[E] validate_sample
    • _question_type_allowed_by_evidence 要求 evidence_strength ≥ medium
      → 弱证据的 boolean 候选会被 evidence_profiler 直接滤掉(allowed_question_types 里没有它)
```

**LLM 调用次数: 0** / 1 次 `judge_claim` 复核

---

## 3. `essay` —— 论述题

```
SampledSubgraph (single edge)
        │
        ▼
[A] question 文本                                templates.py:49-65
    根据 evidence_strength 三档写死的 f-string:
      weak:   "...describe the reported relationship between {h} and {t},
               and explain what the available evidence indicates..."
      medium: "...explain the relationship between {h} and {t}, and discuss
               the strength of the supporting findings."
      strong: "...explain the mechanism by which {h} {rel} {t}, citing the
               supporting evidence."
        │
        ▼
[B] 参考答案                                    generator.py:106-111
    reference_text ← claim_text_conservative    evidence_claims.py
                     (一句模板化的 claim 句子)
    if 有 evidence_snippets:
        reference_text += " Supporting evidence: " + 前 2 条 triple.evidence 拼接
    Answer(text=reference_text, answer_type="Essay")
    ⚠ 这就是"参考答案"的全部 —— 不是真正的解释
        │
        ▼
[C] options = []   (论述题没有选项)
        │
        ▼
[D] grounding/quality
        │
        ▼
[E] validate_sample
    rule 阶段同上
    if hybrid_model:
        model_validator.judge_essay()           model_validator.py:76
        → 让 LLM 对 (question, reference_answer, evidence_bundle)
           打 essay_score ∈ [0,1] 和 essay_rationale
           写入 sample.quality.essay_score / essay_rationale
        ⚠ 这是评分,不是改写参考答案
```

**LLM 调用次数: 0** / 1 次 `judge_essay` 评分(hybrid_model)

---

## 4. `two_hop_tail` —— 两跳链补全题

```
sample_two_hop_subgraphs                        sampler.py:60
   遍历每个 first_hop key:
     pivot ← first_hop[0].tail
     second_hops ← outgoing_by_head[pivot.casefold()]   ⚠ 已知 bug:漏 normalize
     按 (h2,rel2,t2) group second hop
     两端 avg_confidence 都 ≥ min_confidence
        │
        ▼
build_two_hop_subgraph                          subgraph_builder.py:97
   nodes = [A, B(pivot), C]
   edges = [(A,rel1,B), (B,rel2,C)]
   target_answer = B (中间实体,即 left.tail)
   target_answer_type = B 的类型
        │
        ▼
[A] question 文本                                templates.py:73-85
    模板 (强证据):
      "...which intermediate {target_type.lower()} is best supported by the
       evidence chain in which {A} {rel1_phrase} an intermediate entity,
       and that intermediate entity {rel2_phrase} {C}?"
    模板 (弱证据): "most plausibly implicated" 替代 "best supported"
        │
        ▼
[B] 正确答案                                    generator.py:117
    answer_text ← target_answer (即 B)
    Answer(text=B, answer_type=B 的类型)
        │
        ▼
[C] distractors                                 generator.py:63-75
    从 entities_by_type[B 的类型] 里挑 ≠ B 的 entity
    ⚠ 完全不验证候选实体在图里是否构成同样的二跳链 —— 只看类型对得上
        │
        ▼
[D] options = [B] + distractors[:3]
        │
        ▼
[E] validate_sample
    • _answer_unique (验证器)                  validator.py:57-77
      重新跑一遍二跳查询,确保给定 (A,rel1,?,rel2,C) 的中间实体在图中唯一是 B
      → 如果 KG 里有第二个合法 pivot,这条样本会被拒
    • _question_type_allowed_by_evidence
      → 只允许 evidence_strength=strong
```

**LLM 调用次数: 0** / 1 次 `judge_claim` 复核(hybrid_model)

---

## 5. `experiment_code` —— 实验代码题

```
SampledSubgraph (single edge, sampler 已按 difficulty 复制 1/3 份)
        │  (sampler.py:49-67  按 --experiment-difficulty 给同一条 edge 复制
        │   easy/medium/hard 三份候选,uniqueness_key 后缀加 |difficulty=...)
        ▼
build_experiment_sample                         experiment_generator.py:129
        │
        ▼
[A] BlueprintContext                            experiment_generator.py:143
    head, head_type, relation, tail, tail_type, evidence, difficulty
    (head_type/tail_type 取 supporting_triples[0])
        │
        ▼
[B] dispatch_blueprint                          experiments/registry.py:90
    按 priority 顺序跑每个 blueprint 的 predicate(纯 if 判断 relation):
       priority 10  biomarker_screening      relation == associated_with
       priority 20  differential_expression  relation ∈ {up/down/overexpressed_in}
       priority 30  dose_response            relation ∈ {inhibits/improves/promotes}
       fallback     pathway_activity         其它
    返回 (blueprint_name, ExperimentBlueprint)
    ⚠ 4 个 blueprint 的 main_code/data_code/unit_tests 全是预先写死的 Python 源码字符串
        │
        ▼
[C] select_blank_targets(blueprint, difficulty)  experiments/difficulty.py:18
    easy   → incomplete_functions[:1]            (1 个待填函数)
    medium → incomplete_functions                (2 个待填函数,= 旧默认行为)
    hard   → incomplete_functions ∪ hard_extra_blanks  (再挖掉 summarize_*)
        │
        ▼
[D] _build_incomplete_code                      experiment_generator.py:15
    字符串扫描找 "def {fn}(" → 保留 signature + docstring
    → 函数体替换为 "    pass  # [Please complete the code]"
        │
        ▼
[E] _build_github_reference_pack                experiment_generator.py:65
    blueprint.github_repo_query (写死的 f-string)  → search_github_repositories  (REST API)
    blueprint.github_code_query (写死的 const)     → search_github_code          (REST API)
    都过 lru_cache;失败时 status="degraded"
    ⚠ 不是 agent,只是两次 requests.get
        │
        ▼
[F] _render_experiment_prompt                   experiment_generator.py:85
    一个大 f-string 把以下塞进固定 XML-like sections:
      <scientific_claim>     ← head + relation + tail
      <research_direction>   ← blueprint.research_focus + task_objective + 1-2 evidence
      <agent_workflow>       ← 5 步固定文字 + github_status
      <github_repository_search>  ← [E] 的结果原样贴
      <github_code_search>        ← [E] 的结果原样贴
      <data_code>            ← blueprint.data_code_template (完整,不挖空)
      <main_code>            ← [D] 挖空后的版本
        │
        ▼
[G] Answer                                      experiment_generator.py:171
    Answer(text=完整 main_code, answer_type="Code")     ← 标准答案 = blueprint 模板
        │
        ▼
[H] options = []                                experiment_generator.py:218
        │
        ▼
[I] metadata 写入完整字段(供 validator 检查 + 下游用)
    data_code, main_code, incomplete_main_code, incomplete_functions,
    unit_tests, github_references, task_objective, research_direction,
    experiment_blueprint, experiment_difficulty, task_family, ...
        │
        ▼
[J] validate_sample
    • _experiment_metadata_complete  必须包含上面 8 个 key   validator.py:90
    • rule 阶段同上
    • ⚠ 不走 model 验证(experiment_code 在 validate_sample 里被显式短路)
                                                 validator.py:225-228
```

**LLM 调用次数: 0**（无论 rule_only 还是 hybrid_model 都是 0,因为 experiment_code 在 validator 里被显式跳过）

---

## 横向对比表 —— 每个"决策点"目前是谁在做

| 决策点 | claim_choice | boolean_support | essay | two_hop_tail | experiment_code |
|---|---|---|---|---|---|
| **subgraph 选哪些** | sampler 规则(置信度+支持数) | 同左 | 同左 | sampler 规则 + 二跳遍历 | 同左 + 难度复制 |
| **题型是否合法** | evidence_profiler 阈值 | 阈值(≥medium) | 阈值 | 阈值(=strong) | 阈值 |
| **题面措辞** | 2 个写死模板 | 1 个写死模板 | 3 个写死模板 | 2 个写死模板 | 1 个大 f-string |
| **claim/stem 措辞** | 3 档写死 | 3 档写死 | 3 档写死 | — | — |
| **正确答案** | edge.tail → 模板 | 硬编码 "Supported" | claim 句 + evidence 拼接 | edge.tail (pivot) | blueprint 写死的 main_code |
| **distractor 来源** | 图遍历(同 head 同 rel) | 硬编码 "Not supported" | 无 | entities_by_type 同类型池 | 无 |
| **distractor 措辞** | 套同一个 claim 模板 | — | — | 实体名原样 | — |
| **代码题数据** | — | — | — | — | blueprint 写死的 CSV |
| **代码题挖空** | — | — | — | — | 字符串扫描 def + 三档规则 |
| **代码题外部引用** | — | — | — | — | GitHub REST API(非 agent) |
| **代码题 unit test** | — | — | — | — | blueprint 写死期望值 |
| **rule 验证** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **model 验证(可选)** | judge_claim | judge_claim | judge_essay(打分) | judge_claim | ❌ 显式跳过 |

---

## "决策点 → 当前实现 → 你可以告诉我要不要换 Agent" 速查表

| # | 决策点 | 当前实现 | 文件:行 | 题型 |
|---|---|---|---|---|
| ① | subgraph 候选筛选 | 置信度 + 支持数阈值 | sampler.py:23-29 | 全部 |
| ② | evidence_strength 判定 | relation 字典 + hedge keyword | evidence_profiler.py:36-66 | 全部 |
| ③ | 题型 gating | 阈值表 | evidence_profiler.py:68-77 | 全部 |
| ④ | claim 文本三档措辞 | f-string | evidence_claims.py:16-34 | claim_choice/boolean/essay |
| ⑤ | question 句子模板 | f-string | templates.py:42-86 | 全部 |
| ⑥ | claim_choice distractor 选取 | 图遍历 | generator.py:33-61 | claim_choice |
| ⑦ | claim_choice distractor 措辞 | 套 claim 模板 | generator.py:19-25 | claim_choice |
| ⑧ | boolean 反例生成 | **无（只产正例）** | generator.py:103 | boolean_support |
| ⑨ | essay 参考答案 | claim + evidence 拼接 | generator.py:107-111 | essay |
| ⑩ | two_hop distractor 选取 | 同类型实体池 | generator.py:63-75 | two_hop_tail |
| ⑪ | two_hop pivot 唯一性检查 | 重跑图查询 | validator.py:57-77 | two_hop_tail |
| ⑫ | experiment_code blueprint 选 | predicate(只看 relation) | experiments/registry.py:90 | experiment_code |
| ⑬ | experiment_code 数据生成 | blueprint 写死 CSV | experiments/blueprints/*.py | experiment_code |
| ⑭ | experiment_code main_code | blueprint 写死源码 | experiments/blueprints/*.py | experiment_code |
| ⑮ | experiment_code 挖空策略 | def 扫描 + 三档规则 | experiment_generator.py:15 | experiment_code |
| ⑯ | experiment_code 外部引用 | GitHub REST API(2 次 query) | experiment_generator.py:65 | experiment_code |
| ⑰ | experiment_code unit_test | blueprint 写死期望值 | experiments/blueprints/*.py | experiment_code |
| ⑱ | experiment_code 提交执行 | **无(从未跑过 main_code)** | — | experiment_code |
| ⑲ | rule 验证 | 子串/计数检查 | validator.py:35-103 | 全部 |
| ⑳ | model 评委(可选) | judge_claim/judge_essay | model_validator.py | 全部除 experiment_code |

---

## 用法：你给我一份"agent 注入清单"

请按这个格式告诉我哪些点要换/增加 agent，我直接动手：

```
# 决策点 ⑨ essay 参考答案
模式: 替换 (replace) | 加层 (augment-after-rule)
Agent 职责:
  输入: question 文本 + 全部 supporting evidence + claim_text
  输出: 一段 3-5 句的机制解释,只允许引用 evidence 中已出现的实体
  约束: 不引入新实体名;长度 ≤ 200 词
可用工具: 无 / PubMedClient / GitHub / 自定义
失败回退: 退回原 rule-based 拼接
验证: judge_essay 复核 essay_score ≥ 0.6 才接受
```

每条都写清楚 **替换 vs 加层**、**输入/输出/约束**、**失败回退** —— 这三件事是后续实现绕不开的。
