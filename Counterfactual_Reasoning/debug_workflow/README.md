# Debug Workflow - 分步调试指南

逐步测试各模块的每个环节。每一步通过后再测下一步，快速定位问题。

## Part A: evaluate_autogen_life_science.py (Step 1-7)

```bash
cd /mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger

# Step 1: 数据集加载 (无需 Agent 依赖)
python debug_workflow/step1_dataset_loading.py

# Step 2: Prompt 构建 (无需 Agent 依赖)
python debug_workflow/step2_prompt_building.py

# Step 3: 答案提取与归一化 (纯逻辑，无外部依赖)
python debug_workflow/step3_answer_extraction.py

# Step 4: Agent 团队构建 (需要 autogen + API key)
python debug_workflow/step4_agent_build.py

# Step 5: 单条推理 (完整调用一次 Agent)
python debug_workflow/step5_single_inference.py

# Step 6: 评估指标计算 (无需 Agent 依赖)
python debug_workflow/step6_evaluate_metric.py

# Step 7: 端到端测试 (完整跑 2 条数据)
python debug_workflow/step7_end_to_end.py
```

## Part B: run_dataset_autodebug.py + external_agent_controller.py (Step 8-14)

```bash
cd /mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger

# ===== 离线测试 (不需要 backend) =====

# Step 8: 外部数据集导入 (DATASETS_DIR 路径、browse_bio_graph 模块)
python debug_workflow/step8_dataset_import.py

# Step 9: format_task / normalize_answer / extract_answer 逻辑
python debug_workflow/step9_format_and_normalize.py

# ===== 在线测试 (需要先启动 AGDebugger backend) =====
# 启动 backend:
#   python -m agdebugger.cli run test_agent_debug:get_agent_team --port 8081

# Step 10: AGDebuggerClient 连接和 API 调用
python debug_workflow/step10_agdebugger_client.py

# Step 11: 通过 backend 运行单个问题
python debug_workflow/step11_run_single_question.py --component-id <ID>

# ===== 在线测试 (还需要 LLM API key) =====
# export AGENTDEBUG_OPENAI_API_KEY=xxx
# export AGENTDEBUG_OPENAI_BASE_URL=xxx  (可选)
# export AGENTDEBUG_MODEL_NAME=xxx       (可选, 默认 gpt-4o-mini)

# Step 12: LLMPlanner 初始化 + 生成/执行单个 action
python debug_workflow/step12_llm_debug_loop.py

# Step 13: external_agent claim 分析管线
python debug_workflow/step13_claim_analysis.py

# Step 14: 端到端 autodebug (2 条数据)
python debug_workflow/step14_end_to_end_autodebug.py --component-id <ID>
```

## 依赖关系总览

| Step | 需要 Backend | 需要 LLM API | 测试目标 |
|------|:-----------:|:------------:|---------|
| 1-3, 6 | - | - | evaluate 脚本的数据处理 |
| 4-5, 7 | - | autogen API | evaluate 脚本的 Agent 推理 |
| 8-9 | - | - | autodebug 脚本的数据处理 |
| 10 | AGDebugger | - | Client 连通性 |
| 11 | AGDebugger | - | 单题运行 |
| 12 | AGDebugger | OpenAI API | LLM debug planner |
| 13 | AGDebugger | OpenAI API | Claim 分析管线 |
| 14 | AGDebugger | OpenAI API (可选) | 完整 autodebug 流程 |

## 推荐调试顺序

**Part A**: `1 → 2 → 3 → 6` (数据侧) → `4 → 5 → 7` (Agent 侧)

**Part B**: `8 → 9` (离线) → `10 → 11` (backend) → `12 → 13 → 14` (LLM debug)
