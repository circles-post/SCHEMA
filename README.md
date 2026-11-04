# SCHEMA

**Tracing the Cascade: A Topology-Aware Evaluation Framework for Scientific Agent Hallucinations**

Xinshun Feng, Ziqi Miao, Lijun Li, and Jing Shao

Beihang University · Shanghai Artificial Intelligence Laboratory

[Code](https://github.com/circles-post/SCHEMA) · [Dataset](https://huggingface.co/datasets/XsF2001/SCHEMA)

SCHEMA evaluates the reliability of scientific AI agents beyond their final answers. It constructs scientific concept graphs from literature evidence, generates graph-grounded questions, and measures hallucinations in intermediate reasoning. A complementary counterfactual analysis workflow investigates selected failures by intervening on agent trajectories.

## Repository structure

| Directory | Purpose | Main entry point |
| --- | --- | --- |
| `Benchmark_Construction/` | Literature retrieval, evidence extraction, concept normalization, and graph construction | `literature_pipeline.py` |
| `Question_Generation/` | Question generation, validation, and export | `cli.py` |
| `Evaluation/` | Agent execution and answer scoring | `runner.py` |
| `Evaluation/halu/` | Claim extraction, evidence gathering, hallucination judging, and aggregation | `cli.py` |
| `Counterfactual_Reasoning/` | Claim analysis and AGDebugger-based trajectory interventions | `external_agent/cli.py`, `run_dataset_autodebug.py` |

The examples below describe the interfaces in this source snapshot. Full agent evaluation requires external model and retrieval services. Counterfactual replay also requires additional components identified below. The AutoGen dependencies are pinned to the version used for offline release checks; other dependencies are not fully locked. The complete original experiment artifacts are not bundled, and current APIs may not reproduce the paper's numbers exactly.

## Environment setup

Use Python 3.10 or later and a Linux/macOS shell. Run the following from the repository root (the directory containing this README).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[evaluation,data]'
export SCHEMA_ROOT="$(pwd)"
```

The editable installation exposes the lowercase Python packages used by the source without renaming directories or creating import aliases. Additional dependencies depend on the workflow:

| Workflow | Additional requirements |
| --- | --- |
| Local BGE/SapBERT embeddings | `torch`, `transformers`, and the configured model weights; alternatively configure the supported remote embedding service |
| PDF parsing | One of the configured backends: `pdfplumber`, `pypdf`, or `pymupdf` |
| Graph visualization / benchmark download utilities | `matplotlib`, `pandas`, `datasets`, `pyarrow`, as needed by the selected utility |
| Experimental code execution | `sandbox_fusion` client and a running compatible sandbox service |
| Optional ToolUniverse tools | `tooluniverse` and the dependencies of the enabled tools |
| AGDebugger backend | Dependencies listed in `Counterfactual_Reasoning/pyproject.toml`, plus the missing components described in the counterfactual section |

### Command-line entry points

After installation, these commands are available from any working directory:

```bash
schema-evaluate --help
schema-hallucinations --help
schema-build-graph --help
schema-generate --help
```

Run the examples below from the repository root so their relative data paths resolve. `Evaluation/scripts/run_eval.sh` and `run_bench.sh` now locate the checkout automatically and accept `BENCH_A_DATASET` / `BENCH_B_DATASET` overrides; `run_bench.sh` also accepts `BENCH_A_GRAPH` / `BENCH_B_GRAPH`. Other historical launch scripts may still contain author-specific path or service placeholders. Prefer the direct CLI examples unless you have configured a particular launcher.

## Dataset

The public benchmark is hosted at [XsF2001/SCHEMA on Hugging Face](https://huggingface.co/datasets/XsF2001/SCHEMA).

| Subset | Questions | Concept records |
| --- | ---: | ---: |
| Protein Domain | 800 | 1,276 |
| PathVQA-Enhanced | 495 | 811 |

Both released subsets are text-only and cover `claim_choice`, `boolean_support`, `two_hop_tail`, `essay`, and `experiment_code` questions. The dataset uses **CC BY 4.0**.

### Download and prepare evaluation inputs

The public JSONL files encode nested fields as JSON strings. The evaluator now accepts both this public format and the internally generated dictionary/list format, decoding the nested fields automatically. Download the question files directly:

```bash
python - <<'PY'
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

for subset in ("protein_domain", "pathvqa_enhanced"):
    source = hf_hub_download(
        repo_id="XsF2001/SCHEMA",
        repo_type="dataset",
        filename=f"{subset}/questions.jsonl",
    )
    target = Path("data") / subset / "samples.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(target)
PY
```

Keep the dataset revision used for each experiment. Concept inventories and per-question subgraphs do not replace the full `global_graph.graphml` used by graph-weighted hallucination evaluation. Supply the matching graph snapshot when computing topology-based results.

## Configure services

Copy `Evaluation/.env.example` to `Evaluation/.env` if you do not already have a local configuration. Fill in the values needed for your workflow. Keep credentials out of version control.

| Variable | Used for |
| --- | --- |
| `BOYUE_BASE_URL`, `BOYUE_API_KEY` | OpenAI-compatible endpoint for evaluated agents; the variable names are historical |
| `JUDGE_MODEL`, `JUDGE_BASE_URL`, `JUDGE_API_KEY` | Final-answer scoring for essay questions |
| `INTERN_BASE_URL`, `INTERN_API_KEY` | Claim extraction and, when selected, hallucination judging |
| `BRIGHT_DATA_API_KEY`, `BRIGHT_DATA_ZONE` | Web retrieval |
| `SCIVERSE_API_TOKEN` | Literature retrieval through the external SciVerse adapter |
| `QG_SANDBOX_HOST`, `QG_SANDBOX_PORT` | Remote sandbox for experimental code questions |
| `EMBEDDING_SERVICE_URL` or `BGE_MODEL_PATH` | Remote embeddings or local model weights for graph matching |

The CLI loads its selected `--env-file` automatically. The hallucination command below also expands endpoint variables in the shell; for that example, use shell-compatible assignments and load the same file first:

```bash
set -a
source Evaluation/.env
set +a
```

The runner loads retrieval dependencies after dotenv configuration and refreshes sandbox settings before scoring. Shell sourcing is not required for a normal `evaluation.runner --env-file ...` invocation.

**External literature adapter:** tool-enabled evaluation needs `sciverse_tools.py`, which is not included in this checkout. Set `SCIVERSE_DIR` in `Evaluation/.env` to the directory containing the compatible adapter and install its dependencies. It must provide `literature_search` and `sciverse_fetch_markdown`. The web adapter defaults to the bundled `Counterfactual_Reasoning/websearch_tools.py`; `AGENTDEBUG_DIR` can override its location.

Closed-book evaluation (`--no-tools`) does not import or require these retrieval adapters. A tool-enabled run reports an actionable error if the SciVerse adapter is unavailable.

Agents use `BOYUE_BASE_URL` by default. Model-specific URLs can be configured with `BOYUE_MODEL_OVERRIDES="model_id=https://your-provider.example/v1,..."`. Intern-S1-Pro retains its separate endpoint/key route in `_DEFAULT_MODEL_OVERRIDES`; check that route when using an alternative provider. No placeholder provider URL is used for Doubao, Llama, or Grok.

## Run agent evaluation

After installation and service configuration, start with a small run:

```bash
python -m evaluation.runner \
  --dataset data/protein_domain/samples.jsonl \
  --models gpt-4o \
  --output-dir outputs/eval/protein_domain \
  --env-file Evaluation/.env \
  --use-tools \
  --no-tooluniverse \
  --max-messages 12 \
  --per-sample-timeout 600 \
  --sample-concurrency 1 \
  --limit 5
```

Replace the model ID with one available at your endpoint. `--no-tooluniverse` keeps literature/web retrieval enabled while disabling the optional ToolUniverse server. Use `--no-tools` for a closed-book diagnostic, which changes the evaluation setting. Remove `--limit 5` for a full run; use the PathVQA-Enhanced input and a separate output directory for the other subset.

Each model directory contains:

```text
outputs/eval/protein_domain/
├── combined_summary.json
└── <model>/
    ├── answers.jsonl
    ├── trajectory.jsonl
    ├── scored_results.jsonl
    └── summary.json
```

The runner supports resuming existing outputs. Use a fresh output directory when changing the dataset, model configuration, or experimental setting. Do not use `--no-emit-trajectory` if you will run hallucination evaluation.

Answer scoring is type-specific: objective questions use rule-based scoring, essays use the configured judge, and experimental code uses sandbox unit tests. Missing judge or sandbox services produce scoring errors; inspect `scored_results.jsonl` and summary error counts before interpreting scores. Trajectory exports intentionally elide tool-result bodies and truncate tool-call arguments; they are not complete raw API transcripts.

## Evaluate trajectory hallucinations

Use the same dataset and the agent outputs above. The example evaluates GPT-4o trajectories with Intern-S1-Pro as judge to avoid judging the model with itself; replace the extractor with another accessible model as appropriate.

```bash
python -m evaluation.halu.cli \
  --runs-dir outputs/eval/protein_domain \
  --dataset data/protein_domain/samples.jsonl \
  --models gpt-4o \
  --output-dir outputs/halu/protein_domain \
  --env-file Evaluation/.env \
  --extractor-model gpt-4o \
  --extractor-base-url "$BOYUE_BASE_URL" \
  --extractor-api-key-env BOYUE_API_KEY \
  --judge-model intern-s1-pro \
  --judge-base-url "$INTERN_BASE_URL" \
  --judge-api-key-env INTERN_API_KEY \
  --evidence-chain graph,web,literature \
  --graph /absolute/path/to/protein_domain/global_graph.graphml \
  --include-correct \
  --limit 5
```

Provide a real matching graph path. `--include-correct` includes both successful and failed answers; the CLI otherwise analyzes failed answers only. Run models separately when they need different judges, because the CLI's self-judging guard applies to the selected model directories. The graph, web, and literature layers are tried after sample-provided evidence; unavailable retrieval services can leave claims without evidence.

Outputs include `<model>/halu_results.jsonl`, `<model>/halu_summary.json`, `combined_halu_summary.json`, and a cache directory. The implementation reports:

| Metric | Meaning |
| --- | --- |
| `HR` | Fraction of extracted claims judged refuted |
| `HS` | Mean claim severity; supported, unverifiable, and refuted claims receive different scores |
| `HS_weighted` | Severity weighted by matched concepts' graph connectivity |
| `HF` | Whether a trajectory contains at least one refuted claim |

Summary files distinguish macro and micro aggregation. Select the aggregation matching the paper's definition rather than treating these fields as interchangeable. Without graph matches, concept weights default to one and weighted severity reduces to unweighted severity. Keep graph snapshots, judge settings, caches, and the choice of analyzed trajectories with your results.

## Construct graphs and generate new questions

These steps generate new artifacts. Re-running literature retrieval against current services does not recreate a historical benchmark snapshot exactly.

For a small graph-construction run, configure `Benchmark_Construction/.env` from its example and inspect `pipeline_config.minimal.json`:

```bash
python Benchmark_Construction/literature_pipeline.py \
  --config Benchmark_Construction/pipeline_config.minimal.json \
  --output-dir outputs/graph_demo \
  --extract-triples \
  --chunk-limit 24 \
  --export-graph \
  --graph-output outputs/graph_demo/global_graph.graphml
```

This requires literature access and the configured extraction model. Outputs include `chunks.jsonl`, `normalized_triples.jsonl`, `phase_summary.json`, and the exported graph. The minimal configuration disables several optional retrieval/embedding features and is intended for a small workflow check.

Generate a small set of text questions from those outputs:

```bash
python -m question_generation.cli \
  --triples outputs/graph_demo/normalized_triples.jsonl \
  --chunks outputs/graph_demo/chunks.jsonl \
  --graph outputs/graph_demo/global_graph.graphml \
  --output outputs/question_demo/samples.jsonl \
  --summary-output outputs/question_demo/summary.json \
  --max-samples 20 \
  --question-types claim_choice boolean_support two_hop_tail essay \
  --validation-mode rule_only \
  --corroboration-mode off
```

This example deliberately uses rule-only validation for a workflow check; it is not the fully validated paper benchmark. For model validation, configure `--validation-mode hybrid_model`, `--validator-enabled`, and the validator endpoint/model. External corroboration uses `QG_AGENTDEBUG_PATH` and `QG_SCIVERSE_PATH` to locate the retrieval adapters. Experimental code generation additionally requires a working sandbox and the chosen generation mode. See the argument definitions in `Question_Generation/cli.py` for the full interface.

## Counterfactual analysis

`Counterfactual_Reasoning/` includes an AGDebugger-derived backend, claim extraction/judging, and intervention logic. It retains the upstream [README](Counterfactual_Reasoning/README.md) and [license](Counterfactual_Reasoning/LICENSE).

For standalone claim analysis, provide a text file containing an agent response and a separate evidence file, then run from the repository root with model credentials exported:

```bash
PYTHONPATH="$SCHEMA_ROOT/Counterfactual_Reasoning${PYTHONPATH:+:$PYTHONPATH}" \
python -m external_agent.cli \
  --task research_questions \
  --text-file /absolute/path/to/agent_response.txt \
  --evidence-file /absolute/path/to/supporting_evidence.txt \
  --model gpt-4o
```

This interface reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` or the corresponding `AGENTDEBUG_OPENAI_*` variables and prints extracted claims and judgments as JSON. It performs claim analysis; full intervention-and-replay experiments use `run_dataset_autodebug.py` and the backend.

**Full replay setup is incomplete in this snapshot:** `run_dataset_autodebug.py` imports `browse_bio_graph_cluster_examples.py` from `AGDEBUGGER_DATASETS_DIR`, but that helper and its original component-based dataset/graph inputs are not included. Supply the matching helper, dataset artifacts, and service configuration before using the batch replay workflow. The HF question files are not a direct replacement for these component-based inputs. The backend can be installed with `python -m pip install -e ./Counterfactual_Reasoning` and run without UI assets. The upstream README's frontend build instructions still require the absent `frontend/` source; a browser UI is not bundled.

## Reproducibility notes

- Use the released question subsets for benchmark evaluation, and record their revision, sample IDs, and graph version.
- Record the resolved model ID, provider endpoint, inference settings, tool configuration, timeout, and interaction limits. API defaults and provider behavior can change.
- Keep answer scores, trajectory outputs, hallucination judgments, and the configurations needed to interpret them. Original complete trajectories and judge caches are not bundled with this source checkout.
- Check errors and evidence availability before comparing aggregate scores. A run that completes with unavailable services is not equivalent to the configured experiment.
- Preserve raw experiment settings separately for the large-scale trajectory evaluation and the sampled counterfactual study; they use different entry points and inputs.

## Attribution and licensing

The dataset is distributed under **CC BY 4.0**, as specified on its [Hugging Face page](https://huggingface.co/datasets/XsF2001/SCHEMA). Dataset licensing does not license third-party code or underlying source publications.

The AGDebugger-derived component retains the Microsoft copyright notice and MIT license in [Counterfactual_Reasoning/LICENSE](Counterfactual_Reasoning/LICENSE). A repository-wide code license file is not included in this snapshot; the component license should not be treated as a blanket license for all directories.

SCHEMA builds on biomedical literature sources, upstream benchmark resources, AutoGen, and AGDebugger. See the paper for scientific attribution and the component documentation for upstream references.

## Citation

If you use SCHEMA, please cite the paper:

```bibtex
@misc{feng2026tracing,
  title  = {Tracing the Cascade: A Topology-Aware Evaluation Framework for Scientific Agent Hallucinations},
  author = {Xinshun Feng and Ziqi Miao and Lijun Li and Jing Shao},
  year   = {2026},
  url    = {https://github.com/circles-post/SCHEMA}
}
```
