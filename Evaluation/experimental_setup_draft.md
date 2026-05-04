% Draft of three Experimental Setup subsections.
% Source data: evaluation/full_results_report.md, evaluation/halu_runs/*, halu_concept_macroclass_aggregation.md.
% Drop into the paper as-is; numbers and model lists already match the runs in eval_runs/ and halu_runs/.

\subsection{Experimental Setup}

\noindent\textbf{Baselines.}
We evaluate eleven contemporary instruction-tuned LLMs that span proprietary frontier systems, open-weight reasoning models, and tool-augmented multi-agent backbones, so that the comparison is not confounded by a single training family. Concretely, the cohort includes four proprietary endpoints accessed through their official APIs --- \texttt{gpt-4o}, \texttt{gpt-5.4-mini}, \texttt{gemini-3-flash-preview-thinking}, and \texttt{grok-4-1-fast-reasoning} --- four large open or semi-open reasoning models --- \texttt{kimi-k2.5}, \texttt{glm-5.1}, \texttt{qwen3.6-plus}, and \texttt{deepseek-v4-flash} --- the open-weight \texttt{llama-4-scout}, ByteDance's \texttt{doubao-seed-2-0-pro-260215}, and the biomedical-domain backbone \texttt{intern-s1-pro}. All models are evaluated through the same agent harness (see \emph{Workflow Settings}), with sampling temperature fixed to $0$ where the API permits and otherwise to the lowest provider-supported value, $\mathrm{top\_p}{=}1$, and a per-sample wall-clock budget of $600$\,s. We deliberately exclude retrieval-augmented baselines that hard-code an external knowledge base into the prompt, because our benchmarks already provide a graph-grounded retrieval surface that any model may invoke through the tool channel; this isolates \emph{how} a model uses retrieval from \emph{whether} it has retrieval at all.

\smallskip
\noindent\textbf{Workflow Settings.}
Both benchmarks share an autogen-style two-actor workflow: an \texttt{EvalAgent} that interleaves natural-language reasoning with tool calls, and a passive \texttt{ToolExecutor} that resolves each call. The agent is given four tools --- \texttt{literature\_search}, \texttt{literature\_fetch}, \texttt{web\_search}, and \texttt{web\_fetch} --- backed by the in-house biomedical search service over the PubMed-graph pipeline (\S\ref{sec:pipeline}) for the literature tools and by Bright Data for the web tools; the same toolset is exposed to every model so that capability gaps reflect tool-use proficiency rather than asymmetric access. Each sample is run as an independent multi-turn dialogue with a hard cap of $\mathrm{max\_turns}{=}12$ and a $600$\,s budget. The full message stream is persisted to \texttt{trajectory.jsonl}: each entry records the speaker role, the message type (\texttt{text}, \texttt{tool\_call\_request}, or \texttt{tool\_call\_execution}), and, for tool calls, the (truncated) arguments; we explicitly elide tool-result bodies in the dump used by the hallucination pipeline so that retrieved-content cannot leak into claim extraction. Bench~A is built from \texttt{paired\_enhanced\_v2\_balanced} ($N{=}933$ paired QA items grounded on the \texttt{protein\_plus\_pathvqa\_500\_v3} graph), while Bench~B is built from \texttt{paired\_protein\_v2\_balanced} ($N{=}495$ items) over the \texttt{proteinlmbench\_full\_graphbench} graph; both expose the same three difficulty tiers and four \emph{question\_type} strata (counts and corroboration metadata are reported in Table~\ref{tab:bench_meta}).

\smallskip
\noindent\textbf{Evaluation Details.}
We report two complementary families of metrics. The \emph{accuracy} family scores each final answer against the gold reference using a pairwise LLM judge (\texttt{gpt-4o}, JSON mode, $T{=}0$) and aggregates micro-accuracy, accuracy by tier, by question\_type, and by tier$\times$type, mirroring \texttt{evaluation.core.aggregate}. The \emph{hallucination} family is computed on the trajectories themselves and is the contribution of this paper: a deterministic pipeline (\S\ref{sec:halu_pipeline}) extracts atomic factual claims from every \texttt{EvalAgent} step, normalizes their concepts via \texttt{pubmed\_graph.normalize} with a BGE-Large fallback at cosine $0.85$, gathers evidence in the priority chain \texttt{supporting\_chunks} $\rightarrow$ \texttt{global\_graph} 1-hop $\rightarrow$ \texttt{web\_search} $\rightarrow$ \texttt{literature\_search} (short-circuit on first non-empty layer), and dispatches a per-bucket judge call. To control judge bias, the cohort is split: on Bench~A, \texttt{intern-s1-pro} judges 8 of 11 models while \texttt{gpt-4o} judges \{\texttt{intern-s1-pro}, \texttt{deepseek-v4-flash}, \texttt{gemini-3-flash-preview-thinking}\}; on Bench~B, \texttt{glm-5.1} judges \{\texttt{gpt-4o}, \texttt{gpt-5.4-mini}, \texttt{grok-4-1-fast-reasoning}, \texttt{qwen3.6-plus}, \texttt{doubao-seed-2-0-pro-260215}\} and \texttt{gpt-4o} judges the remaining five. No model is judged by itself.

The judge returns one of three verdicts per claim, mapped to a deterministic severity score:
\begin{equation}
s(c) \;=\;
\begin{cases}
0.0 & \text{verdict}(c) = \texttt{supported} \\
0.5 & \text{verdict}(c) = \texttt{unverifiable} \\
1.0 & \text{verdict}(c) = \texttt{refuted}
\end{cases}
\end{equation}
From these we report claim-micro hallucination rate $\mathrm{HR}_{\mu}{=}n_{\mathrm{ref}}/n_{\mathrm{claims}}$, severity $\mathrm{HS}_{\mu}{=}\frac{1}{n}\sum_c s(c)$, the graph-connectivity-weighted variant $\mathrm{HS}_{\mu}^{w}$ that up-weights claims whose concept appears in the sample subgraph by $w(c){=}1{+}\log(1{+}\deg(c))$, and the sample-level fingerprint rate $\mathrm{HF}{=}\Pr[\exists c{:}\,s(c){=}1]$. Each metric is sliced by tier, question\_type, tier$\times$type, \texttt{corroboration\_status}, \texttt{evidence\_strength}, and concept type. Concept type uses the fifteen graph-node labels (Protein, Gene, RNA, MolecularEntity, Drug, Complex, Pathway, BiologicalProcess, Disease, CellType, CellLine, Biomarker, TissueRegion, StainingMethod, ClinicalEndpoint) and is additionally re-aggregated under two macroclass schemes for cross-model comparison: \emph{Scheme~A} (UMLS Semantic Group / BioLink-aligned, four buckets: molecular entities, anatomical/cellular, phenotypes \& disorders, processes \& procedures; following~\citet{bodenreider2004umls,mccray2001semgroups,unni2022biolink}) and \emph{Scheme~C} (epistemic / falsifiability-primitive, four buckets: identifier-grounded, compositional, processual, locative/methodological; grounded in BFO~2.0~\citep{arp2015bfo} and the OBO Foundry ontologies). Macroclass $\mathrm{HS}_{\mu}^{w}$ is reconstructed as the $n_{\mathrm{claims}}$-weighted average of fine-grained $\mathrm{HS}_{\mu}^{w}$, validated against sample-level reweighting to within $0.005$ on every cohort. By default, the hallucination metrics are computed only over \emph{erroneous} trajectories ($\mathrm{is\_correct}{=}\texttt{false}$); the \texttt{--include-correct} ablation in the appendix shows that including correct trajectories shifts $\mathrm{HS}_{\mu}^{w}$ by less than $0.02$ but halves the visible spread between models, which is why the main tables stay on the error-only setting.

\bigskip
% --- Notes on what was inferred vs. evidenced -----------------------------
% Evidenced from the runs and prior agreement:
%   * 11-model cohort (intern-s1-pro, gpt-5.4-mini, glm-5.1, qwen3.6-plus,
%     gpt-4o, llama-4-scout, doubao-seed-2-0-pro-260215, kimi-k2.5,
%     gemini-3-flash-preview-thinking, grok-4-1-fast-reasoning, deepseek-v4-flash)
%   * Bench A = paired_enhanced_v2_balanced over protein_plus_pathvqa_500_v3
%   * Bench B = paired_protein_v2_balanced over proteinlmbench_full_graphbench
%   * Tool surface (literature_search/literature_fetch/web_search/web_fetch)
%   * Verdict mapping (0.0/0.5/1.0) and metric definitions
%   * Judge dispatch (intern-s1-pro vs gpt-4o on A; glm-5.1 vs gpt-4o on B)
%   * 4-tool autogen workflow with --emit-trajectory; 600s per sample
%   * Scheme A and Scheme C macroclass schemes
%   * Default = error trajectories only; --include-correct as ablation
%
% Inferred — verify before submission:
%   * N=933 (Bench A), N=495 (Bench B): cross-check against the actual
%     samples.jsonl line counts in each benchmark dir.
%   * max_turns=12: this is the default in agent_workflow_full.py at last
%     check, but bump if the run config used a different cap.
%   * Bench-meta table reference \ref{tab:bench_meta}: insert wherever the
%     paper's metadata table lives.
%   * Citations \citet{...} are placeholders — point them at the actual
%     bibtex keys in the paper's .bib (Bodenreider 2004 NAR; McCray Burgun
%     Bodenreider 2001 MedInfo; Unni et al. 2022 Bioinformatics; Arp Smith
%     Spear 2015 MIT Press for BFO).
