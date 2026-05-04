% Appendix — One curated, error-checked example per Bench-A question type.
%
% Picks were screened against the supporting_chunks evidence quote, not just the
% KG triple. Samples whose triple direction was reversed relative to the
% evidence sentence were rejected (e.g., qg_000007 where the triple says
% "p300 inhibits lactylation of YY1" but the evidence says "Inhibiting p300
% reduces lactylation"). Samples where the claim verb and evidence verb did
% not match (e.g., qg_001142 "gp120 inhibits macrophages" vs. evidence
% "target macrophages") were also rejected.
%
% Source file:
% benchmark_runs/protein_plus_pathvqa_500_v3/qgen_paired_enhanced_v2/samples_balanced.jsonl

\section{Bench-A Question-Type Examples}
\label{app:bench_a_examples}

Bench~A (\texttt{paired\_enhanced\_v2\_balanced}) contains six question types: \texttt{claim\_choice}, \texttt{two\_hop\_tail}, \texttt{vqa}, \texttt{boolean\_support}, \texttt{essay}, and \texttt{experiment\_code}. We display one representative, evidence-verified sample per type. For each multichoice item the correct option is marked with a $\bigstar$; the supporting passage shown below the question is the verbatim chunk from the source paper that justifies the labelled answer.

\bigskip

\begin{tcolorbox}[
    breakable,
    colframe=black!75!black,
    colback=gray!10!white,
    colbacktitle=gray!30!white,
    title=\textbf{Example A.1 --- \texttt{claim\_choice}\quad(\texttt{qg\_000048})},
    coltitle=black,
    boxrule=0.3mm,
    rounded corners,
    top=6pt, bottom=6pt, left=6pt, right=6pt,
    fonttitle=\bfseries,
    before upper={\parindent15pt}
]

\textbf{Source paper:} \textit{Cancer cachexia: molecular mechanisms and treatment strategies} (PMID:37217930).

\vspace{0.5em}
\textbf{Question.} Based on the reported evidence from the paper above, which candidate claim is most directly supported by the provided evidence?

\vspace{0.5em}
\textbf{Options.}

\hspace{10pt}\hspace{1em}A. The evidence supports a contextual relationship in which AMPK$\alpha$1 activates PHD2.

\hspace{10pt}\hspace{1em}B. The evidence supports a contextual relationship in which GST--hnRNP E1 activates translational signal.

\hspace{10pt}\hspace{1em}C. The evidence supports a contextual relationship in which Heterogeneous nuclear ribonucleoprotein E1 activates p53 and p21.

\hspace{10pt}\hspace{1em}$\bigstar$ D. The evidence supports a contextual relationship in which Cytokines activates NF-$\kappa$B.

\vspace{0.5em}
\textbf{Supporting passage.} ``\textit{Cytokines from tumors and immune cells induce activation of transcription factor NF-$\kappa$B, leading to UPS and ALS activation, which leads to muscle wasting.}''

\vspace{0.5em}
\textbf{Underlying triple.} \texttt{(Cytokines, activates, NF-$\kappa$B)} with confidence $0.95$.

\end{tcolorbox}

\bigskip

\begin{tcolorbox}[
    breakable,
    colframe=black!75!black,
    colback=gray!10!white,
    colbacktitle=gray!30!white,
    title=\textbf{Example A.2 --- \texttt{two\_hop\_tail}\quad(\texttt{qg\_000802\_\_d2})},
    coltitle=black,
    boxrule=0.3mm,
    rounded corners,
    top=6pt, bottom=6pt, left=6pt, right=6pt,
    fonttitle=\bfseries,
    before upper={\parindent15pt}
]

\textbf{Source paper:} \textit{Peripheral Vascular Calcification} (PMID:41537264).

\vspace{0.5em}
\textbf{Question.} Based on the reported evidence from the paper above, which intermediate pathway is best supported by the evidence chain in which Niclosamide inhibits an intermediate entity, and that intermediate entity participates in peripheral vascular calcification?

\vspace{0.5em}
\textbf{Options.}

\hspace{10pt}\hspace{1em}A. AKT/mTOR signaling pathway

\hspace{10pt}\hspace{1em}$\bigstar$ B. Wnt signaling pathway

\hspace{10pt}\hspace{1em}C. AKT signaling pathways

\hspace{10pt}\hspace{1em}D. AKT signaling pathway

\vspace{0.5em}
\textbf{Supporting passage.} ``\textit{Inhibiting Wnt signaling, exemplified by the use of Niclosamide, has been shown to effectively suppress the expression of osteogenic genes, including Runx2, and reduces vascular calcification.}''

\vspace{0.5em}
\textbf{Underlying triple chain.} \texttt{(Niclosamide, inhibits, Wnt signaling pathway)} $\rightarrow$ \texttt{(Wnt signaling pathway, reduces, vascular calcification)}; both hops are visible in the single sentence above.

\end{tcolorbox}

\bigskip

\begin{tcolorbox}[
    breakable,
    colframe=black!75!black,
    colback=gray!10!white,
    colbacktitle=gray!30!white,
    title=\textbf{Example A.3 --- \texttt{vqa}\quad(\texttt{qg\_000533})},
    coltitle=black,
    boxrule=0.3mm,
    rounded corners,
    top=6pt, bottom=6pt, left=6pt, right=6pt,
    fonttitle=\bfseries,
    before upper={\parindent15pt}
]

\textbf{Image.} A pathology micrograph from PathVQA (item \texttt{902}); a binucleate Reed--Sternberg cell with a mixed inflammatory background is the classical histologic finding of \emph{classical Hodgkin lymphoma}.

\vspace{0.5em}
\textbf{Question.} Is a diagnostic, binucleate Reed--Sternberg cell surrounded by eosinophils, lymphocytes, and histiocytes? (Answer Yes or No.)

\vspace{0.5em}
\textbf{Options.}

\hspace{10pt}\hspace{1em}$\bigstar$ A. Yes

\hspace{10pt}\hspace{1em}B. No

\vspace{0.5em}
\textbf{Rationale.} The combination of a binucleate Reed--Sternberg cell with a polymorphous reactive infiltrate of eosinophils, lymphocytes, and histiocytes is the textbook histopathological signature of classical Hodgkin lymphoma~\citep{stein2008who} and is unambiguously diagnostic on H\&E.

\end{tcolorbox}

\bigskip

\begin{tcolorbox}[
    breakable,
    colframe=black!75!black,
    colback=gray!10!white,
    colbacktitle=gray!30!white,
    title=\textbf{Example A.4 --- \texttt{boolean\_support}\quad(\texttt{qg\_000999})},
    coltitle=black,
    boxrule=0.3mm,
    rounded corners,
    top=6pt, bottom=6pt, left=6pt, right=6pt,
    fonttitle=\bfseries,
    before upper={\parindent15pt}
]

\textbf{Source paper:} \textit{CYLD Limits Neutrophil-Driven Psoriatic Inflammation} (\textit{Inflammation} 2026, \texttt{10.1007/s10753-026-02452-3}).

\vspace{0.5em}
\textbf{Question.} Based on the reported evidence from the paper above, is the following scientific claim supported by the provided evidence set: \textit{The reported evidence suggests that CYLD is upregulated in psoriasis}?

\vspace{0.5em}
\textbf{Options.}

\hspace{10pt}\hspace{1em}$\bigstar$ A. Supported

\hspace{10pt}\hspace{1em}B. Not supported

\vspace{0.5em}
\textbf{Supporting passage.} ``\textit{The results showed that CYLD expression was significantly upregulated in lesional skin of psoriasis patients; CYLD$^{-/-}$ mice displayed more severe psoriasiform symptoms.}''

\vspace{0.5em}
\textbf{Underlying triple.} \texttt{(CYLD, upregulated\_in, Psoriasis)} with confidence $0.89$.

\end{tcolorbox}

\bigskip

\begin{tcolorbox}[
    breakable,
    colframe=black!75!black,
    colback=gray!10!white,
    colbacktitle=gray!30!white,
    title=\textbf{Example A.5 --- \texttt{essay}\quad(\texttt{qg\_000131})},
    coltitle=black,
    boxrule=0.3mm,
    rounded corners,
    top=6pt, bottom=6pt, left=6pt, right=6pt,
    fonttitle=\bfseries,
    before upper={\parindent15pt}
]

\textbf{Source paper:} \textit{Infection-Associated Thymic Atrophy} (PMID:34113341).

\vspace{0.5em}
\textbf{Question.} Based on the reported evidence from the paper above, explain the relationship between Vpu and the release of virion, and discuss the strength of the supporting findings.

\vspace{0.5em}
\textbf{Reference answer.} The reported evidence suggests that Vpu enhances the release of virion. Supporting evidence: \textit{The Vpu protein of HIV-1 enhances the release of virion from infected cells.}

\vspace{0.5em}
\textbf{Supporting passage.} ``\textit{The Vpu protein of HIV-1 enhances the release of virion from infected cells. However, the Simian--Human Immunodeficiency Virus with the deletion of Vpu sequence (novpuSHIV KU-1bMC33) can still result in severe thymic atrophy in macaques.}''

\vspace{0.5em}
\textbf{Underlying triple.} \texttt{(Vpu, enhances, release of virion)}.

\end{tcolorbox}

\bigskip

\begin{tcolorbox}[
    breakable,
    colframe=black!75!black,
    colback=gray!10!white,
    colbacktitle=gray!30!white,
    title=\textbf{Example A.6 --- \texttt{experiment\_code}\quad(\texttt{qg\_999001})},
    coltitle=black,
    boxrule=0.3mm,
    rounded corners,
    top=6pt, bottom=6pt, left=6pt, right=6pt,
    fonttitle=\bfseries,
    before upper={\parindent15pt}
]

\textbf{Scientific claim.} \textit{Insulin-like growth factor 2 upregulates Myogenic differentiation 1.}

\vspace{0.5em}
\textbf{Research direction.} IGF-II upregulates MyoD expression in myoblast differentiation. Compute the Pearson correlation between IGF-II and MyoD expression levels and return a support flag.

\vspace{0.5em}
\textbf{Evidence summary.} ``\textit{IGF-II plays a direct role in the differentiation of mesoderm into myoblasts by upregulating the expression of MyoD, a key factor determining musculoskeletal fate.}'' (PMID:35741015)

\vspace{0.5em}
\textbf{Provided synthetic data} (\texttt{data\_en.py}, deterministic positive-correlation fixture):

\hspace{10pt}\texttt{IGF2 = [1.0, 2.0, 3.0, 4.0, 5.0]}

\hspace{10pt}\texttt{MyoD = [2.0, 4.0, 6.0, 8.0, 10.0]}

\vspace{0.5em}
\textbf{Skeleton to complete} (\texttt{main\_en.py}):

\hspace{10pt}\texttt{def compute\_correlation(data):}

\hspace{20pt}\texttt{\#\# Pearson correlation between data['IGF2'] and data['MyoD']}

\hspace{20pt}\texttt{pass  \#\# [Please complete the code]}

\vspace{0.5em}
\hspace{10pt}\texttt{def summarize\_igf2\_myo\_upregulation():}

\hspace{20pt}\texttt{data = load\_igf2\_myo\_data()}

\hspace{20pt}\texttt{corr = compute\_correlation(data)}

\hspace{20pt}\texttt{return \{'correlation': corr, 'supported': corr > 0\}}

\vspace{0.5em}
\textbf{Reference solution.} A standard Pearson correlation: centre both vectors, divide their dot product by the product of their L2 norms; under the supplied fixture this returns $\rho = 1.0$ and \texttt{supported} = \texttt{True}, consistent with the evidence sentence.

\end{tcolorbox}
