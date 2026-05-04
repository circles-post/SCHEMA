# Experiment Code Agent Workflow

## Goal

Generate GitHub-assisted life-science experiment coding questions from grounded scientific triples.

## Pipeline

```text
grounded triple + chunk evidence
        ↓
evidence profiler
        ↓
relation → experiment template family
        ↓
GitHub repository search
GitHub code search
        ↓
experiment spec assembly
  ├─ research direction
  ├─ task objective
  ├─ data_code
  ├─ main_code
  ├─ incomplete_main_code
  └─ unit_tests
        ↓
rule-based validation
        ↓
export question sample
```

## Current Task Families

- biomarker screening
- differential expression
- drug response
- pathway activity

## Design Constraints

- code must be grounded in the sampled scientific claim
- GitHub search is used as reference retrieval, not for verbatim copying
- each task blanks only 1-2 key functions
- prompts must include enough local context to solve the task without browsing again
- output should remain benchmark-friendly and easy to auto-grade later
