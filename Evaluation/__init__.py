"""Evaluation toolkit for question_generation outputs.

Typical use::

    from evaluation import score_one, score_many, aggregate, EvalResult

    # one sample at a time
    result = score_one(sample_dict, model_answer="A")

    # batch
    results = score_many(
        "samples.jsonl",
        answers={"qg_000001": "A", "qg_000002": 2, ...},
        judge_model_config={"model": "...", "base_url": "...", "api_key": "..."},
    )
    stats = aggregate(results)

See ``core.score_one`` for the model_answer shape expected per question_type.
"""

from .core import Evaluator, aggregate, score_many, score_one
from .drivers import run_and_score
from .types import EvalResult
from .vlm import route_info, run_vlm

__all__ = [
    "Evaluator", "EvalResult",
    "aggregate", "score_many", "score_one",
    "run_and_score",
    "run_vlm", "route_info",
]
