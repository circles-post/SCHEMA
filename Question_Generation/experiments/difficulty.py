from __future__ import annotations

from .registry import ExperimentBlueprint

VALID_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
DEFAULT_DIFFICULTY = "medium"


def normalize_difficulty(value: str | None) -> str:
    if value is None:
        return DEFAULT_DIFFICULTY
    lowered = value.strip().casefold()
    if lowered not in VALID_DIFFICULTIES:
        return DEFAULT_DIFFICULTY
    return lowered


def select_blank_targets(blueprint: ExperimentBlueprint, difficulty: str) -> tuple[str, ...]:
    """Pick which functions get blanked given the requested difficulty.

    - easy: only the first listed function (one focused fill-in-the-blank)
    - medium: every function in `incomplete_functions` (current default)
    - hard: every function in `incomplete_functions` plus `hard_extra_blanks`
            (typically the orchestration / summarize_* function)
    """
    difficulty = normalize_difficulty(difficulty)
    if difficulty == "easy":
        return blueprint.incomplete_functions[:1] or tuple(blueprint.incomplete_functions)
    if difficulty == "hard":
        seen: set[str] = set()
        ordered: list[str] = []
        for name in (*blueprint.incomplete_functions, *blueprint.hard_extra_blanks):
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return tuple(ordered)
    return tuple(blueprint.incomplete_functions)
