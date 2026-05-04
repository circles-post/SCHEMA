from __future__ import annotations

from .difficulty import (
    DEFAULT_DIFFICULTY,
    VALID_DIFFICULTIES,
    normalize_difficulty,
    select_blank_targets,
)
from .registry import (
    REGISTRY,
    BlueprintContext,
    BlueprintRegistry,
    ExperimentBlueprint,
    normalize_relation,
)

# Importing blueprints triggers their REGISTRY.register(...) side-effects.
from . import blueprints  # noqa: F401  (registration side-effect)


def dispatch_blueprint(context: BlueprintContext) -> tuple[str, ExperimentBlueprint]:
    return REGISTRY.dispatch(context)


__all__ = [
    "REGISTRY",
    "BlueprintContext",
    "BlueprintRegistry",
    "ExperimentBlueprint",
    "DEFAULT_DIFFICULTY",
    "VALID_DIFFICULTIES",
    "dispatch_blueprint",
    "normalize_difficulty",
    "normalize_relation",
    "select_blank_targets",
]
