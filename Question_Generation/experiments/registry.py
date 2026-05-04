from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pubmed_graph.utils import normalize_text


@dataclass(frozen=True)
class ExperimentBlueprint:
    """A self-contained life-science experiment template.

    `incomplete_functions` are blanked at medium difficulty (and above).
    `hard_extra_blanks` are *additionally* blanked at hard difficulty —
    typically the orchestration / summarize_* function — so the model has
    to redesign the whole pipeline rather than just translate a formula.
    """

    name: str
    task_family: str
    relation: str
    direction: str
    discipline: str
    function_type: str
    task_objective: str
    research_focus: str
    data_code_template: str
    main_code_template: str
    incomplete_functions: tuple[str, ...]
    github_repo_query: str
    github_code_query: str
    unit_tests: tuple[dict[str, object], ...]
    hard_extra_blanks: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlueprintContext:
    head: str
    head_type: str
    relation: str  # already normalized: lowercase, underscored
    tail: str
    tail_type: str
    evidence: str = ""
    difficulty: str = "medium"


BlueprintPredicate = Callable[[BlueprintContext], bool]
BlueprintFactory = Callable[[BlueprintContext], ExperimentBlueprint]


@dataclass
class _RegistryEntry:
    name: str
    predicate: BlueprintPredicate
    factory: BlueprintFactory
    priority: int


class BlueprintRegistry:
    """Predicate-driven dispatch for experiment blueprints.

    Lower `priority` runs first. The first entry whose predicate returns True
    wins. If nothing matches, the registered fallback is used; if no fallback
    exists, dispatch raises.
    """

    def __init__(self) -> None:
        self._entries: list[_RegistryEntry] = []
        self._fallback: BlueprintFactory | None = None
        self._fallback_name: str = ""

    def register(
        self,
        name: str,
        predicate: BlueprintPredicate,
        factory: BlueprintFactory,
        priority: int = 100,
    ) -> None:
        self._entries.append(
            _RegistryEntry(name=name, predicate=predicate, factory=factory, priority=priority)
        )
        self._entries.sort(key=lambda entry: entry.priority)

    def register_fallback(self, name: str, factory: BlueprintFactory) -> None:
        self._fallback = factory
        self._fallback_name = name

    def dispatch(self, context: BlueprintContext) -> tuple[str, ExperimentBlueprint]:
        for entry in self._entries:
            if entry.predicate(context):
                return entry.name, entry.factory(context)
        if self._fallback is None:
            raise RuntimeError(
                f"No experiment blueprint matched for relation={context.relation!r} "
                f"and no fallback is registered"
            )
        return self._fallback_name, self._fallback(context)

    @property
    def registered_names(self) -> list[str]:
        return [entry.name for entry in self._entries] + (
            [self._fallback_name] if self._fallback else []
        )


REGISTRY = BlueprintRegistry()


def normalize_relation(relation: str) -> str:
    return normalize_text(relation).replace(" ", "_").casefold()
