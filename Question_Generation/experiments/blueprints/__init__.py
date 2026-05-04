from __future__ import annotations

# Import each blueprint module so it registers itself with the global REGISTRY.
# Order here is irrelevant — registration uses explicit `priority`.
from . import biomarker_screening  # noqa: F401
from . import differential_expression  # noqa: F401
from . import dose_response  # noqa: F401
from . import pathway_activity  # noqa: F401
