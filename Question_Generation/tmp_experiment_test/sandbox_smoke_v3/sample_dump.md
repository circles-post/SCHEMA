# experiment_code sample dump

Source: `question_samples.jsonl` (2 samples)

This file is a *human-readable* dump of one experiment_code sample so you can
see exactly what gets injected into the prompt vs. what lives in `metadata`.

---

## Sample `qg_000001`

- **question_type**: `experiment_code`
- **edge**: `MDM2 inhibits TP53`
- **blueprint**: `dose_response` (dose_response)
- **difficulty**: `medium`
- **blanked functions**: `['compute_activity_drop', 'estimate_ic50']`

### Validation

- `quality.validation_status`: **passed**
- `quality.validator_version`: `experiment_sandbox_v1`
- `grounding.validation_mode`: **sandbox**
- `grounding.validation_status_detail`: `sandbox_passed`
- `grounding.support_score` (= reference unit-test pass-rate): `1.0`
- `grounding.contradiction_count` (= incomplete unit-tests that wrongly pass): `0`

### Sandbox evaluation

- **verdict**: `passed`
- **reference run**: `2/2 passed` (status=ok)
  - ✓ `activity_drop`  actual = `{"drug": "MDM2", "target": "TP53", "activity_drop": 0.9158, "ic50_um": 1.0}`
  - ✓ `ic50`  actual = `{"drug": "MDM2", "target": "TP53", "activity_drop": 0.9158, "ic50_um": 1.0}`
- **incomplete run**: `0/2 passed` (status=ok)
  - ✗ `activity_drop`  err = `TypeError: type NoneType doesn't define __round__ method`
  - ✗ `ic50`  err = `TypeError: type NoneType doesn't define __round__ method`

### GitHub reference fetch

- **status**: `ok`
- **repo_query**: `MDM2 TP53 dose response python`
- **code_query**: `estimate ic50 dose response python`

#### Selected code excerpts (LLM function-level extraction)

##### Excerpt 1: `snap-stanford/Biomni` :: `biomni/tool/biochemistry.py`

- source: <https://github.com/snap-stanford/Biomni/blob/400c1f366b96a35ca253e13c9b06c5076af41d65/biomni/tool/biochemistry.py>
- file size: 41579 bytes (fetch_truncated=False)
- selection method: **llm**  (6 top-level funcs in file)
- LLM rationale: _analyze_enzyme_kinetics_assay directly deals with dose-dependent response analysis and could provide a suitable framework for computing activity drops and estimating IC50 values._

###### `analyze_enzyme_kinetics_assay` (lines 458-667) *(truncated)*
_Performs in vitro enzyme kinetics assay and analyzes the dose-dependent effects of modulators._

```python
def analyze_enzyme_kinetics_assay(
    enzyme_name,
    substrate_concentrations,
    enzyme_concentration,
    modulators=None,
    time_points=None,
    output_dir="./",
):
    """Performs in vitro enzyme kinetics assay and analyzes the dose-dependent effects of modulators.

    Parameters
    ----------
    enzyme_name : str
        Name of the purified enzyme being tested
    substrate_concentrations : list or numpy.ndarray
        List of substrate concentrations in μM for kinetic analysis
    enzyme_concentration : float
        Concentration of the enzyme in nM
    modulators : dict, optional
        Dictionary of modulators where keys are modulator names and values are lists of
        concentrations in μM. Default is None (no modulators).
    time_points : list or numpy.ndarray, optional
        Time points in minutes for time-course measurements. Default is None, which uses
        [0, 5, 10, 15, 20, 30, 45, 60].
    output_dir : str, optional
        Directory to save output files. Default is current directory.

    Returns
    -------
    str
        Research log summarizing the enzyme kinetics assay procedure and results

    """
    import csv
    import os

    import numpy as np
    from scipy.optimize import curve_fit

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set default time points if not provided
    if time_points is None:
        time_points = np.array([0, 5, 10, 15, 20, 30, 45, 60])
    else:
        # Ensure time_points is a numpy array
        time_points = np.array(time_points)

    # Initialize research log
    log = f"## In Vitro Enzyme Kinetics Assay: {enzyme_name}\n\n"
    log += f"Enzyme concentration: {enzyme_concentration} nM\n"

    # Michaelis-Menten equation for curve fitting
    def michaelis_menten(s, vmax, km):
        return vmax * s / (km + s)

    # 1. Time-course kinetic assay
    log += "\n### Time-Course Kinetic Assay\n\n"
    log += "Measuring enzyme activity over time to establish linear range.\n"

    # Simulate time-course data (realistic enzyme kinetics with some noise)
    # Using a simple exponential approach to equilibrium model
    max_activity = 100  # arbitrary units
    rate_constant = 0.05  # min^-1

    # Simulate enzyme activity over time with some noise
    np.random.seed(42)  # For reproducibility
    time_course_activity = max_activity * (1 - np.exp(-rate_constant * time_points))
    time_course_activity += np
```

##### Excerpt 2: `sunnivass/rainbow` :: `app/ic50_mic_app.py`

- source: <https://github.com/sunnivass/rainbow/blob/f63490308d07d844b9697d4070e22f1926656b38/app/ic50_mic_app.py>
- file size: 22516 bytes (fetch_truncated=False)
- selection method: **llm**  (6 top-level funcs in file)
- LLM rationale: _The body of _compute_inhibition_percent directly implements a dose-response activity calculation, which is the closest conceptual match to compute_activity_drop and can be adapted to compute the activity drop metric._

###### `_compute_inhibition_percent` (lines 99-119)
_(no docstring)_

```python
def _compute_inhibition_percent(
    y_pred: np.ndarray,
    *,
    mode: str,
    control_y: float,
    top: float,
    bottom: float,
) -> np.ndarray | None:
    if mode == "control":
        denom = abs(control_y)
        if denom <= 1e-9:
            return None
        return 100.0 * (control_y - y_pred) / denom

    if mode == "range_norm":
        dyn = top - bottom
        if abs(dyn) <= 1e-9:
            return None
        return 100.0 * (top - y_pred) / dyn

    return None
```

#### Repo tree summary

- repo: `snap-stanford/Biomni` @ `main`
- entries (filtered to .py): 20

  - `biomni/__init__.py`  (60 bytes)
  - `biomni/agent/__init__.py`  (45 bytes)
  - `biomni/agent/a1.py`  (131378 bytes)
  - `biomni/agent/env_collection.py`  (13813 bytes)
  - `biomni/agent/function_generator.py`  (4298 bytes)
  - `biomni/agent/qa_llm.py`  (1754 bytes)
  - `biomni/agent/react.py`  (20927 bytes)
  - `biomni/biorxiv_scripts/extract_biorxiv_tasks.py`  (12663 bytes)
  - `biomni/biorxiv_scripts/generate_function.py`  (2201 bytes)
  - `biomni/biorxiv_scripts/process_all_subjects.py`  (10081 bytes)

---

### Generated experiment specification (the actual question shown to the model)

```text
Please read the following experiment specification and complete the missing functions in `main_en.py`. The implementation should be grounded in the scientific evidence. Use the GitHub reference material below as inspiration rather than copying code verbatim — the code excerpts have been pulled from real public repositories matching the task topic.

<scientific_claim>
MDM2 inhibits TP53
</scientific_claim>

<research_direction>
The evidence suggests that MDM2 is reported to inhibits TP53. Construct a compact experimental analysis task around inhibitory response estimation.
Task objective: Implement the core response metrics for a dose-response study involving MDM2 and TP53.
Evidence summary: MDM2 inhibits TP53 MDM2 inhibits TP53
</research_direction>

<agent_workflow>
1. Read the scientific evidence and task objective.
2. Skim the GitHub code excerpts and repository layout to identify implementation patterns.
3. Inspect `data_en.py` to understand the synthetic experiment inputs.
4. Fill in only the missing functions in `main_en.py`.
5. Ensure the implementation is numerically stable and aligned with the intended scientific computation.
GitHub reference status: ok
</agent_workflow>

<github_repository_search>
No repositories found for query: MDM2 TP53 dose response python
</github_repository_search>

<github_code_search>
GitHub code search results for 'estimate ic50 dose response python':

1. biochemistry.py
   Repository: snap-stanford/Biomni
   Path: biomni/tool/biochemistry.py
   URL: https://github.com/snap-stanford/Biomni/blob/400c1f366b96a35ca253e13c9b06c5076af41d65/biomni/tool/biochemistry.py

2. ic50_mic_app.py
   Repository: sunnivass/rainbow
   Path: app/ic50_mic_app.py
   URL: https://github.com/sunnivass/rainbow/blob/f63490308d07d844b9697d4070e22f1926656b38/app/ic50_mic_app.py

3. filter-cdd-export.py
   Repository: choderalab/perses
   Path: examples/moonshot-mainseries/molecules/filter-cdd-export.py
   URL: https://github.com/choderalab/perses/blob/c716ba936fff992d596a342daebe4248597d287d/examples/moonshot-mainseries/molecules/filter-cdd-export.py

Total code results found: 70
</github_code_search>

<github_code_excerpts>
### snap-stanford/Biomni :: biomni/tool/biochemistry.py
# source: https://github.com/snap-stanford/Biomni/blob/400c1f366b96a35ca253e13c9b06c5076af41d65/biomni/tool/biochemistry.py
# selection: method=llm, total_functions_in_file=6, rationale="analyze_enzyme_kinetics_assay directly deals with dose-dependent response analysis and could provide a suitable framework for computing activity drops and estimating IC50 values."
# snap-stanford/Biomni :: biomni/tool/biochemistry.py  L458-667  function: analyze_enzyme_kinetics_assay
```python
def analyze_enzyme_kinetics_assay(
    enzyme_name,
    substrate_concentrations,
    enzyme_concentration,
    modulators=None,
    time_points=None,
    output_dir="./",
):
    """Performs in vitro enzyme kinetics assay and analyzes the dose-dependent effects of modulators.

    Parameters
    ----------
    enzyme_name : str
        Name of the purified enzyme being tested
    substrate_concentrations : list or numpy.ndarray
        List of substrate concentrations in μM for kinetic analysis
    enzyme_concentration : float
        Concentration of the enzyme in nM
    modulators : dict, optional
        Dictionary of modulators where keys are modulator names and values are lists of
        concentrations in μM. Default is None (no modulators).
    time_points : list or numpy.ndarray, optional
        Time points in minutes for time-course measurements. Default is None, which uses
        [0, 5, 10, 15, 20, 30, 45, 60].
    output_dir : str, optional
        Directory to save output files. Default is current directory.

    Returns
    -------
    str
        Research log summarizing the enzyme kinetics assay procedure and results

    """
    import csv
    import os

    import numpy as np
    from scipy.optimize import curve_fit

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set default time points if not provided
    if time_points is None:
        time_points = np.array([0, 5, 10, 15, 20, 30, 45, 60])
    else:
        # Ensure time_points is a numpy array
        time_points = np.array(time_points)

    # Initialize research log
    log = f"## In Vitro Enzyme Kinetics Assay: {enzyme_name}\n\n"
    log += f"Enzyme concentration: {enzyme_concentration} nM\n"

    # Michaelis-Menten equation for curve fitting
    def michaelis_menten(s, vmax, km):
        return vmax * s / (km + s)

    # 1. Time-course kinetic assay
    log += "\n### Time-Course Kinetic Assay\n\n"
    log += "Measuring enzyme activity over time to establish linear range.\n"

    # Simulate time-course data (realistic enzyme kinetics with some noise)
    # Using a simple exponential approach to equilibrium model
    max_activity = 100  # arbitrary units
    rate_constant = 0.05  # min^-1

    # Simulate enzyme activity over time with some noise
    np.random.seed(42)  # For reproducibility
    time_course_activity = max_activity * (1 - np.exp(-rate_constant * time_points))
    time_course_activity += np
# ... [function body truncated]
```

### sunnivass/rainbow :: app/ic50_mic_app.py
# source: https://github.com/sunnivass/rainbow/blob/f63490308d07d844b9697d4070e22f1926656b38/app/ic50_mic_app.py
# selection: method=llm, total_functions_in_file=6, rationale="The body of _compute_inhibition_percent directly implements a dose-response activity calculation, which is the closest conceptual match to compute_activity_drop and can be adapted to compute the activity drop metric."
# sunnivass/rainbow :: app/ic50_mic_app.py  L99-119  function: _compute_inhibition_percent
```python
def _compute_inhibition_percent(
    y_pred: np.ndarray,
    *,
    mode: str,
    control_y: float,
    top: float,
    bottom: float,
) -> np.ndarray | None:
    if mode == "control":
        denom = abs(control_y)
        if denom <= 1e-9:
            return None
        return 100.0 * (control_y - y_pred) / denom

    if mode == "range_norm":
        dyn = top - bottom
        if abs(dyn) <= 1e-9:
            return None
        return 100.0 * (top - y_pred) / dyn

    return None
```
</github_code_excerpts>

<github_repo_tree>
snap-stanford/Biomni @ main (default=main)
------------------------------------------
  biomni/__init__.py  (1 KB)
  biomni/agent/__init__.py  (1 KB)
  biomni/agent/a1.py  (129 KB)
  biomni/agent/env_collection.py  (14 KB)
  biomni/agent/function_generator.py  (5 KB)
  biomni/agent/qa_llm.py  (2 KB)
  biomni/agent/react.py  (21 KB)
  biomni/biorxiv_scripts/extract_biorxiv_tasks.py  (13 KB)
  biomni/biorxiv_scripts/generate_function.py  (3 KB)
  biomni/biorxiv_scripts/process_all_subjects.py  (10 KB)
  biomni/config.py  (4 KB)
  biomni/env_desc.py  (24 KB)
  biomni/env_desc_cm.py  (25 KB)
  biomni/eval/__init__.py  (1 KB)
  biomni/eval/biomni_eval1.py  (12 KB)
  biomni/know_how/__init__.py  (1 KB)
  biomni/know_how/loader.py  (12 KB)
  biomni/llm.py  (11 KB)
  biomni/model/__init__.py  (0 KB)
  biomni/model/retriever.py  (9 KB)
</github_repo_tree>

<data_code>
#!/usr/bin/env python3
"""Synthetic dose-response table for MDM2 against TP53."""

from io import StringIO

import pandas as pd


CSV_TEXT = """dose_um,response
0.01,0.95
0.1,0.82
1.0,0.51
10.0,0.19
50.0,0.08"""


def load_dose_response() -> pd.DataFrame:
    return pd.read_csv(StringIO(CSV_TEXT))

</data_code>

<main_code>
#!/usr/bin/env python3
"""Estimate inhibitory response metrics for MDM2 and TP53."""

import numpy as np

from data_en import load_dose_response


def compute_activity_drop(responses: np.ndarray) -> float:
    """
    Measure the fractional activity drop from the first to the last dose.
    """
    pass  # [Please complete the code]

def estimate_ic50(doses: np.ndarray, responses: np.ndarray) -> float:
    """
    Estimate the IC50 by selecting the dose whose response is closest to 50% activity.
    """
    pass  # [Please complete the code]

def summarize_drug_response() -> dict[str, float]:
    table = load_dose_response()
    doses = table["dose_um"].to_numpy(dtype=float)
    responses = table["response"].to_numpy(dtype=float)
    activity_drop = compute_activity_drop(responses)
    ic50 = estimate_ic50(doses, responses)
    return {
        "drug": "MDM2",
        "target": "TP53",
        "activity_drop": round(activity_drop, 4),
        "ic50_um": round(ic50, 4),
    }


if __name__ == "__main__":
    print(summarize_drug_response())

</main_code>

```

### Reference answer (`answer.text`, full main_code)

```python
#!/usr/bin/env python3
"""Estimate inhibitory response metrics for MDM2 and TP53."""

import numpy as np

from data_en import load_dose_response


def compute_activity_drop(responses: np.ndarray) -> float:
    """
    Measure the fractional activity drop from the first to the last dose.
    """
    baseline = float(responses[0])
    final = float(responses[-1])
    return float((baseline - final) / max(baseline, 1e-8))


def estimate_ic50(doses: np.ndarray, responses: np.ndarray) -> float:
    """
    Estimate the IC50 by selecting the dose whose response is closest to 50% activity.
    """
    idx = int(np.argmin(np.abs(responses - 0.5)))
    return float(doses[idx])


def summarize_drug_response() -> dict[str, float]:
    table = load_dose_response()
    doses = table["dose_um"].to_numpy(dtype=float)
    responses = table["response"].to_numpy(dtype=float)
    activity_drop = compute_activity_drop(responses)
    ic50 = estimate_ic50(doses, responses)
    return {
        "drug": "MDM2",
        "target": "TP53",
        "activity_drop": round(activity_drop, 4),
        "ic50_um": round(ic50, 4),
    }


if __name__ == "__main__":
    print(summarize_drug_response())

```

### `metadata.unit_tests`

```json
[
  {
    "name": "activity_drop",
    "input": {},
    "expected_output": {
      "activity_drop": 0.9158
    }
  },
  {
    "name": "ic50",
    "input": {},
    "expected_output": {
      "ic50_um": 1.0
    }
  }
]
```

### `metadata.data_code` (synthetic dataset module)

```python
#!/usr/bin/env python3
"""Synthetic dose-response table for MDM2 against TP53."""

from io import StringIO

import pandas as pd


CSV_TEXT = """dose_um,response
0.01,0.95
0.1,0.82
1.0,0.51
10.0,0.19
50.0,0.08"""


def load_dose_response() -> pd.DataFrame:
    return pd.read_csv(StringIO(CSV_TEXT))

```

### `metadata.incomplete_main_code` (the prompt's masked main_en.py)

```python
#!/usr/bin/env python3
"""Estimate inhibitory response metrics for MDM2 and TP53."""

import numpy as np

from data_en import load_dose_response


def compute_activity_drop(responses: np.ndarray) -> float:
    """
    Measure the fractional activity drop from the first to the last dose.
    """
    pass  # [Please complete the code]

def estimate_ic50(doses: np.ndarray, responses: np.ndarray) -> float:
    """
    Estimate the IC50 by selecting the dose whose response is closest to 50% activity.
    """
    pass  # [Please complete the code]

def summarize_drug_response() -> dict[str, float]:
    table = load_dose_response()
    doses = table["dose_um"].to_numpy(dtype=float)
    responses = table["response"].to_numpy(dtype=float)
    activity_drop = compute_activity_drop(responses)
    ic50 = estimate_ic50(doses, responses)
    return {
        "drug": "MDM2",
        "target": "TP53",
        "activity_drop": round(activity_drop, 4),
        "ic50_um": round(ic50, 4),
    }


if __name__ == "__main__":
    print(summarize_drug_response())

```
