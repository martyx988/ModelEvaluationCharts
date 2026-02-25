# Task Contract

## Goal
Create a Jupyter notebook for manual testing of simulated data functionality, including calls to the data generator and visible table outputs, and ensure notebook kernel setup works.

## Acceptance Criteria
- A notebook file exists for manual testing.
- Notebook contains cells that:
  - import and call the simulation function
  - display `model_score`, `target_store`, and `situation`
  - show basic shape/range checks
- Data simulation logic enforces:
  - `score` in range `0..1`
  - `percentile` in range `1..100`
- Notebook metadata references a valid Python kernel for this project.

## Non-Goals
- Building Plotly report charts in this slice.
- Full automated notebook test harness.

## Assumptions / Open Questions
- Assumption: `.ipynb` was intended by user text `ipyng`.
- Assumption: local environment can register a user-level kernel.

# Strategic Plan
- Update simulation logic for strict percentile range.
- Add a manual testing notebook with clear execution cells.
- Register/validate project kernel for notebook execution.

# Tactical Plan
- [x] Update `simulated_data.py` percentile generation to `1..100`.
- [x] Create `notebooks/simulated_data_testing.ipynb`.
- [x] Add import, generation, display, and assertions in notebook cells.
- [x] Validate kernel availability in environment and register if missing.
- [x] Execute smoke checks via Python script equivalent to notebook assertions.

# Architecture Notes
- Simulation API remains unchanged: `create_simulated_tables(seed=42)`.
- Notebook imports from repo root by adjusting `sys.path` when launched from `notebooks/`.
- Kernel name chosen: `modelevaluationcharts` with display name `Python (ModelEvaluationCharts)`.

# Test Plan

## Automated tests (what/where)
- Planned follow-up: add pytest checks for range constraints and schema in `tests/test_simulated_data.py`.

## Manual verification script
- Open `notebooks/simulated_data_testing.ipynb` in VS Code/Jupyter.
- Select kernel `Python (ModelEvaluationCharts)`.
- Run all cells and confirm:
  - Shapes print as expected.
  - Table previews render.
  - Range prints are in expected bounds.
  - Final assertion cell prints `All notebook checks passed.`

# Progress Log
- 2026-02-25: Created task file for notebook manual testing slice.
- 2026-02-25: Updated percentile generation to inclusive `1..100`.
- 2026-02-25: Added notebook with generation, display, and validation cells.
- 2026-02-25: Installed `ipykernel`, registered kernel `modelevaluationcharts`, and verified kernel list.
- 2026-02-25: Ran smoke assertions confirming score range `0..1`, percentile range `1..100`, and expected table shapes.

# Final Summary
Notebook manual-test slice completed. Data generation now satisfies requested ranges, notebook is ready for local execution, and a working project kernel is registered.
