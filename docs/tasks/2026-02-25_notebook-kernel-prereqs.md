# Task Contract

## Goal
Set up notebook kernel prerequisites, libraries, and reproducible setup utilities so the testing notebook runs smoothly.

## Acceptance Criteria
- Required notebook/runtime libraries are installed in current Python environment.
- `Python (ModelEvaluationCharts)` kernel is available.
- Repo includes reproducible dependency file for notebook usage.
- Repo includes a script that installs dependencies and registers the kernel.
- Smoke check confirms imports and key project functions run.

## Non-Goals
- Full environment isolation via virtual environment tooling.
- CI pipeline setup.

## Assumptions / Open Questions
- Assumption: system Python is acceptable for current local workflow.

# Strategic Plan
- Install/verify missing notebook runtime libraries.
- Add reproducible setup files.
- Validate with an execution smoke test.

# Tactical Plan
- [x] Install missing packages (`notebook`, `ipywidgets`, `nbformat`).
- [x] Add `requirements-notebook.txt`.
- [x] Add `scripts/setup_notebook_kernel.ps1`.
- [x] Verify kernel exists and run import/function smoke check.

# Architecture Notes
- Kernel name remains `modelevaluationcharts` to match notebook metadata.
- Setup script is idempotent and can be rerun safely.

# Test Plan

## Automated tests (what/where)
- Not in this slice.

## Manual verification script
- Run:
  - `powershell -ExecutionPolicy Bypass -File scripts/setup_notebook_kernel.ps1`
- In notebook:
  - Select `Python (ModelEvaluationCharts)` kernel
  - Run all cells in `notebooks/simulated_data_testing.ipynb`

# Progress Log
- 2026-02-25: Installed notebook runtime dependencies and widget support.
- 2026-02-25: Added dependency lock file and kernel setup script.
- 2026-02-25: Verified kernel `modelevaluationcharts` is registered and runnable.
- 2026-02-25: Smoke check passed for `create_simulated_tables` and `EvaluateModel`.

# Final Summary
Notebook prerequisites are set and reproducible. The environment has required libraries, kernel registration is valid, and project functions execute successfully from Python.
