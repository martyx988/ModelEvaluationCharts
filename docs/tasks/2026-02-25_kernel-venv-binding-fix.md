# Task Contract

## Goal
Fix notebook kernel visibility/binding by setting up a dedicated project virtual environment kernel and binding the testing notebook to it.

## Acceptance Criteria
- Project `.venv` exists with all notebook prerequisites installed.
- Jupyter kernel is registered for `.venv` and visible in kernel list.
- Testing notebook metadata points to the `.venv` kernel.
- Core project functions run successfully with `.venv` Python.

## Non-Goals
- Removing or altering unrelated existing kernels.
- CI automation for environment provisioning.

## Assumptions / Open Questions
- Assumption: Python 3.11 is preferred for this project environment.

# Strategic Plan
- Create and provision `.venv` with notebook requirements.
- Register a dedicated kernelspec for `.venv`.
- Bind notebook metadata to that kernelspec.
- Verify via kernel listing and execution smoke test.

# Tactical Plan
- [x] Create `.venv` with Python 3.11.
- [x] Install dependencies from `requirements-notebook.txt` into `.venv`.
- [x] Register kernel `modelevaluationcharts-venv`.
- [x] Update notebook metadata to use `.venv` kernel.
- [x] Run smoke execution of `create_simulated_tables` and `EvaluateModel`.
- [x] Update setup script to reproduce this `.venv` kernel setup.

# Architecture Notes
- Kernel identifier: `modelevaluationcharts-venv`
- Kernel display name: `Python (ModelEvaluationCharts .venv)`
- Notebook binding file: `notebooks/simulated_data_testing.ipynb`
- Setup script now provisions `.venv` first, then installs and registers kernel.

# Test Plan

## Automated tests (what/where)
- Not in this slice.

## Manual verification script
- Run:
  - `powershell -ExecutionPolicy Bypass -File scripts/setup_notebook_kernel.ps1`
- Open notebook:
  - `notebooks/simulated_data_testing.ipynb`
- Confirm kernel shown as `Python (ModelEvaluationCharts .venv)`.
- Run all cells and verify report generation succeeds.

# Progress Log
- 2026-02-25: Verified pre-existing kernel state and interpreter inventory.
- 2026-02-25: Created project `.venv` and installed all notebook dependencies.
- 2026-02-25: Registered new kernel `modelevaluationcharts-venv`.
- 2026-02-25: Updated notebook kernelspec to `.venv` kernel.
- 2026-02-25: Smoke check passed for simulated data and report generation using `.venv`.
- 2026-02-25: Updated setup script for reproducible `.venv` kernel provisioning.

# Final Summary
Kernel binding issue addressed by introducing a dedicated project `.venv` kernel and wiring the testing notebook directly to it.
