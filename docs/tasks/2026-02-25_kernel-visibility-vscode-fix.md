# Task Contract

## Goal
Ensure the testing notebook shows and uses this project's kernel in VS Code Jupyter kernel picker.

## Acceptance Criteria
- A clearly named project kernel is registered and visible in `jupyter kernelspec list`.
- Testing notebook metadata points to that exact kernel.
- Workspace settings pin interpreter to this project's `.venv`.
- Setup script provisions the same kernel name for reproducibility.

## Non-Goals
- Managing kernels in other repositories.
- Removing unrelated kernels from user environment.

## Assumptions / Open Questions
- Assumption: user is opening this repository workspace (`ModelEvaluationCharts`) when selecting kernel.

# Strategic Plan
- Register an explicit kernel name for this project.
- Bind notebook metadata to that kernel.
- Configure VS Code workspace interpreter settings.
- Align setup script with the same kernel identity.

# Tactical Plan
- [x] Register `modelevalcharts-local` kernel with distinctive display name.
- [x] Update notebook kernelspec to `modelevalcharts-local`.
- [x] Add `.vscode/settings.json` with `.venv` interpreter path.
- [x] Update `scripts/setup_notebook_kernel.ps1` to install same kernel name.
- [x] Verify kernel list includes `modelevalcharts-local`.

# Architecture Notes
- Kernel display name: `ModelEvaluationCharts (LOCAL .venv)`
- Kernel name: `modelevalcharts-local`
- Workspace interpreter: `${workspaceFolder}\\.venv\\Scripts\\python.exe`

# Test Plan

## Automated tests (what/where)
- Not in this slice.

## Manual verification script
- Open `ModelEvaluationCharts` folder as the active workspace.
- Open `notebooks/simulated_data_testing.ipynb`.
- In kernel picker, select `ModelEvaluationCharts (LOCAL .venv)`.
- Run all cells.

# Progress Log
- 2026-02-25: Registered explicit local project kernel.
- 2026-02-25: Updated notebook metadata to use local project kernel.
- 2026-02-25: Added VS Code workspace interpreter settings.
- 2026-02-25: Updated setup script to register matching kernel.

# Final Summary
Kernel visibility/binding is now explicitly configured for this workspace and notebook.
