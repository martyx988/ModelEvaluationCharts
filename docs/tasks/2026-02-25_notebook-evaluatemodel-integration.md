# Task Contract

## Goal
Enable manual testing of `EvaluateModel` directly in the existing testing notebook.

## Acceptance Criteria
- Testing notebook imports `EvaluateModel`.
- Notebook includes a cell that runs `EvaluateModel` and prints generated HTML path.
- Notebook includes a cell to preview the generated report.

## Non-Goals
- Refactoring core evaluation logic.
- Adding automated notebook execution tests.

## Assumptions / Open Questions
- Assumption: inline iframe preview is acceptable for quick validation.

# Strategic Plan
- Extend notebook imports.
- Add execution and preview cells.
- Commit notebook integration changes.

# Tactical Plan
- [x] Import `EvaluateModel` in notebook.
- [x] Add cell to generate report.
- [x] Add inline report preview cell.
- [x] Commit changes.

# Architecture Notes
- Reused existing `repo_root` path logic in notebook.
- `EvaluateModel` output path set to `outputs/model_evaluation_report.html`.

# Test Plan

## Automated tests (what/where)
- Not in this slice.

## Manual verification script
- Open `notebooks/simulated_data_testing.ipynb`.
- Run all cells.
- Confirm printed report path and visible iframe rendering.

# Progress Log
- 2026-02-25: Added `EvaluateModel` import and execution cell in notebook.
- 2026-02-25: Added inline iframe preview cell for generated report.

# Final Summary
Notebook is now ready to test simulated data generation and `EvaluateModel` report generation in one place.
