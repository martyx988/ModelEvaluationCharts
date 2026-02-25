# Task Contract
- Goal:
  - Make `notebooks/testing_msp.ipynb` run successfully in a notebook session, including imports and report generation.
- Acceptance Criteria (clear, measurable):
  - Running the notebook import cell from `notebooks/` working directory succeeds without `ModuleNotFoundError`.
  - The notebook uses a valid callable from `evaluate_model.py` and executes the report-generation cell successfully.
  - Notebook JSON remains valid after edits.
- Non-Goals:
  - Refactor the reporting architecture in `evaluate_model.py`.
  - Introduce packaging/build-system changes.
- Assumptions / Open Questions (minimize):
  - Assumption: `testing_msp` should use the existing `EvaluateModel` API with campaign-selection enabled rather than a separate `EvaluateModel_msp` function.

# Strategic Plan
1. Reproduce and identify why `testing_msp` import fails.
2. Add a regression test for notebook import behavior (tests-first).
3. Patch notebook imports/path handling and function usage.
4. Verify execution of import and generation cells from notebook working directory.
5. Record evidence and close the slice.

# Tactical Plan
- [x] Inspect notebook cells and module exports to identify root causes.
- [x] Add regression test `tests/test_testing_msp_notebook.py` that executes import cell from `notebooks/` cwd.
- [x] Update notebook import cell to prepend project root to `sys.path`.
- [x] Replace nonexistent `EvaluateModel_msp` call with existing `EvaluateModel`.
- [x] Verify import cell and report-generation cell run successfully via subprocess from `notebooks/` cwd.
- [x] Verify import cell and report-generation cell also run from repository-root cwd.
- [x] Confirm notebook JSON is still parseable/valid.

# Architecture Notes
- Next-slice decisions:
  - Keep implementation notebook-local to avoid wider API churn.
  - Public API used by notebook: `EvaluateModel(...) -> Path`.
  - Input/Output contract remains unchanged:
    - Inputs: keyword args to `EvaluateModel` (`output_html_path`, `seed`, `include_campaign_selection`, `campaign_clients`).
    - Output: filesystem path to generated HTML report.
- Error handling strategy:
  - Resolve module imports deterministically by adding parent directory of `notebooks/` to `sys.path` in the notebook import cell.
  - Avoid silent fallback; keep direct imports so failures remain actionable.

# Test Plan
- Automated tests (what/where):
  - Added `tests/test_testing_msp_notebook.py`:
    - Reads `notebooks/testing_msp.ipynb`.
    - Executes first code cell in subprocess with cwd=`notebooks/`.
    - Asserts return code is zero.
- Manual verification script (how to open/view report):
  1. Open `notebooks/testing_msp.ipynb`.
  2. Run all cells top-to-bottom.
  3. Confirm `output_path` is returned and points to HTML.
  4. Confirm IFrame preview renders the generated report.

# Progress Log
- 2026-02-25 (Intake, Gate PASS): Restated goal as fixing notebook runtime; acceptance criteria defined around import success + cell execution.
- 2026-02-25 (Planner, Gate PASS): Updated this task file with strategic/tactical plans and ordered dependencies.
- 2026-02-25 (Architect, Gate PASS): Chose notebook-local path bootstrap and existing `EvaluateModel` API to minimize rework.
- 2026-02-25 (QA tests-first, Gate PASS): Added `tests/test_testing_msp_notebook.py` before implementation. `pytest` unavailable in `.venv`, so red-phase evidence captured by direct subprocess failure (`ImportError` before fix).
- 2026-02-25 (Developer, Gate PASS): Patched `notebooks/testing_msp.ipynb` import cell (`sys.path` bootstrap) and replaced `EvaluateModel_msp` with `EvaluateModel`.
- 2026-02-25 (Build/Verify, Gate PASS): Executed import cell and import+run cells in subprocess with cwd=`notebooks/` and repo root; all returned code 0.
- 2026-02-25 (QA re-run, Gate PASS): Re-ran automated tests (`python -m pytest -q`): 7 passed.
- 2026-02-25 (Reviewer, Gate PASS): Reviewed for correctness and minimal blast radius; approved.
- 2026-02-25 (Planner close-out, Gate PASS): Checklist reflects verified outcomes and slice is complete.

# Final Summary
- Fixed `notebooks/testing_msp.ipynb` so it works from notebook cwd by:
  - adding project-root path bootstrap in the import cell,
  - switching from nonexistent `EvaluateModel_msp` to existing `EvaluateModel`.
- Added regression test scaffold in `tests/test_testing_msp_notebook.py` for notebook import behavior.
- Verified via subprocess execution that both import and report-generation cells now run successfully.
