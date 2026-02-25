# Task Contract

## Goal
Restore `EvaluateModel` compatibility after pulling latest `main` by supporting legacy notebook arguments `historical_period_start` and `historical_period_end` without breaking current `historical_period` usage.

## Acceptance Criteria
- Calling `EvaluateModel(..., historical_period_start="YYYY-MM-DD", historical_period_end="YYYY-MM-DD")` no longer raises `TypeError`.
- Legacy kwargs resolve deterministically to the same monthly historical period selection behavior used by `historical_period`.
- Ambiguous/conflicting combinations (`historical_period` plus legacy args, or start/end month mismatch) raise a clear `ValueError`.
- Automated tests cover successful legacy call and error paths.

## Non-Goals
- Reworking campaign-selection chart logic.
- Changing report layout/content beyond compatibility and validation messaging.

## Assumptions / Open Questions
- Assumption: legacy start/end are intended to reference a single month snapshot; when both are provided they must resolve to the same month.

# Strategic Plan
- Add a small compatibility shim in `EvaluateModel` API.
- Add tests first for red/green flow (legacy success + conflict validation).
- Run tests and confirm no regressions.

# Tactical Plan
- [x] Add tests for legacy kwargs acceptance and conflict handling.
- [x] Implement `EvaluateModel` signature/update with compatibility mapping.
- [x] Validate with targeted pytest run.
- [x] Update this task file checklist/progress/final summary.

# Architecture Notes
- Public API remains `EvaluateModel(...) -> Path`.
- Add optional kwargs:
  - `historical_period_start: str | pd.Timestamp | None = None`
  - `historical_period_end: str | pd.Timestamp | None = None`
- Resolution rules:
  - If `historical_period` is set, legacy args must be unset.
  - If legacy args are set, map to one `historical_period` month.
  - If both legacy args are set, require same calendar month.
- Keep validation near the beginning of `EvaluateModel` so callers fail fast with actionable errors.

# Test Plan
## Automated tests
- Extend `tests/test_evaluate_model.py`:
  - Legacy kwargs accepted when start/end same month.
  - Clear errors for conflicting period arguments.
  - Clear errors for start/end month mismatch.

## Manual verification script
- Run `python -m pytest tests/test_evaluate_model.py -k historical_period`.
- In notebook, rerun cell that calls `EvaluateModel(... historical_period_start=..., historical_period_end=...)` and confirm report path prints.

# Progress Log
- 2026-02-25: Intake complete. Identified API mismatch: notebook uses `historical_period_start/end` while `EvaluateModel` currently accepts only `historical_period`.
- 2026-02-25: Planner created this task file with acceptance criteria and tactical slices.
- 2026-02-25: QA tests-first added three legacy-arg tests; red phase confirmed with `TypeError` for unknown `historical_period_start`.
- 2026-02-25: Developer added `_resolve_historical_period_input(...)` and wired `EvaluateModel` to accept/validate legacy args.
- 2026-02-25: Build/Verify passed. Targeted tests passed (`3 passed`), full suite passed (`12 passed`), and sample report generated at `outputs/model_evaluation_report_legacy_args.html`.
- 2026-02-25: Reviewer pass approved; no blockers found for this slice.

# Final Summary
Implemented backward-compatible `EvaluateModel` period arguments:
- Added support for `historical_period_start` and `historical_period_end`.
- Added explicit conflict validation when combined with `historical_period`.
- Added explicit month-mismatch validation when start/end fall in different calendar months.
- Added tests for successful legacy usage and both validation errors.
- Verified with passing test suite and generated sample HTML report via legacy arguments.
