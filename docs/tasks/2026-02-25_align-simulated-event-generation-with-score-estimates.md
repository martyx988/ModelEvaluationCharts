# Task Contract

## Goal
Find and fix why campaign-selection bottom charts diverge strongly from top charts when `campaign_clients` contains all clients from `model_score`.

## Acceptance Criteria
- Root cause is identified with code-level evidence.
- With simulated data (`seed=42`) and `campaign_clients` set to all latest scored clients, campaign estimated curves are reasonably close to top actual curves.
- A regression test captures the expected alignment behavior and passes.

## Non-Goals
- Redesigning the chart UI.
- Changing report section structure.

## Assumptions / Open Questions
- Assumption: in simulation, score should be the direct proxy for expected event propensity used by campaign estimated charts.

# Strategic Plan
- Reproduce and quantify top-vs-bottom divergence.
- Add a failing regression test for all-client alignment.
- Fix simulation/event generation mismatch.
- Verify tests and regenerate sample report.

# Tactical Plan
- [x] Add regression test for all-client top-vs-campaign curve alignment.
- [x] Update simulator event sampling to match score-based expectation assumptions.
- [x] Run pytest and confirm pass.
- [x] Update this task file with progress and final summary.

# Architecture Notes
- Keep `EvaluateModel` API unchanged.
- Fix should be localized to `simulated_data.py` so score semantics are consistent across:
  - actual outcome generation (`target_store`)
  - estimated campaign metrics (`_build_estimated_metrics_by_contact_percentile`).

# Test Plan
## Automated tests
- Add test in `tests/test_evaluate_model.py`:
  - Build top metrics from latest snapshot + `target_store`.
  - Build campaign metrics from all latest clients.
  - Assert max absolute difference for gain and cumulative success rate stay within defined tolerance.

## Manual verification script
- Run `python -m pytest`.
- Generate report with all-client campaign selection and compare top vs bottom curves visually.

# Progress Log
- 2026-02-25: Intake complete. Reproduced issue and measured large divergence (`max gain diff ~15.8pp`, `max SR diff ~19.0pp`) in all-client case.
- 2026-02-25: Planner created task file and tactical slices.
- 2026-02-25: QA tests-first added `test_all_client_campaign_curves_align_with_top_curves_for_simulated_data`; red phase failed as expected (`max gain diff 15.826 > 8.0`).
- 2026-02-25: Developer updated `simulated_data.py` event sampling to use linear score weights instead of exponentiated score weights.
- 2026-02-25: Build/Verify passed. Targeted regression test passed; full suite passed (`13 passed`).
- 2026-02-25: Post-fix check for all-client case: `max gain diff 2.065pp`, `max SR diff 2.227pp`.
- 2026-02-25: Reviewer approved slice; no remaining blockers.

# Final Summary
Root cause:
- Simulation generated actual events with probability proportional to `score ** k` (`k` from 1.8 to 2.4), but campaign bottom charts estimate expected successes linearly from `score`.
- This mismatch produced large top-vs-bottom divergence even when campaign included all clients.

Fix:
- Changed simulated event generation to sample using linear `score` weights.
- Added regression test asserting all-client campaign curves stay close to top curves.

Evidence:
- Before fix: max gain diff ~15.8pp, max cumulative SR diff ~19.0pp.
- After fix: max gain diff ~2.1pp, max cumulative SR diff ~2.2pp.
- Automated verification: `python -m pytest` passed (`13 passed`).
