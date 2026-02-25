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
- [x] Add follow-up regression test for all-client KS-optimal cutoff parity.
- [x] Implement all-client campaign metric parity for KS cutoff.
- [x] Re-run pytest and confirm pass.
- [x] Update this task file with follow-up summary.
- [x] Add tests-first check for default-rate comparison bar label placement.
- [x] Update chart trace text placement to inside-bar labels.
- [x] Re-run pytest and confirm pass for this UI tweak.
- [x] Update task log/final summary for this UI tweak.
- [x] Add regression test that validates full-base campaign chart values stay close to top chart values.
- [x] Re-run pytest and confirm pass for regression guard.
- [x] Add tests-first coverage for report language parameter (`en`/`cs`) and validation.
- [x] Implement `EvaluateModel(language=...)` with Czech HTML text variants.
- [x] Re-run pytest and confirm pass for localization slice.
- [x] Remove MSP-dependent test artifacts after MSP module/notebook deletion.
- [x] Re-run pytest and confirm pass after MSP cleanup.
- [x] Add chart-level localization (titles, axes, legends, hover, annotations) for `language="cs"` with diacritics.
- [x] Update Czech HTML copy to use diacritics consistently.
- [x] Re-run pytest and confirm pass for diacritics + chart-localization slice.
- [x] Localize tooltip body texts and definitions footnote in Czech.

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
- 2026-02-25: Follow-up reported by user: all-client scenario still shows different KS-optimal cutoff in bottom gain chart. New QA/Dev slice started.
- 2026-02-25: QA added `test_all_client_campaign_ks_cutoff_matches_top_metrics`; red phase reproduced mismatch (`47` vs `58`).
- 2026-02-25: Developer added `_campaign_covers_whole_base(...)` and `_resolve_campaign_metrics_for_report(...)`; report now reuses top actual metrics for campaign gain/success charts when campaign equals full latest base.
- 2026-02-25: Build/Verify follow-up passed. Targeted tests passed (`2 passed`), full suite passed (`14 passed`).
- 2026-02-25: New follow-up requested: place “Default Success Rate Comparison” values inside bars (instead of above). Starting tests-first + implementation slice.
- 2026-02-25: QA tests-first added `test_campaign_rate_comparison_labels_render_inside_bars`; red phase failed (`textposition='outside'`).
- 2026-02-25: Developer updated `_make_campaign_rate_comparison_figure(...)` to use inside labels (`textposition='inside'`, centered).
- 2026-02-25: Build/Verify UI tweak passed. Targeted test passed and full suite passed (`15 passed`).
- 2026-02-25: Added report-level regression `test_full_base_campaign_report_chart_values_remain_close` to compare top and campaign chart payloads (`gain`, `sr`) when campaign uses full latest base.
- 2026-02-25: Regression guard verification passed. Full suite now `16 passed`.
- 2026-02-25: New follow-up requested: add report language parameter with `en`/`cs` and Czech HTML variant. Starting tests-first + implementation slice.
- 2026-02-25: User confirmed MSP module and MSP notebook were intentionally deleted; requested cleanup of remaining MSP-dependent artifacts.
- 2026-02-25: Removed MSP-dependent artifacts (`tests/test_testing_msp_notebook.py` and stale `evaluate_model_msp` import in tests).
- 2026-02-25: Implemented `EvaluateModel(language="en"|"cs")` with language validation and Czech HTML labels.
- 2026-02-25: Build/Verify passed after cleanup/localization: `python -m pytest` -> `17 passed`.
- 2026-02-25: User requested Czech with diacritics and translation of chart texts (not only surrounding HTML). Starting follow-up localization slice.
- 2026-02-25: Added `_resolve_chart_text(language)` and wired localized chart text through figure builders (gain/success/distribution/rate comparison), including hover, annotations, axis titles, and trace names.
- 2026-02-25: Updated Czech report/UI copy to proper diacritics.
- 2026-02-25: Added regression test for Czech chart labels on figure builders; full test suite passed (`18 passed`).
- 2026-02-25: Localized remaining tooltip paragraphs/list items and “Definitions” footnote via language labels; regression tests still pass (`18 passed`).

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

Follow-up:
- Added explicit all-client parity handling so campaign bottom gain/success charts share the same KS optimum as the top charts when `campaign_clients` covers the full latest scored base.
- Follow-up verification: `python -m pytest` passed (`14 passed`).

UI tweak:
- Updated “Default Success Rate Comparison” bar labels to render inside each bar.
- Verification after tweak: `python -m pytest` passed (`15 passed`).

Regression guard:
- Added a full-base campaign regression test that parses report chart payloads and enforces closeness between top and campaign chart values.
- Threshold selected: `<= 1.0` percentage point for both gain and cumulative success-rate series.
- Verification after test addition: `python -m pytest` passed (`16 passed`).

Localization (in progress):
- Add `EvaluateModel(language="en"|"cs")` and Czech HTML copy for report sections and UI labels.
  - Completed.

MSP cleanup (in progress):
- Remove tests/imports that still depend on deleted `evaluate_model_msp.py` and `notebooks/testing_msp.ipynb`.
  - Completed.
