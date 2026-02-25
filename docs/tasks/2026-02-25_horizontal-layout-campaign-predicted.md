# Task Contract
- Goal:
  - Redesign the report layout and add a new campaign-clients predicted-performance section with interactive charts.
- Acceptance Criteria (clear, measurable):
  - The first section renders two original charts side-by-side horizontally.
  - Narrative descriptions formerly on the right are removed and replaced by tooltip buttons attached to each chart.
  - A new campaign section appears below with three charts:
    - A wide, short percentile distribution bar chart with gray bars (all scored clients) and blue bars (`campaign_clients` subset).
    - Two charts below it side-by-side (gain + cumulative success rate) that mirror the original chart logic but are computed on `campaign_clients`-filtered base.
  - In the new campaign section, success estimation uses predicted `score` as expected successes (sum of score as total expected successes), not `target_store` outcomes.
  - Existing slider controls in the top section update both top and campaign filtered charts.
- Non-Goals:
  - Rebuild repository architecture or split into multiple modules.
  - Change notebook kernel setup or unrelated notebooks.
- Assumptions / Open Questions (minimize):
  - Assumption: Predicted scores are probability-like and numeric; expected successes are computed as cumulative sum of score.
  - Assumption: Distribution chart uses newest model snapshot (same snapshot as first section).

# Strategic Plan
1. Add/adjust tests for required layout, tooltips, and campaign predicted section (red phase).
2. Implement chart/data helpers for split gain/SR figures and campaign predicted metrics.
3. Recompose HTML/CSS/JS for new layout and shared sliders.
4. Run test suite and validate generated report artifacts.
5. Review against acceptance criteria and close out.

# Tactical Plan
- [x] Add tests-first assertions for horizontal chart cards and tooltip buttons.
- [x] Add tests-first assertions for campaign predicted section and new chart container ids.
- [x] Add helper to compute percentile distribution for all vs campaign clients.
- [x] Add helper to compute expected-success metrics using `score` instead of actual outcomes.
- [x] Split original combined figure into reusable gain and success-rate figures.
- [x] Update report HTML to two top chart cards with tooltip buttons.
- [x] Add new campaign section with one wide distribution chart and two side-by-side filtered charts.
- [x] Update JS slider behavior to synchronize both top and campaign filtered charts.
- [x] Run `python -m pytest -q`.

# Architecture Notes
- Public API remains:
  - `EvaluateModel(...) -> Path`
- Data contracts for new slice:
  - Input: `campaign_clients` with required `pt_unified_key` when `include_campaign_selection=True`.
  - New computed datasets:
    - `distribution_df`: counts by `percentile` for all scored clients and selected campaign clients.
    - `campaign_estimated_metrics`: 1..100 percentile curve with expected successes derived from `score`.
- Chart assembly:
  - Build separate figure functions for gain and success-rate so they can be rendered side-by-side in both sections.
  - Add one distribution figure builder for campaign section.
- Error handling:
  - Validate required columns and empty joins.
  - Raise actionable `ValueError` when campaign keys do not match scored clients.

# Test Plan
- Automated tests (what/where):
  - Extend `tests/test_evaluate_model.py` to verify:
    - top chart card ids/tooltips exist in HTML,
    - campaign distribution + filtered chart ids exist,
    - campaign section text indicates expected-success estimation from score.
  - Keep existing tests for interactive controls, update expected strings where layout wording changes.
- Manual verification script (how to open/view report):
  1. Run `python -c "from evaluate_model import EvaluateModel; print(EvaluateModel(include_campaign_selection=True))"`.
  2. Open generated HTML in browser.
  3. Confirm top charts are horizontal and tooltips open on hover.
  4. Move sliders and confirm both top and campaign filtered charts update markers/required cutoff.
  5. Confirm campaign distribution chart shows gray + blue bars per percentile.

# Progress Log
- 2026-02-25 (Intake, Gate PASS): Captured requested redesign: horizontal top charts, tooltip migration, and new campaign predicted-performance charts with shared slider behavior.
- 2026-02-25 (Planner, Gate PASS): Created task file with strategic/tactical plan and measurable acceptance criteria.
- 2026-02-25 (Architect, Gate PASS): Kept `EvaluateModel` API stable and defined new internal helpers for split charts and campaign predicted metrics/distribution.
- 2026-02-25 (QA tests-first, Gate PASS): Updated tests in `tests/test_evaluate_model.py`; confirmed red phase with failing expectations for missing card ids and campaign predicted text.
- 2026-02-25 (Developer, Gate PASS): Implemented split top charts, tooltip controls, campaign distribution chart, and campaign filtered gain/SR charts using score-based expected successes.
- 2026-02-25 (Build/Verify, Gate PASS): Ran `python -m pytest tests/test_evaluate_model.py -q` and `python -m pytest -q`; both passed.
- 2026-02-25 (QA re-run, Gate PASS): Rechecked automated coverage on updated layout/section assertions; all tests green.
- 2026-02-25 (Reviewer, Gate PASS): Verified acceptance criteria alignment, slider synchronization behavior, and campaign score-based estimation logic.
- 2026-02-25 (Planner close-out, Gate PASS): Checklist updated with verified completion evidence.

# Final Summary
- First section now renders as two horizontal chart cards (gain and cumulative success rate), each with a tooltip button replacing the former right-side narrative panel.
- Added campaign predicted-performance section:
  - wide/short percentile distribution chart (`all clients` gray vs `campaign clients` blue),
  - two side-by-side campaign-filtered charts mirroring top-section gain and success-rate charts.
- Campaign expected-success calculations now use predicted `score` sums (not target outcomes) for campaign filtered metrics.
- Slider controls in the top section update both top and campaign filtered charts.
- Validation evidence: `python -m pytest -q` passed (6 passed).
