# Task Contract

## Goal
Create a new main function `EvaluateModel` that uses simulated data and generates an HTML report with two professional interactive Plotly charts: cumulative gain and cumulative success rate.

## Acceptance Criteria
- A Python function `EvaluateModel` exists in a dedicated file.
- Calling `EvaluateModel` generates and returns an HTML file path.
- Report contains two interactive charts:
  - Cumulative Gain chart
  - Cumulative Success Rate chart
- Success is defined by joining `model_score` with `target_store` and flagging target events that happen from `fs_time` up to one calendar month later.
- A single interactive percentile control affects both charts simultaneously.
- Styling is professional and readable for reporting use.

## Non-Goals
- Production deployment and scheduling.
- Using real confidential data.
- Additional model diagnostics beyond requested charts.

## Assumptions / Open Questions
- Assumption: "up to this percentile" means contacting top-scored clients up to selected share of population.
- Assumption: one month window is inclusive of both boundaries.
- Open question: none blocking for this slice.

# Strategic Plan
- Implement evaluation and metric preparation pipeline.
- Build a two-subplot Plotly figure with one shared percentile slider.
- Export a self-contained HTML report and verify generation.

# Tactical Plan
- [x] Create new module with `EvaluateModel`.
- [x] Implement schema validation and one-month success window join logic.
- [x] Compute cumulative gain and cumulative success rate by contacted percentile.
- [x] Build professional Plotly visuals and shared slider interaction.
- [x] Export report to HTML and return output path.
- [x] Run smoke generation and verify file output.

# Architecture Notes
- Public API:
  - `EvaluateModel(output_html_path: str | Path = "outputs/model_evaluation_report.html", seed: int | None = 42) -> Path`
- Data source:
  - Uses `create_simulated_tables` from `simulated_data.py`.
- Chart strategy:
  - One Plotly figure with two subplots and shared x-axis.
  - Single slider updates a shared cutoff indicator and annotations across both charts.
- Error handling:
  - Explicit required-column validation.
  - Explicit datetime/numeric coercion checks with actionable exceptions.

# Test Plan

## Automated tests (what/where)
- Planned follow-up: add pytest unit tests for:
  - success window logic
  - metric monotonicity and ranges
  - HTML generation path and basic content checks

## Manual verification script
- Run: `python evaluate_model.py`
- Open generated file `outputs/model_evaluation_report.html` in browser.
- Verify:
  - both charts render
  - slider updates cutoff for both charts
  - hover interactions work and legends are visible

# Progress Log
- 2026-02-25: Created `evaluate_model.py` with `EvaluateModel` and helper pipeline.
- 2026-02-25: Added two-chart Plotly report with one shared percentile slider.
- 2026-02-25: Added this task definition file for the slice.
- 2026-02-25: Installed `plotly` dependency in environment.
- 2026-02-25: Ran smoke generation and produced `outputs/model_evaluation_report.html`.

# Final Summary
Slice completed. `EvaluateModel` now builds a self-contained HTML report with cumulative gain and cumulative success rate charts, both controlled by one shared percentile slider.

---

# Slice 2 - Gain Chart Enrichment + KS Split

## Task Contract

### Goal
Enhance the cumulative gain visualization to include additional benchmark/diagnostic lines and a KS line with an explicit optimal split marker.

### Acceptance Criteria
- Gain-related metrics include cumulative non-success distribution needed for KS.
- Figure includes a KS curve on the gain panel context and marks the max-KS split.
- Figure exposes additional gain context lines beyond model and random baseline.
- Existing chart generation still produces a self-contained HTML report.

### Non-Goals
- Replacing the shared slider interaction model.
- Reworking simulated data generation.

### Assumptions / Open Questions
- Assumption: KS uses `cum_success_share - cum_non_success_share` with percentile on x-axis.
- Assumption: "more lines" means adding at least ideal/reference and non-success cumulative context.

## Strategic Plan
- Extend metrics computation with non-success cumulative shares and KS values.
- Extend figure traces and annotation to surface KS and optimal split.
- Verify with focused unit tests and smoke run.

## Tactical Plan
- [x] Add tests that assert new metrics columns and KS max split behavior.
- [x] Add tests that assert figure contains KS and extra gain context traces.
- [x] Implement metric extensions for cumulative non-success share and KS.
- [x] Implement KS trace and optimal split marker/annotation on figure.
- [x] Run pytest and smoke-generate report HTML.

## Architecture Notes
- Public API unchanged:
  - `EvaluateModel(output_html_path: str | Path = "outputs/model_evaluation_report.html", seed: int | None = 42) -> Path`
- Metrics contract extension from `_build_metrics_by_contact_percentile`:
  - Add `cum_non_successes`, `non_success_share_pct`, `ks_pct`, and `best_ks_percentile`.
- Chart strategy:
  - Keep two-subplot structure.
  - Add KS trace to top subplot using secondary y-axis.
  - Add ideal gain line and non-success cumulative line for richer diagnostic context.
- Error handling:
  - Guard divide-by-zero when no successes or no non-successes.

## Test Plan

### Automated tests (what/where)
- `tests/test_evaluate_model.py`:
  - metric column presence and value range checks for KS and non-success shares.
  - figure trace presence checks for KS and new gain context lines.
  - best KS split exists and aligns with max KS row.

### Manual verification script
- Run: `python -m pytest`
- Run: `python evaluate_model.py`
- Open generated file `outputs/model_evaluation_report.html`.
- Verify:
  - gain panel includes additional lines and KS line.
  - a visible marker/annotation indicates KS optimal split.
  - hover and slider interactions still work.

## Progress Log
- 2026-02-25: Opened Slice 2 for gain-chart enrichment and KS optimal split.
- 2026-02-25: Added red-phase tests in `tests/test_evaluate_model.py` for KS metrics, added lines, and optimal split marker.
- 2026-02-25: Extended metrics with cumulative non-success share, KS curve values, and best KS percentile.
- 2026-02-25: Extended figure with Ideal Gain, Cumulative Non-Success Share, KS trace (secondary axis), and KS optimal split annotation/line.
- 2026-02-25: Verified with `python -m pytest -q` (2 passed) and `python evaluate_model.py` (report generated).

## Final Summary
Slice 2 completed. The gain panel now includes richer reference lines plus a KS line and explicit optimal KS split, backed by automated tests and a regenerated HTML report.
