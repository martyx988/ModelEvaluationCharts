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
