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

---

# Slice 3 - Visual Redesign + KS Cutoff Marker Only

## Task Contract

### Goal
Redesign the report and charts to professional reporting standards, and replace KS curve visualization with only a vertical line at optimal KS cutoff.

### Acceptance Criteria
- KS is not rendered as a line trace on the gain chart.
- A vertical line marks the optimal KS cutoff percentile.
- Report visual design is materially improved for readability and professionalism:
  - cleaner typography hierarchy
  - reduced clutter in annotations/legend/hover
  - consistent color system and spacing
  - polished report container and explanatory text
- HTML remains self-contained and generated by `EvaluateModel`.

### Non-Goals
- Changing evaluation logic for success window.
- Removing the percentile slider interaction.

### Assumptions / Open Questions
- Assumption: keep additional gain context lines (model, random, ideal, non-success share) while removing only KS curve trace.

## Strategic Plan
- Adjust tests to assert absence of KS trace and presence of optimal-cutoff vertical marker.
- Redesign figure layout/annotation/hover for cleaner reporting UX.
- Redesign surrounding HTML shell styling.
- Re-verify tests and generated report.

## Tactical Plan
- [x] Update tests for "no KS trace" and persistent optimal KS vertical line.
- [x] Refactor figure styling and annotations for reduced clutter and professional appearance.
- [x] Remove KS trace and secondary KS axis while preserving KS cutoff marker.
- [x] Improve HTML report structure and CSS.
- [x] Run pytest and regenerate report.

## Architecture Notes
- Public API unchanged.
- Metrics still compute KS numerically to identify `best_ks_percentile`.
- Chart contract:
  - no KS plotted curve
  - optimal KS shown as vertical reference line + concise annotation.

## Test Plan

### Automated tests (what/where)
- `tests/test_evaluate_model.py`:
  - assert no trace named `KS`
  - assert `ks_optimal_split` shape exists
  - retain checks for new gain-context lines and metrics.

### Manual verification script
- Run: `python -m pytest -q`
- Run: `python evaluate_model.py`
- Open `outputs/model_evaluation_report.html` and confirm visual polish and reduced clutter.

## Progress Log
- 2026-02-25: Opened Slice 3 for report visual redesign and KS cutoff marker-only display.
- 2026-02-25: Updated tests to assert KS is not plotted as a trace and that optimal split marker remains.
- 2026-02-25: Reworked figure styling (axes, spacing, colors, legend, hover behavior, slider aesthetics) for report-grade readability.
- 2026-02-25: Removed KS line and KS secondary axis; retained only vertical optimal cutoff marker and concise annotation.
- 2026-02-25: Redesigned HTML shell with clearer header hierarchy and cleaner visual framing.
- 2026-02-25: Verified with `python -m pytest -q` (2 passed) and regenerated `outputs/model_evaluation_report.html`.

## Final Summary
Slice 3 completed. The report visual design has been polished for professional presentation, and KS is now represented only by an optimal-cutoff vertical marker (no KS line trace).

---

# Slice 4 - Presentation Layout and Visual Hierarchy

## Task Contract

### Goal
Implement a presentation-ready redesign with external KPI cards, simplified gain chart traces, clearer cutoff storytelling, and cleaner subplot/legend structure.

### Acceptance Criteria
- KPI summary is rendered in a separate HTML header card row, not inside plot area.
- Top chart visually emphasizes Model line and de-emphasizes baselines.
- Legend is outside the plotting area and uses concise labels.
- Selected cutoff is the primary vertical marker and has one concise callout.
- Optimal cutoff remains a secondary, lighter reference marker.
- Shared x-axis remains, with reduced clutter and tighter subplot spacing.
- Footer definitions are present below charts.

### Non-Goals
- Changing model/event matching logic.
- Adding new data sources.

### Assumptions / Open Questions
- Assumption: presentation version can remove slider and use fixed selected cutoff (default 20%).

## Strategic Plan
- Add tests for simplified trace set, no slider, and cutoff markers.
- Refactor figure to static presentation view (no slider annotations).
- Add KPI cards and footnote text to HTML shell.
- Re-verify with tests and report generation.

## Tactical Plan
- [x] Update tests to reflect simplified trace names and no slider.
- [x] Implement static selected cutoff + optimal cutoff markers with concise callout.
- [x] Simplify gain chart lines to Model/Random/Ideal (remove non-success trace from display).
- [x] Add HTML KPI cards above figure and definitions footnote below.
- [x] Run pytest and regenerate report.

## Architecture Notes
- Public API unchanged.
- Internal helper addition:
  - derive selected-cutoff KPI values from `metrics` for header cards.
- Figure contract:
  - no slider
  - static lines/shapes/annotations for selected and optimal cutoffs.

## Test Plan

### Automated tests (what/where)
- `tests/test_evaluate_model.py`:
  - assert simplified trace names (`Model`, `Random`, `Ideal`, `Success Rate`)
  - assert no sliders in layout
  - assert both `selected_cutoff` and `ks_optimal_split` shapes exist.

### Manual verification script
- Run: `python -m pytest -q`
- Run: `python evaluate_model.py`
- Open `outputs/model_evaluation_report.html` and review KPI cards + clean chart layout.

## Progress Log
- 2026-02-25: Opened Slice 4 for presentation-style chart and report redesign.
- 2026-02-25: Updated tests to require presentation trace names, no slider, and both cutoff marker shapes.
- 2026-02-25: Refactored figure to static presentation mode with shared x-axis clarity, cleaner legend placement, and concise selected-cutoff callout.
- 2026-02-25: Simplified top chart display to Model/Random/Ideal and removed displayed non-success line.
- 2026-02-25: Added KPI card row and definitions footnote to HTML report shell.
- 2026-02-25: Verified with `python -m pytest -q` (2 passed) and `python evaluate_model.py` (report regenerated).

## Final Summary
Slice 4 completed. The report now uses a presentation-ready layout with KPI cards above the charts, simplified gain-chart styling, one hero selected-cutoff marker, a secondary optimal-cutoff marker, and a cleaner overall visual hierarchy.
