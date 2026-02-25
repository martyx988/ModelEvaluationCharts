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

---

# Slice 5 - Interactive Cutoff Control (Presentation-Safe)

## Task Contract

### Goal
Restore user-selectable cutoff interaction while keeping the presentation-style layout clean.

### Acceptance Criteria
- User can choose cutoff percentile interactively in the HTML report.
- Cutoff interaction updates KPI cards and selected-cutoff chart marker/callout.
- Optimal KS marker remains visible as a secondary static reference.
- No reintroduction of in-chart slider clutter.

### Non-Goals
- Reintroducing the old Plotly slider control.

### Assumptions / Open Questions
- Assumption: best UX is an external header control (range input) driving Plotly relayout + KPI updates.

## Strategic Plan
- Add tests for presence of interactive cutoff UI in generated HTML.
- Add lightweight JS controller bound to range input.
- Keep figure static except relayout updates for selected-cutoff elements.

## Tactical Plan
- [x] Add/extend tests for interactive control markup in generated report.
- [x] Add cutoff range control and value label in HTML.
- [x] Add JS to update KPI cards and selected-cutoff shape/annotation.
- [x] Run pytest and regenerate report.

## Architecture Notes
- Public API unchanged.
- `EvaluateModel` embeds a cutoff data map (1..100) in HTML JS.
- JS applies `Plotly.relayout` on `model-figure` to move selected-cutoff line/callout.

## Test Plan

### Automated tests (what/where)
- `tests/test_evaluate_model.py`:
  - assert generated HTML contains cutoff slider id and relayout script marker.

### Manual verification script
- Run: `python evaluate_model.py`
- Open report and move cutoff slider.
- Verify KPI cards and selected cutoff marker update accordingly.

## Progress Log
- 2026-02-25: Opened Slice 5 to restore user-selectable cutoff via external control panel.
- 2026-02-25: Added report-level test asserting interactive cutoff markup and Plotly relayout hook.
- 2026-02-25: Added header slider control for cutoff percentile with live value badge.
- 2026-02-25: Embedded cutoff data map and JS controller to update KPI cards and selected-cutoff line/callout via `Plotly.relayout`.
- 2026-02-25: Verified with `python -m pytest -q` (3 passed) and regenerated report via `python evaluate_model.py`.

## Final Summary
Slice 5 completed. Interactive cutoff selection is restored through a clean external control that updates KPI cards and chart cutoff annotation/line without reintroducing in-plot slider clutter.

---

# Slice 6 - Narrower Charts + Business Reading Guide

## Task Contract

### Goal
Make charts visually thinner and add a business-facing interpretation panel to the right of the charts.

### Acceptance Criteria
- Chart area width is reduced from current presentation layout.
- A right-side panel explains how to interpret both charts in business terms.
- Layout remains responsive (right panel stacks below on smaller screens).

### Non-Goals
- Changing evaluation logic or interactive cutoff behavior.

### Assumptions / Open Questions
- Assumption: desktop layout uses a two-column content area (chart left, guide right).

## Strategic Plan
- Add test for business guidance panel content in generated HTML.
- Refactor report body layout into two columns.
- Reduce figure width to improve balance with right-side narrative panel.

## Tactical Plan
- [x] Add/extend report-level test for right-side guidance panel text.
- [x] Reduce plot width and adjust margins for thinner chart rendering.
- [x] Add right-side “How to read these charts” business narrative panel.
- [x] Ensure responsive behavior with media query.
- [x] Run pytest and regenerate report.

## Architecture Notes
- Public API unchanged.
- HTML shell adds a `content-grid` container with figure and narrative aside.

## Test Plan

### Automated tests (what/where)
- `tests/test_evaluate_model.py`:
  - assert generated HTML includes `"How to read these charts"` and business guidance keywords.

### Manual verification script
- Run: `python evaluate_model.py`
- Open report and verify chart is narrower, with explanatory panel on right.

## Progress Log
- 2026-02-25: Opened Slice 6 for chart width tuning and business interpretation panel.
- 2026-02-25: Added report-level test expectation for `"How to read these charts"` content.
- 2026-02-25: Reduced Plotly figure width and shifted layout to two-column report content.
- 2026-02-25: Added right-side business interpretation panel with practical guidance for gain/success rate/cutoff lines.
- 2026-02-25: Added responsive media behavior so guide panel stacks below charts on smaller screens.
- 2026-02-25: Verified with `python -m pytest -q` (3 passed) and regenerated report.

## Final Summary
Slice 6 completed. Charts are narrower and paired with a right-side business-oriented reading guide, producing a clearer and more presentation-ready layout.

---

# Slice 7 - Dual Controls + Bar Success Chart + Matched Guide Panels

## Task Contract

### Goal
Further reduce chart footprint, widen and split the explanation panel by chart row, default selected cutoff to KS optimum, add captured-success percentage, and convert cumulative success plot to an interactive threshold-colored bar chart with desired-success-rate control.

### Acceptance Criteria
- Selected cutoff defaults to KS-optimal percentile.
- Captured successes KPI includes both count and percentage.
- Bottom chart is a bar chart (not line) and remains interactive.
- Add desired-success-rate control that returns required cutoff percentile.
- Bottom bars are colored by desired-success-rate threshold, with intensity varying by distance to threshold.
- Right-side explanatory panel is wider, split into top/bottom sections aligned with chart rows.
- Charts are narrower than current layout.

### Non-Goals
- Changing event/success calculation logic.

### Assumptions / Open Questions
- Assumption: "required cutoff for desired success rate" means the largest cutoff percentile where cumulative success rate is still at or above desired target.

## Strategic Plan
- Add tests for new controls and bar chart output.
- Update figure traces/layout and JS interactivity for dual controls.
- Update side guidance layout to row-aligned top/bottom explanation cards.

## Tactical Plan
- [x] Add/extend tests for desired-rate control and bar-chart rendering.
- [x] Set default selected cutoff to KS optimum and include captured percentage in KPI.
- [x] Convert success-rate subplot to bars and add threshold-driven color shading.
- [x] Add desired-success-rate control and computed required-cutoff output.
- [x] Split/widen right-side guide into top/bottom chart-specific sections.
- [x] Run pytest and regenerate report.

## Architecture Notes
- Public API unchanged.
- HTML/JS includes two controls:
  - selected cutoff percentile
  - desired success rate
- JS updates:
  - selected cutoff KPI + marker/callout
  - required cutoff output for desired rate
  - success-rate bar colors by distance from desired threshold.

## Test Plan

### Automated tests (what/where)
- `tests/test_evaluate_model.py`:
  - assert desired-rate slider/control exists in generated HTML
  - assert success-rate trace type is bar
  - assert report contains required-cutoff output hook.

### Manual verification script
- Run: `python evaluate_model.py`
- Open report:
  - move selected cutoff slider and confirm marker + KPI update
  - move desired success rate slider and confirm required cutoff + bar recoloring update.

## Progress Log
- 2026-02-25: Opened Slice 7 for dual-control business interactivity and bar-chart redesign.
- 2026-02-25: Added tests to assert desired-rate control markup and bar-type success trace.
- 2026-02-25: Set default selected cutoff to KS optimum and added captured-success percentage to KPI output.
- 2026-02-25: Converted cumulative success subplot to bar chart with threshold-relative shading.
- 2026-02-25: Added desired success-rate slider with live required-cutoff computation and marker update.
- 2026-02-25: Narrowed figure further and widened/split right-side explanation into top and bottom chart sections.
- 2026-02-25: Verified with `python -m pytest -q` (3 passed) and regenerated report.

## Final Summary
Slice 7 completed. The report now defaults to KS-optimal cutoff, includes dual interactive controls (selected cutoff + desired success rate), uses a threshold-shaded success-rate bar chart with required-cutoff feedback, and presents a tighter chart area with wider split business guidance aligned to top and bottom charts.

---

# Slice 8 - Bidirectional Control Sync + Vertically Centered Guide

## Task Contract

### Goal
Synchronize selected-cutoff and desired-success-rate controls bidirectionally, and vertically center right-side chart descriptions within each chart-aligned panel.

### Acceptance Criteria
- Changing selected cutoff recalculates and updates desired success rate.
- Changing desired success rate recalculates and updates selected cutoff.
- Controls remain consistent/matched after every interaction.
- Right-side top/bottom description sections are vertically centered to chart rows.

### Non-Goals
- Reworking chart semantics or KPI definitions.

### Assumptions / Open Questions
- Assumption: synced mapping is exact to cumulative rate at the selected cutoff and required cutoff for desired rate.

## Strategic Plan
- Add report-level test assertion for sync hooks.
- Implement synchronized event handlers in client JS.
- Adjust guide-panel section flex alignment to vertical center.

## Tactical Plan
- [x] Add/extend tests for synchronized control hooks in HTML script.
- [x] Implement cutoff -> desired-rate synchronization.
- [x] Implement desired-rate -> cutoff synchronization.
- [x] Vertically center top/bottom guide sections.
- [x] Run pytest and regenerate report.

## Architecture Notes
- Public API unchanged.
- JS introduces dedicated sync handlers to avoid drift between controls.

## Test Plan

### Automated tests (what/where)
- `tests/test_evaluate_model.py`:
  - assert generated HTML includes sync statements for both control directions.

### Manual verification script
- Move cutoff slider and confirm desired rate updates.
- Move desired rate slider and confirm selected cutoff updates.

## Progress Log
- 2026-02-25: Opened Slice 8 for bidirectional control synchronization and centered guide sections.
- 2026-02-25: Added report-level test assertions for both sync directions in embedded JS hooks.
- 2026-02-25: Implemented cutoff-to-desired synchronization (`desiredRateSlider` updates when cutoff changes).
- 2026-02-25: Implemented desired-to-cutoff synchronization (required cutoff recalculates selected cutoff).
- 2026-02-25: Vertically centered guide sections to align with top and bottom chart rows.
- 2026-02-25: Verified with `python -m pytest -q` (3 passed) and regenerated report.

## Final Summary
Slice 8 completed. The selected cutoff and desired success-rate controls are now fully synchronized in both directions, and the right-side chart explanations are vertically centered within their chart-aligned panels.

---

# Slice 9 - Desired-Rate Color Semantics (Chosen vs Unused)

## Task Contract

### Goal
When desired success rate changes, recolor bottom bars so chosen percentiles are blue and unused percentiles are shadow gray.

### Acceptance Criteria
- Desired-rate interaction recolors bars by required cutoff boundary.
- Percentiles up to required cutoff are blue.
- Percentiles above required cutoff are shadow gray.
- Initial render uses the same color semantics.

### Non-Goals
- Changing the cutoff/desired-rate synchronization logic.

## Strategic Plan
- Add test assertion for new color-classification JS hook.
- Update figure default bar colors to chosen/unused scheme.
- Update client-side recoloring logic used by desired-rate interaction.

## Tactical Plan
- [x] Add test for chosen-vs-unused color hook.
- [x] Update Python-side default bar colors to blue/shadow by required cutoff.
- [x] Update JS recoloring to use required-cutoff boundary.
- [x] Run pytest and regenerate report.

## Progress Log
- 2026-02-25: Opened Slice 9 for chosen-vs-unused bar coloring based on desired-rate interaction.
- 2026-02-25: Added report-level assertion for required-cutoff-based color hook in HTML JS.
- 2026-02-25: Switched default bar palette to required-cutoff semantics (blue for chosen, shadow gray for unused).
- 2026-02-25: Updated desired-rate interaction recolor logic to classify by percentile boundary (`point.p <= required`).
- 2026-02-25: Updated guide copy to reflect blue/gray bar semantics.
- 2026-02-25: Verified with `python -m pytest -q` (3 passed) and regenerated report.

## Final Summary
Slice 9 completed. Desired-rate interactions now recolor the bottom chart with blue for chosen percentiles and shadow gray for unused percentiles, both on initial render and during live updates.

---

# Slice 10 - Hard Sync for Cutoff Markers

## Task Contract

### Goal
Ensure selected-cutoff and desired-rate cutoff markers always align to the same percentile with no drift.

### Acceptance Criteria
- After any control interaction, selected cutoff marker and required-cutoff marker share the same x-position.
- Desired rate display is snapped to selected cutoff's actual cumulative success rate.
- No visual mismatch between the two vertical markers.

## Tactical Plan
- [x] Add regression assertion for hard-sync JS hook.
- [x] Update JS flow so selected cutoff is authoritative and required marker follows it.
- [x] Run pytest and regenerate report.

## Progress Log
- 2026-02-25: Opened Slice 10 to enforce strict marker alignment.
- 2026-02-25: Added regression check for hard-sync statement (`const required = point.p;`) in generated HTML script.
- 2026-02-25: Updated control-sync flow so cutoff interaction forces required-cutoff marker to same percentile.
- 2026-02-25: Verified with `python -m pytest -q` (3 passed) and regenerated report.

## Final Summary
Slice 10 completed. Selected-cutoff and desired-rate cutoff markers are now hard-synchronized to the same percentile after interactions, eliminating line mismatch.

---

# Slice 11 - Campaign Selection Comparison Mode

## Task Contract

### Goal
Add an optional campaign-selection mode where users pass selected client keys and get additional charts estimating potential success performance for that selection, while still seeing the original two charts.

### Acceptance Criteria
- `EvaluateModel` accepts a better-named campaign-selection DataFrame parameter (with `pt_unified_key`).
- Default mode (`False`) renders current charts only.
- Compare mode (`True` + valid selection df) renders original charts plus additional selection-potential comparison charts.
- Selection mode clearly shows:
  - success rate of selected clients
  - model-guided success rate at same volume
  - overall baseline.

### Tactical Plan
- [x] Add tests for campaign-selection mode HTML output.
- [x] Extend API with campaign-selection toggle + DataFrame input validation.
- [x] Implement selection summary computation and comparison figure.
- [x] Embed additional section in report when compare mode is enabled.
- [x] Run pytest and regenerate report.

## Progress Log
- 2026-02-25: Opened Slice 11 for optional campaign-selection comparison reporting.
- 2026-02-25: Added test covering campaign selection section rendering in compare mode.
- 2026-02-25: Extended `EvaluateModel` signature with `include_campaign_selection` and `campaign_clients`.
- 2026-02-25: Implemented campaign selection summary metrics and a dedicated comparison figure.
- 2026-02-25: Embedded campaign potential section while preserving original two charts.
- 2026-02-25: Verified with `python -m pytest -q` (4 passed) and regenerated report.

## Final Summary
Slice 11 completed. Report now supports optional campaign-selection comparison mode with additional potential-success benchmarking charts, while keeping the original two charts for direct comparison.

---

# Slice 12 - Notebook Integration for Campaign Selection Mode

## Task Contract

### Goal
Update the testing notebook to call `EvaluateModel` with campaign selection dataframe input and produce the compare-mode report.

### Tactical Plan
- [x] Rename notebook variable from `situation` to `campaign_clients` in source cells.
- [x] Update notebook report-generation cell to run both baseline and campaign-selection modes.
- [x] Update notebook display cell to link baseline and campaign-selection reports.
- [x] Run regression tests for codebase integrity.

## Progress Log
- 2026-02-25: Updated `notebooks/simulated_data_testing.ipynb` to use `campaign_clients` naming and call `EvaluateModel(..., include_campaign_selection=True, campaign_clients=campaign_clients)`.
- 2026-02-25: Notebook now generates `model_evaluation_report_campaign_selection.html` alongside baseline output.

---

# Slice 13 - Actual (Latest) vs User-Period Historical Comparison

## Task Contract

### Goal
Extend simulated scoring data to include monthly snapshots with January 2026 as newest actual scores, and compare campaign selected-client potential success rate on actual scores versus a user-defined historical scoring period.

### Acceptance Criteria
- Simulated `model_score` includes multiple `fs_time` snapshots and newest snapshot is January 2026.
- Main report charts are computed from newest actual scores only.
- In campaign-selection mode, selected-client success rate is computed on actual scores and compared to user-defined historical period.
- Campaign comparison chart/section displays explicit actual/historical period labels.

### Tactical Plan
- [x] Add tests for monthly snapshots with latest January 2026.
- [x] Add tests for historical period labels in campaign-selection report output.
- [x] Update simulated data generator to produce monthly score snapshots.
- [x] Add historical-period parameters and period filtering to `EvaluateModel`.
- [x] Implement actual-vs-historical campaign comparison summaries and chart labels.
- [x] Run pytest and regenerate report.

## Progress Log
- 2026-02-25: Opened Slice 13 for latest-vs-historical campaign comparison with user-defined historical window.
- 2026-02-25: Added tests for multi-snapshot simulated scores and campaign report period labeling.
- 2026-02-25: Updated `create_simulated_tables` to emit monthly score snapshots (Oct 2025 through Jan 2026), with Jan 2026 as newest actual scores.
- 2026-02-25: Updated `EvaluateModel` baseline charts to compute from newest actual snapshot only.
- 2026-02-25: Added historical-period parameters (`historical_period_start`, `historical_period_end`) and validated filtering logic.
- 2026-02-25: Implemented campaign comparison charts for actual vs historical score periods, including explicit period labels in section and chart legends.
- 2026-02-25: Updated notebook validation assertion for new multi-month target simulation behavior.
- 2026-02-25: Verified with `python -m pytest -q` (6 passed) and `python evaluate_model.py`.

## Final Summary
Slice 13 completed. The report now treats January 2026 as newest actual model scores, computes baseline charts from that snapshot, and supports campaign selection comparison against a user-defined historical scoring period with explicit time-period labels in the comparison visuals.

---

# Slice 14 - Notebook Single Historical Date + Clearer Campaign Comparison Chart

## Task Contract

### Goal
Set notebook campaign comparison to one historical period date (`2025-12-31`) and improve campaign comparison chart readability, especially the second success-rate chart.

### Tactical Plan
- [x] Update notebook `EvaluateModel` campaign call to fixed historical start/end = `2025-12-31`.
- [x] Redesign second campaign comparison chart for clearer business interpretation.
- [x] Keep actual/historical period labeling visible in comparison visuals.
- [x] Run pytest and regenerate report.

## Progress Log
- 2026-02-25: Opened Slice 14 for notebook historical-date configuration and campaign chart clarity improvements.
- 2026-02-25: Updated notebook campaign-mode call to use single historical date (`2025-12-31` start/end).
- 2026-02-25: Redesigned second campaign comparison chart to remove confusing horizontal reference and emphasize campaign volume marker plus actual/historical model SR points.
- 2026-02-25: Preserved explicit actual/historical period labels in section and chart legends.
- 2026-02-25: Fixed hover-template interpolation issue and verified with `python -m pytest -q` (6 passed) and report regeneration.

## Final Summary
Slice 14 completed. Notebook now uses one historical period (`2025-12-31`), and the campaign comparison second chart is simplified for clearer business interpretation.

---

# Slice 15 - Notebook Execution Health Fix

## Task Contract

### Goal
Run the full testing notebook, detect execution errors, and fix failing cells so notebook executes cleanly end-to-end.

### Tactical Plan
- [x] Execute notebook in-place with nbconvert to surface real errors.
- [x] Fix failing notebook assertions for current simulated-data behavior.
- [x] Re-execute notebook and confirm no execution errors.
- [x] Run pytest regression check.

## Progress Log
- 2026-02-25: Executed `notebooks/simulated_data_testing.ipynb` with nbconvert and found failing assertion expecting events only in Feb 2026.
- 2026-02-25: Updated assertion to valid multi-month event range (`2025-11-01` to `2026-02-28 23:59:59`).
- 2026-02-25: Re-executed notebook successfully and confirmed test suite remains green (`6 passed`).
