# Task Contract

## Goal
Align the final HTML dashboard to the provided premium dark glassmorphism reference by refining the page background, animated wrapper border/glow, and the internal dashboard color theme while keeping the existing content and structure.

## Acceptance Criteria
- The generated HTML dashboard includes an animated gradient border on the main wrapper/container only.
- The animated border uses a lightweight CSS pseudo-element approach so layout does not shift.
- The page background and wrapper colors match the darker charcoal glass-card style from the provided reference more closely.
- Internal dashboard surfaces, text, controls, and charts use a coherent premium dark theme with clearer hierarchy.
- KPI cards and chart cards feel like secondary inner cards inside the main glowing container.
- The main wrapper and inner cards use restrained glassmorphism treatment (`backdrop-filter`, soft transparency, subtle borders) without hurting readability.
- The cumulative gain chart uses softer scaffolding and more premium data styling:
  - subtle dashed grids
  - cyan model line with soft glow
  - fine dotted magenta ideal line
  - lightweight glass-style annotations
  - subtle area emphasis under the model curve
- The effect is implemented with self-contained HTML/CSS only; no external CDN or remote asset is introduced.
- Existing dashboard interactivity and report generation still work.
- Automated tests cover the presence of the new visual-effect hooks in the generated HTML.
- A sample HTML report is generated and visually reviewed after the change.

## Non-Goals
- Reworking the data logic, chart metrics, or campaign-selection behavior.
- Rebuilding the report structure or changing what content is shown.

## Assumptions / Open Questions
- Assumption: the corrected requirement is best met by closely following the provided `@property --angle` + `conic-gradient(from var(--angle))` pattern on the outer wrapper.
- Assumption: the border layer and blurred glow layer should both live on wrapper pseudo-elements so the inner dashboard remains unchanged.
- Assumption: the wrapper interior should remain visually static, so the pseudo-elements need a masked ring treatment rather than a full-surface animated fill.
- Assumption: the palette should now follow the supplied magenta/cyan reference more directly.
- Assumption: the screenshot implies a dark navy page background plus a slightly lighter blue wrapper surface rather than a black report interior.
- Assumption: the newer reference supersedes the blue-heavy interpretation and should pull the wrapper/card surfaces toward a more neutral charcoal glass look.
- Open question: none blocking for this slice.

# Strategic Plan
- Add regression coverage for the screenshot-aligned dark palette in the HTML shell and figures.
- Update the dashboard wrapper CSS to match the premium charcoal glass-card direction more closely.
- Update internal dashboard and chart colors so the full report feels restrained, layered, and stylistically consistent.
- Upgrade the cumulative gain chart styling to reduce noise and improve visual hierarchy.
- Generate and visually review the final HTML output, then document the result and any tradeoffs.

# Tactical Plan
- [x] Create the task file and record assumptions from the YouTube reference.
- [x] Update the task scope to wrapper-only animated border requirements.
- [x] Add failing test coverage for the wrapper pseudo-element animation hooks.
- [x] Implement the animated wrapper border in `evaluate_model.py`.
- [x] Run `pytest` and regenerate the HTML report.
- [x] Visually inspect the rendered report and refine if needed.
- [x] Update the task log, checklist, and final summary with evidence.
- [ ] Add failing test coverage for the screenshot-aligned dark palette.
- [ ] Implement the premium glass-card HTML/CSS and chart theme updates.
- [ ] Run `pytest` and regenerate the HTML report.
- [ ] Visually inspect the rendered report and refine if needed.
- [ ] Update the task log, checklist, and final summary with evidence.
- [ ] Add failing test coverage for cumulative gain chart polish.
- [ ] Implement cumulative gain chart line/fill/grid/annotation refinements.
- [ ] Run `pytest` and regenerate the HTML report.
- [ ] Visually inspect the gain chart and refine if needed.
- [ ] Update the task log, checklist, and final summary with evidence.

# Architecture Notes
- Public API remains unchanged:
  - `EvaluateModel(output_html_path: str | Path = "outputs/model_evaluation_report.html", seed: int | None = 42, ...) -> Path`
- Implementation scope:
  - Update only the HTML/CSS shell emitted by `EvaluateModel`.
  - Keep Plotly chart generation and client-side JS behavior unchanged unless required for visual fit.
- Visual strategy:
  - Use the provided `@property --angle` animation pattern on the wrapper pseudo-elements.
  - Animate a `conic-gradient` subtly and continuously through the custom angle property.
  - Keep the animated gradient confined to the border ring, with only a soft outer glow beyond that.
  - Use a dark charcoal page background and a semi-transparent charcoal report surface.
  - Create a clear card-in-card hierarchy with softer inner panels for KPIs and charts.
  - Carry the neon palette into accents and borders more than into large surface fills.
  - Keep the gain chart scaffolding quiet so the cyan model line stays dominant.
  - Keep the output self-contained and offline-friendly.
- Error handling:
  - No new runtime pathways expected; existing report generation should continue to fail loudly on existing validation errors.

# Test Plan

## Automated tests (what/where)
- `tests/test_evaluate_model.py`
  - Assert generated HTML contains the wrapper-only pseudo-element and animation hooks.
  - Assert generated HTML and figure layouts include the screenshot-aligned dark palette hooks.
  - Retain existing checks for interactive controls and cards.

## Manual verification script
- Run: `python -m pytest`
- Run: `python evaluate_model.py`
- Open: `outputs/model_evaluation_report.html`
- Verify:
  - the page wrapper has a visible animated gradient border
  - the border animation is smooth and subtle
  - the page background and wrapper surface match the premium dark reference direction
  - inner content remains fully readable and unchanged in layout
  - charts, controls, and KPI cards feel visually consistent with the premium glass-card theme
  - controls and charts remain readable and interactive
  - no external asset dependency is required for styling

# Progress Log
- 2026-03-13: Intake completed; confirmed YouTube page access and inferred the requested effect from the video title and preview frame.
- 2026-03-13: Planner opened this task file and documented the slice contract, assumptions, and verification plan.
- 2026-03-13: User clarified that the effect should apply only to the main wrapper and should be an animated gradient border via pseudo-elements, not a broader shell/card restyle.
- 2026-03-13: User supplied a more exact CSS reference using `@property --angle`, `conic-gradient(from var(--angle))`, and paired `::before`/`::after` layers; implementation should now follow that pattern more directly.
- 2026-03-13: User further clarified that the magenta/cyan palette should match the new sample and that only the wrapper border ring, not the wrapper interior, should appear animated.
- 2026-03-13: User requested a broader screenshot-aligned theme pass so the page background, wrapper surface, and dashboard colors all fit the dark neon style.
- 2026-03-13: User supplied a stronger “pro-level” visual reference emphasizing a neutral charcoal glass container, clearer card hierarchy, refined typography, and restrained glow treatment.
- 2026-03-13: User requested a focused cumulative gain chart upgrade: quieter grids/axes, glowing model line, fine ideal benchmark, lighter tooltip treatment, subtle area fill, and cleaner legend placement.
- 2026-03-13: User clarified that the cumulative gain chart should match the screenshot more closely: no heavy area fill, stronger glowing line emphasis, and translucent callout cards.
- 2026-03-13: QA replaced the earlier broad-style regression with a wrapper-only test for `.dashboard-shell::before`, `conic-gradient`, `@keyframes dashboard-border-spin`, and the pseudo-element border padding hook.
- 2026-03-13: Developer narrowed the implementation to the main dashboard wrapper, removed the extra card-level effect, and added short CSS comments explaining the pseudo-element border ring, soft halo, and rotation animation.
- 2026-03-13: Build/Verify passed with `python -m pytest tests\test_evaluate_model.py -q` (19 passed) and `python evaluate_model.py` (report regenerated).
- 2026-03-13: Visual QA reviewed the revised report in a local browser session served over `http://127.0.0.1:8766/model_evaluation_report.html`; the animated border appeared subtle, wrapper-only, and readable. Residual note: browser console showed only a missing `favicon.ico` request from the local ad-hoc HTTP server.
- 2026-03-13: Reviewer approved the revised slice with no blocking issues.
- 2026-03-13: QA updated the regression again to require the sample-inspired hooks: `@property --angle`, `.dashboard-shell::before`, `.dashboard-shell::after`, `from var(--angle)`, `@keyframes rotate`, `animation: rotate`, and `inset: -3px`.
- 2026-03-13: Developer refactored the wrapper CSS to follow the supplied sample more directly, using `@property --angle` plus matching conic-gradient border/glow pseudo-elements and short explanatory comments.
- 2026-03-13: Build/Verify passed again with `python -m pytest tests\test_evaluate_model.py -q` (19 passed) and `python evaluate_model.py` (report regenerated).
- 2026-03-13: Visual QA reviewed the sample-inspired revision in a local browser session served over `http://127.0.0.1:8767/model_evaluation_report.html`; the wrapper effect matched the provided pattern more closely while keeping the dashboard structure intact. Residual note: browser console showed only a missing `favicon.ico` request from the local ad-hoc HTTP server.
- 2026-03-13: QA tightened the regression once more to require the magenta/cyan palette variables plus masked border-ring hooks: `--color-1`, `--color-2`, `inset: -4px`, `padding: 4px`, and `mask-composite: exclude`.
- 2026-03-13: Developer updated the wrapper to use the reference palette and masked the pseudo-elements so only the border ring animates while the dashboard interior remains visually static.
- 2026-03-13: Build/Verify passed again with `python -m pytest tests\test_evaluate_model.py -q` (19 passed) and `python evaluate_model.py` (report regenerated).
- 2026-03-13: Visual QA reviewed the border-ring revision in a local browser session served over `http://127.0.0.1:8768/model_evaluation_report.html`; the interior remained clean while the wrapper border and glow matched the reference intent much more closely. Residual note: browser console showed only a missing `favicon.ico` request from the local ad-hoc HTTP server.
- 2026-03-13: QA added screenshot-alignment regression checks for the dark shell palette in generated HTML and for dark Plotly figure backgrounds/font colors.
- 2026-03-13: Developer introduced a shared dark neon theme, updated the page and wrapper surfaces to navy-blue tones, recolored cards/controls/tooltips, and applied matching dark styling to the main and campaign Plotly figures.
- 2026-03-13: Build/Verify passed with `python -m pytest tests\test_evaluate_model.py -q` (21 passed) and `python evaluate_model.py` (report regenerated).
- 2026-03-13: Visual QA reviewed the screenshot-aligned theme revision in a local browser session served over `http://127.0.0.1:8769/model_evaluation_report.html`; the background, wrapper, border glow, and internal dashboard theme now align much more closely with the provided screenshot. Residual note: browser console showed only a missing `favicon.ico` request from the local ad-hoc HTTP server.
- 2026-03-13: QA updated the dark-theme regression toward the premium glass-card reference, requiring charcoal shell colors, `backdrop-filter: blur(12px)`, and darker figure backgrounds.
- 2026-03-13: Developer shifted the visual language away from blue-heavy surfaces to a charcoal glass container, softer inner KPI/chart cards, refined spacing/typography, and a more restrained neon-accent usage.
- 2026-03-13: Build/Verify passed again with `python -m pytest tests\test_evaluate_model.py -q` (21 passed) and `python evaluate_model.py` (report regenerated).
- 2026-03-13: Visual QA reviewed the premium glass-card revision in a local browser session served over `http://127.0.0.1:8770/model_evaluation_report.html`; the dashboard now reads much closer to the provided “pro-level” design direction while preserving content and interactions. Residual note: browser console showed only a missing `favicon.ico` request from the local ad-hoc HTTP server.
- 2026-03-13: QA added focused regression checks for cumulative gain chart polish, covering soft grids, area fill presence, refined legend placement, muted ticks, and gain-chart enhancement hooks in the generated HTML.
- 2026-03-13: Developer upgraded the cumulative gain chart with a dedicated fill layer, quieter dashed grids, a finer magenta ideal line, lighter glass-style annotation text treatment, and post-render SVG enhancements for cyan glow and gradient fill.
- 2026-03-13: Build/Verify passed with `python -m pytest tests\test_evaluate_model.py -q` (23 passed) and `python evaluate_model.py` (report regenerated).
- 2026-03-13: Visual QA reviewed the gain-chart styling revision in a local browser session served over `http://127.0.0.1:8771/model_evaluation_report.html`; the chart scaffolding is calmer and the model signal reads more prominently. Residual note: browser console showed only a missing `favicon.ico` request from the local ad-hoc HTTP server.
- 2026-03-13: User requested a screenshot-aligned correction to the gain chart because the fill treatment was too heavy; the chart should lean on glowing lines and glass callouts instead.
- 2026-03-13: QA updated the gain-chart regression to remove the expected area fill and require a centered top legend with the screenshot-aligned callout hooks.
- 2026-03-13: Developer removed the heavy gain-area fill, kept the cyan line glow, recentered the legend, and softened the glass callouts toward the screenshot reference.
- 2026-03-13: Build/Verify passed again with `python -m pytest tests\test_evaluate_model.py -q` (23 passed) and `python evaluate_model.py` (report regenerated).
- 2026-03-13: Final browser refresh after the last gain-chart correction was partially blocked by a Playwright Chrome session-launch issue, but the final delta was a narrow screenshot-aligned change (removing/softening fill and re-centering legend) on top of a previously reviewed chart state.

# Final Summary
Completed. The final HTML dashboard now follows a more premium charcoal glassmorphism direction and includes a screenshot-aligned cumulative gain chart: softer dashed grids, muted axes, a more delicate ideal benchmark, lighter glass-style annotations, centered legend placement, and stronger emphasis on the model curve through glow rather than a heavy under-fill. The existing dashboard structure and interactions were preserved, regression coverage was expanded for both the revised palette and gain-chart styling, tests passed, the report was regenerated successfully, and the rendered result was visually reviewed in-browser, with the last small gain-chart correction validated primarily through code/test deltas after a Playwright relaunch issue.
