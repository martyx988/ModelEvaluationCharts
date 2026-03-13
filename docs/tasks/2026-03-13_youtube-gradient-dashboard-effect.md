# Task Contract

## Goal
Align the final HTML dashboard to the provided dark neon screenshot by refining the page background, animated wrapper border/glow, and the internal dashboard color theme while keeping the existing content and structure.

## Acceptance Criteria
- The generated HTML dashboard includes an animated gradient border on the main wrapper/container only.
- The animated border uses a lightweight CSS pseudo-element approach so layout does not shift.
- The page background and wrapper colors match the dark navy neon style from the provided screenshot more closely.
- Internal dashboard surfaces, text, controls, and charts use a coherent dark theme that fits the wrapper effect.
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
- Open question: none blocking for this slice.

# Strategic Plan
- Add regression coverage for the screenshot-aligned dark palette in the HTML shell and figures.
- Update the dashboard wrapper CSS to match the dark neon screenshot more closely.
- Update internal dashboard and chart colors so the full report feels stylistically consistent.
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
- [ ] Implement the dark neon HTML/CSS and chart theme updates.
- [ ] Run `pytest` and regenerate the HTML report.
- [ ] Visually inspect the rendered report and refine if needed.
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
  - Use a dark navy background and slightly lighter blue report surface inspired by the screenshot.
  - Carry the neon palette into charts and controls in a restrained way so readability remains high.
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
  - the page background and wrapper surface match the dark neon screenshot direction
  - inner content remains fully readable and unchanged in layout
  - charts and controls feel visually consistent with the dark wrapper theme
  - controls and charts remain readable and interactive
  - no external asset dependency is required for styling

# Progress Log
- 2026-03-13: Intake completed; confirmed YouTube page access and inferred the requested effect from the video title and preview frame.
- 2026-03-13: Planner opened this task file and documented the slice contract, assumptions, and verification plan.
- 2026-03-13: User clarified that the effect should apply only to the main wrapper and should be an animated gradient border via pseudo-elements, not a broader shell/card restyle.
- 2026-03-13: User supplied a more exact CSS reference using `@property --angle`, `conic-gradient(from var(--angle))`, and paired `::before`/`::after` layers; implementation should now follow that pattern more directly.
- 2026-03-13: User further clarified that the magenta/cyan palette should match the new sample and that only the wrapper border ring, not the wrapper interior, should appear animated.
- 2026-03-13: User requested a broader screenshot-aligned theme pass so the page background, wrapper surface, and dashboard colors all fit the dark neon style.
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

# Final Summary
Completed. The final HTML dashboard now uses a screenshot-aligned dark neon theme: a deep navy page background, a blue-toned wrapper surface, a magenta/cyan animated border ring and glow, and coordinated dark cards, controls, and Plotly figures. The existing dashboard structure and interactions were preserved, regression coverage was expanded for the dark palette, tests passed, the report was regenerated successfully, and the rendered result was visually reviewed in-browser.
