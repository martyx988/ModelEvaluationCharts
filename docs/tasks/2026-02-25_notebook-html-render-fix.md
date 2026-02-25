# Task Contract

## Goal
Fix notebook report preview so generated HTML report is visible during manual testing.

## Acceptance Criteria
- Notebook no longer relies on `IFrame(src=...)` local path loading.
- Notebook includes a reliable inline HTML render mechanism.
- Notebook also shows a clickable link to open generated report directly.

## Non-Goals
- Refactoring report generation function internals.
- Changing chart logic or report business metrics.

## Assumptions / Open Questions
- Assumption: VS Code notebook local-file iframe handling is restricted/inconsistent.

# Strategic Plan
- Replace iframe rendering with direct HTML display.
- Add file link fallback.
- Commit notebook-only fix.

# Tactical Plan
- [x] Update imports to include `HTML`, `FileLink`, and `display`.
- [x] Replace iframe cell with link + inline HTML rendering.
- [x] Commit changes.

# Architecture Notes
- Rendering method:
  - `display(FileLink(report_path))`
  - `display(HTML(report_path.read_text(encoding="utf-8")))`
- This avoids local iframe path restrictions in notebook frontends.

# Test Plan

## Automated tests (what/where)
- Not in this slice.

## Manual verification script
- Run all cells in `notebooks/simulated_data_testing.ipynb`.
- Confirm:
  - clickable report link appears
  - report HTML renders inline in notebook output

# Progress Log
- 2026-02-25: Replaced iframe preview with direct HTML rendering and file-link fallback.

# Final Summary
Notebook preview reliability improved for VS Code Jupyter by avoiding local-file iframe rendering.
