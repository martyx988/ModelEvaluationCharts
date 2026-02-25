# Task Contract

## Goal
Create a Python function that generates three simulated pandas DataFrames for model evaluation work where real data cannot be uploaded.

## Acceptance Criteria
- A callable function exists that returns exactly three pandas DataFrames: `model_score`, `target_store`, and `situation`.
- `model_score` has 10,000 clients with `pt_unified_key` values from 1 to 10000 and columns: `pt_unified_key`, `fs_time`, `score`, `percentile`.
- `model_score.fs_time` is set to `2026-01-31` for all rows.
- `target_store` has 1,000 random unique clients sampled from the same 10,000 and columns: `pt_unified_key`, `atsp_event_timestamp`.
- `target_store.atsp_event_timestamp` values are random datetimes within February 2026.
- `situation` has 7,000 random unique clients sampled from the same 10,000 and column: `pt_unified_key`.

## Non-Goals
- Building the HTML report and Plotly charts.
- Persisting generated data to external storage or databases.
- Matching any real production distribution beyond simple random simulation.

## Assumptions / Open Questions
- Assumption: pseudo-random values are acceptable for this phase.
- Assumption: reproducibility is preferred, so the function supports a random seed parameter.
- Open question: none blocking for this slice.

# Strategic Plan
- Define a small standalone data simulation API.
- Implement table generation logic with deterministic seed support.
- Run a local smoke check to verify shapes and required columns.

# Tactical Plan
- [x] Create Python module for simulated table generation.
- [x] Implement `create_simulated_tables(seed=...)` returning three DataFrames.
- [x] Implement schema and row counts per contract.
- [x] Add a simple executable smoke section for local verification.
- [x] Run smoke execution and confirm expected dimensions.

# Architecture Notes
- Module placement: repository root for now (`simulated_data.py`) to keep the slice minimal.
- Public API: `create_simulated_tables(seed: int | None = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`.
- Inputs: optional integer seed for reproducibility.
- Outputs: tuple in fixed order: `(model_score, target_store, situation)`.
- Strategy:
  - Generate base client universe as integers 1..10000.
  - Generate `score` as uniform random values.
  - Generate `percentile` as rank-based percent values from score.
  - Sample without replacement for `target_store` and `situation`.
  - Generate event timestamps by random second offsets between `2026-02-01` and `2026-03-01`.

# Test Plan

## Automated tests (what/where)
- Planned next slice: add `pytest` tests in `tests/test_simulated_data.py` for:
  - row counts
  - required columns
  - key ranges and uniqueness constraints
  - timestamp month boundaries

## Manual verification script
- Run: `python simulated_data.py`
- Verify console output shows:
  - `model_score shape: (10000, 4)`
  - `target_store shape: (1000, 2)`
  - `situation shape: (7000, 1)`

# Progress Log
- 2026-02-25: Intake completed. Defined measurable acceptance criteria from user request.
- 2026-02-25: Implemented `simulated_data.py` with `create_simulated_tables`.
- 2026-02-25: Executed smoke run and verified expected DataFrame shapes.
- 2026-02-25: Added required task definition file under `docs/tasks/`.

# Final Summary
Slice complete for simulated source tables. The repository now contains a callable data generator function and documented task contract, plan, and evidence for this phase.
