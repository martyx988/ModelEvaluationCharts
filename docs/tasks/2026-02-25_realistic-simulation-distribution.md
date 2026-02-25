# Task Contract

## Goal
Improve data simulation realism so model scores and target successes follow a plausible predictive-model distribution.

## Acceptance Criteria
- `model_score.score` remains in `0..1`.
- `model_score.percentile` remains in `1..100`.
- Score distribution is non-uniform and more realistic than uniform random.
- `target_store` successes are distributed with higher likelihood among higher-score clients.
- Existing table sizes and schemas remain unchanged.

## Non-Goals
- Exact calibration to any real production portfolio.
- Changing output schema or row counts.

## Assumptions / Open Questions
- Assumption: realism here means monotonic relationship between score and event likelihood plus skewed risk distribution.

# Strategic Plan
- Replace uniform score generation with latent-risk-based simulation.
- Replace random target sampling with score-weighted sampling.
- Run smoke checks for ranges, shapes, and target-vs-population score separation.

# Tactical Plan
- [x] Update score simulation in `simulated_data.py`.
- [x] Update target sampling to be score-probability weighted.
- [x] Validate score/percentile bounds and table shapes.
- [x] Validate target mean score exceeds population mean.

# Architecture Notes
- Score generation:
  - latent risk sampled from beta distribution (skewed toward lower risk)
  - additive noise applied
  - clipped to `[0, 1]`
- Target generation:
  - weighted sampling without replacement
  - weights proportional to `score^2.8` to amplify high-score concentration

# Test Plan

## Automated tests (what/where)
- Planned follow-up: add `pytest` checks for monotonic enrichment by score deciles.

## Manual verification script
- Run:
  - `python -c "from simulated_data import create_simulated_tables; ..."`
- Verify:
  - score bounds in `[0, 1]`
  - percentile bounds in `[1, 100]`
  - target mean score > population mean

# Progress Log
- 2026-02-25: Implemented risk-based non-uniform score simulation.
- 2026-02-25: Implemented score-weighted target success sampling.
- 2026-02-25: Smoke checks passed for bounds, shapes, and enrichment signal.

# Final Summary
Simulation now better reflects realistic model behavior: most clients are lower risk, and observed successes are enriched among higher scores.
