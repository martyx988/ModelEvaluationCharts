from __future__ import annotations

import numpy as np
import pandas as pd


def create_simulated_tables(seed: int | None = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create simulated model data tables for local development.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        model_score, target_store, campaign_clients
    """
    rng = np.random.default_rng(seed)

    client_ids = np.arange(1, 10001)
    fs_times = pd.to_datetime(["2025-10-31", "2025-11-30", "2025-12-31", "2026-01-31"])

    # Base risk is time-stable at client level; each month adds calibration drift and noise.
    latent_risk = rng.beta(2.2, 7.0, size=client_ids.size)
    monthly_quality = [0.80, 0.84, 0.88, 0.92]
    monthly_target_sizes = [900, 950, 1000, 1050]
    score_frames: list[pd.DataFrame] = []
    monthly_scores: dict[pd.Timestamp, np.ndarray] = {}
    for fs_time, quality, target_size in zip(fs_times, monthly_quality, monthly_target_sizes):
        month_noise = rng.normal(0.0, 0.07, size=client_ids.size)
        raw_scores = np.clip(quality * latent_risk + (1.0 - quality) * (1.0 - latent_risk) + month_noise, 1e-6, 1.0)
        # Calibrate score mass so expected successes (sum of probabilities) align with target event volume.
        calibration = target_size / float(raw_scores.sum())
        scores = np.clip(raw_scores * calibration, 1e-6, 1.0)
        percentiles = (
            pd.Series(scores)
            .rank(method="average", pct=True)
            .mul(100)
            .apply(np.ceil)
            .clip(1, 100)
            .astype(int)
            .to_numpy()
        )
        score_frames.append(
            pd.DataFrame(
                {
                    "pt_unified_key": client_ids,
                    "fs_time": fs_time,
                    "score": scores,
                    "percentile": percentiles,
                }
            )
        )
        monthly_scores[fs_time] = scores

    model_score = pd.concat(score_frames, ignore_index=True)

    # Create events for the month after each score snapshot.
    # Keep sampling linear in score so realized outcomes align with score-based expected metrics.
    target_frames: list[pd.DataFrame] = []
    for idx, fs_time in enumerate(fs_times):
        scores = monthly_scores[fs_time]
        success_weights = np.clip(scores, 1e-6, 1.0)
        success_prob = success_weights / success_weights.sum()
        target_ids = rng.choice(client_ids, size=monthly_target_sizes[idx], replace=False, p=success_prob)
        month_start = fs_time + pd.DateOffset(days=1)
        next_month_start = month_start + pd.DateOffset(months=1)
        total_seconds = int((next_month_start - month_start).total_seconds())
        random_seconds = rng.integers(0, total_seconds, size=target_ids.size)
        target_frames.append(
            pd.DataFrame(
                {
                    "pt_unified_key": target_ids,
                    "atsp_event_timestamp": month_start + pd.to_timedelta(random_seconds, unit="s"),
                }
            )
        )

    target_store = pd.concat(target_frames, ignore_index=True).sort_values(
        ["atsp_event_timestamp", "pt_unified_key"], ignore_index=True
    )

    latest_scores = monthly_scores[fs_times[-1]]
    campaign_weights = np.power(np.clip(latest_scores, 1e-6, 1.0), 1.4)
    campaign_prob = campaign_weights / campaign_weights.sum()
    campaign_clients_ids = rng.choice(client_ids, size=7000, replace=False, p=campaign_prob)
    campaign_clients = pd.DataFrame(
        {
            "pt_unified_key": campaign_clients_ids,
        }
    ).sort_values("pt_unified_key", ignore_index=True)

    return model_score, target_store, campaign_clients


if __name__ == "__main__":
    model_score_df, target_store_df, campaign_clients_df = create_simulated_tables()

    print("model_score shape:", model_score_df.shape)
    print("target_store shape:", target_store_df.shape)
    print("campaign_clients shape:", campaign_clients_df.shape)
