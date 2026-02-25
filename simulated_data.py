from __future__ import annotations

import numpy as np
import pandas as pd


def create_simulated_tables(seed: int | None = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create simulated model data tables for local development.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        model_score, target_store, situation
    """
    rng = np.random.default_rng(seed)

    client_ids = np.arange(1, 10001)

    scores = rng.random(client_ids.size)
    # Convert score ranks to integer percentiles in inclusive range 1..100.
    percentiles = (
        pd.Series(scores)
        .rank(method="average", pct=True)
        .mul(100)
        .apply(np.ceil)
        .clip(1, 100)
        .astype(int)
        .to_numpy()
    )

    model_score = pd.DataFrame(
        {
            "pt_unified_key": client_ids,
            "fs_time": "2026-01-31",
            "score": scores,
            "percentile": percentiles,
        }
    )

    target_ids = rng.choice(client_ids, size=1000, replace=False)
    feb_start = pd.Timestamp("2026-02-01")
    mar_start = pd.Timestamp("2026-03-01")
    total_seconds = int((mar_start - feb_start).total_seconds())
    random_seconds = rng.integers(0, total_seconds, size=target_ids.size)

    target_store = pd.DataFrame(
        {
            "pt_unified_key": target_ids,
            "atsp_event_timestamp": feb_start + pd.to_timedelta(random_seconds, unit="s"),
        }
    ).sort_values("pt_unified_key", ignore_index=True)

    situation_ids = rng.choice(client_ids, size=7000, replace=False)
    situation = pd.DataFrame(
        {
            "pt_unified_key": situation_ids,
        }
    ).sort_values("pt_unified_key", ignore_index=True)

    return model_score, target_store, situation


if __name__ == "__main__":
    model_score_df, target_store_df, situation_df = create_simulated_tables()

    print("model_score shape:", model_score_df.shape)
    print("target_store shape:", target_store_df.shape)
    print("situation shape:", situation_df.shape)
