from __future__ import annotations

import pandas as pd

from evaluate_model import _build_metrics_by_contact_percentile, _make_figure


def _performance_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pt_unified_key": [f"k{i}" for i in range(1, 11)],
            "fs_time": pd.to_datetime(["2026-01-01"] * 10),
            "score": [0.99, 0.95, 0.93, 0.9, 0.84, 0.75, 0.63, 0.51, 0.42, 0.31],
            "percentile": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            "is_success": [True, True, True, False, True, False, False, False, False, False],
        }
    )


def test_metrics_include_ks_and_non_success_columns() -> None:
    metrics = _build_metrics_by_contact_percentile(_performance_fixture())

    expected = {
        "cum_non_successes",
        "non_success_share_pct",
        "ks_pct",
        "best_ks_percentile",
    }
    assert expected.issubset(set(metrics.columns))
    assert metrics["ks_pct"].between(-100, 100).all()
    assert metrics["best_ks_percentile"].nunique() == 1

    best_row = metrics.loc[metrics["ks_pct"].idxmax()]
    assert int(best_row["best_ks_percentile"]) == int(metrics["best_ks_percentile"].iat[0])


def test_figure_has_additional_gain_and_ks_traces() -> None:
    metrics = _build_metrics_by_contact_percentile(_performance_fixture())
    figure = _make_figure(metrics, default_percentile=20)

    trace_names = {trace.name for trace in figure.data}
    assert "Model Gain" in trace_names
    assert "Random Baseline" in trace_names
    assert "Ideal Gain" in trace_names
    assert "Cumulative Non-Success Share" in trace_names
    assert "KS" not in trace_names

    layout_dict = figure.to_plotly_json().get("layout", {})
    shapes = layout_dict.get("shapes", [])
    assert any(shape.get("name") == "ks_optimal_split" for shape in shapes)
