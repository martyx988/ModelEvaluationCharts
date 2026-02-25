from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulated_data import create_simulated_tables


def _prepare_performance_data(
    model_score: pd.DataFrame,
    target_store: pd.DataFrame,
) -> pd.DataFrame:
    required_model_cols = {"pt_unified_key", "fs_time", "score", "percentile"}
    required_target_cols = {"pt_unified_key", "atsp_event_timestamp"}
    missing_model = required_model_cols - set(model_score.columns)
    missing_target = required_target_cols - set(target_store.columns)

    if missing_model:
        raise ValueError(f"model_score is missing required columns: {sorted(missing_model)}")
    if missing_target:
        raise ValueError(f"target_store is missing required columns: {sorted(missing_target)}")

    scored = model_score.copy()
    scored["fs_time"] = pd.to_datetime(scored["fs_time"], errors="coerce")
    scored["score"] = pd.to_numeric(scored["score"], errors="coerce")

    if scored["fs_time"].isna().any():
        raise ValueError("model_score.fs_time contains invalid datetime values.")
    if scored["score"].isna().any():
        raise ValueError("model_score.score contains non-numeric values.")

    targets = target_store.copy()
    targets["atsp_event_timestamp"] = pd.to_datetime(targets["atsp_event_timestamp"], errors="coerce")
    if targets["atsp_event_timestamp"].isna().any():
        raise ValueError("target_store.atsp_event_timestamp contains invalid datetime values.")

    merged = scored.merge(
        targets[["pt_unified_key", "atsp_event_timestamp"]],
        on="pt_unified_key",
        how="left",
    )

    event_window_end = merged["fs_time"] + pd.DateOffset(months=1)
    merged["is_success"] = (
        merged["atsp_event_timestamp"].ge(merged["fs_time"])
        & merged["atsp_event_timestamp"].le(event_window_end)
    )

    performance = (
        merged.groupby(["pt_unified_key", "fs_time", "score", "percentile"], as_index=False)["is_success"]
        .max()
        .sort_values("score", ascending=False, ignore_index=True)
    )
    return performance


def _build_metrics_by_contact_percentile(performance: pd.DataFrame) -> pd.DataFrame:
    n_clients = len(performance)
    if n_clients == 0:
        raise ValueError("No scored clients available to evaluate.")

    performance = performance.copy()
    performance["cum_successes"] = performance["is_success"].astype(int).cumsum()
    performance["cum_non_successes"] = (~performance["is_success"].astype(bool)).astype(int).cumsum()
    performance["cum_clients"] = np.arange(1, n_clients + 1)

    total_successes = int(performance["is_success"].sum())
    total_non_successes = int(n_clients - total_successes)
    percentile_marks = np.arange(1, 101)
    idx = np.ceil(percentile_marks * n_clients / 100).astype(int) - 1
    points = performance.iloc[idx].copy().reset_index(drop=True)

    points["contacted_percentile"] = percentile_marks
    points["gain_pct"] = np.where(
        total_successes > 0,
        points["cum_successes"] / total_successes * 100.0,
        0.0,
    )
    points["cumulative_success_rate_pct"] = points["cum_successes"] / points["cum_clients"] * 100.0
    points["random_baseline_gain_pct"] = points["contacted_percentile"].astype(float)
    prevalence_pct = total_successes / n_clients * 100.0
    points["ideal_gain_pct"] = np.where(
        prevalence_pct > 0,
        np.minimum(points["contacted_percentile"] / prevalence_pct * 100.0, 100.0),
        0.0,
    )
    points["non_success_share_pct"] = np.where(
        total_non_successes > 0,
        points["cum_non_successes"] / total_non_successes * 100.0,
        0.0,
    )
    points["ks_pct"] = points["gain_pct"] - points["non_success_share_pct"]
    best_ks_percentile = int(points.loc[points["ks_pct"].idxmax(), "contacted_percentile"])
    points["best_ks_percentile"] = best_ks_percentile
    points["total_successes"] = total_successes
    points["total_non_successes"] = total_non_successes

    return points[
        [
            "contacted_percentile",
            "cum_clients",
            "cum_successes",
            "cum_non_successes",
            "gain_pct",
            "random_baseline_gain_pct",
            "ideal_gain_pct",
            "non_success_share_pct",
            "ks_pct",
            "best_ks_percentile",
            "cumulative_success_rate_pct",
            "total_successes",
            "total_non_successes",
        ]
    ]


def _make_figure(metrics: pd.DataFrame, default_percentile: int = 20) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Cumulative Gain Chart",
            "Cumulative Success Rate Chart",
        ),
    )

    x = metrics["contacted_percentile"]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["gain_pct"],
            mode="lines+markers",
            name="Model Gain",
            line={"color": "#0B5FFF", "width": 3},
            marker={"size": 5},
            hovertemplate="Contacted: %{x}%<br>Gain: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["random_baseline_gain_pct"],
            mode="lines",
            name="Random Baseline",
            line={"color": "#6B7280", "width": 2, "dash": "dash"},
            hovertemplate="Contacted: %{x}%<br>Baseline gain: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["ideal_gain_pct"],
            mode="lines",
            name="Ideal Gain",
            line={"color": "#0284C7", "width": 2, "dash": "dot"},
            hovertemplate="Contacted: %{x}%<br>Ideal gain: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["non_success_share_pct"],
            mode="lines",
            name="Cumulative Non-Success Share",
            line={"color": "#EF4444", "width": 2, "dash": "dashdot"},
            hovertemplate="Contacted: %{x}%<br>Non-success share: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["ks_pct"],
            mode="lines",
            name="KS",
            line={"color": "#7C3AED", "width": 2},
            hovertemplate="Contacted: %{x}%<br>KS: %{y:.2f}pp<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["cumulative_success_rate_pct"],
            mode="lines+markers",
            name="Cumulative Success Rate",
            line={"color": "#00A76F", "width": 3},
            marker={"size": 5},
            hovertemplate="Contacted: %{x}%<br>Success rate: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Contacted Population Percentile (%)", row=2, col=1)
    fig.update_yaxes(title_text="Gain / Share (%)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="KS (pp)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Success Rate (%)", row=2, col=1)

    base_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    best_ks_percentile = int(metrics["best_ks_percentile"].iat[0])
    best_ks_row = metrics.loc[metrics["contacted_percentile"] == best_ks_percentile].iloc[0]
    best_ks_value = float(best_ks_row["ks_pct"])

    def slider_layout_for(percentile_value: int) -> dict:
        row = metrics.loc[metrics["contacted_percentile"] == percentile_value].iloc[0]
        gain = float(row["gain_pct"])
        success_rate = float(row["cumulative_success_rate_pct"])
        ks_value = float(row["ks_pct"])
        contacted_clients = int(row["cum_clients"])
        successes = int(row["cum_successes"])
        total_successes = int(row["total_successes"])

        dynamic_annotations = [
            dict(
                x=percentile_value,
                y=gain,
                xref="x",
                yref="y",
                text=f"{percentile_value}% -> Gain {gain:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=30,
                ay=-30,
                bgcolor="rgba(11,95,255,0.12)",
                bordercolor="#0B5FFF",
                borderwidth=1,
                font={"size": 11, "color": "#0F172A"},
            ),
            dict(
                x=percentile_value,
                y=success_rate,
                xref="x2",
                yref="y3",
                text=f"{percentile_value}% -> SR {success_rate:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=30,
                ay=-30,
                bgcolor="rgba(0,167,111,0.12)",
                bordercolor="#00A76F",
                borderwidth=1,
                font={"size": 11, "color": "#0F172A"},
            ),
            dict(
                x=percentile_value,
                y=ks_value,
                xref="x",
                yref="y2",
                text=f"{percentile_value}% -> KS {ks_value:.1f}pp",
                showarrow=True,
                arrowhead=2,
                ax=30,
                ay=30,
                bgcolor="rgba(124,58,237,0.12)",
                bordercolor="#7C3AED",
                borderwidth=1,
                font={"size": 11, "color": "#0F172A"},
            ),
            dict(
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                text=(
                    f"Selected cutoff: top {percentile_value}% clients ({contacted_clients:,}) | "
                    f"Captured successes: {successes:,}/{total_successes:,}"
                ),
                showarrow=False,
                font={"size": 13, "color": "#0F172A"},
                bgcolor="rgba(255,255,255,0.85)",
            ),
            dict(
                x=best_ks_percentile,
                y=best_ks_value,
                xref="x",
                yref="y2",
                text=f"Best KS: {best_ks_value:.1f}pp @ {best_ks_percentile}%",
                showarrow=True,
                arrowhead=2,
                ax=-45,
                ay=-35,
                bgcolor="rgba(124,58,237,0.12)",
                bordercolor="#7C3AED",
                borderwidth=1,
                font={"size": 11, "color": "#0F172A"},
            ),
        ]
        return {
            "shapes": [
                dict(
                    type="line",
                    name="selected_cutoff",
                    xref="x2",
                    yref="paper",
                    x0=percentile_value,
                    x1=percentile_value,
                    y0=0.0,
                    y1=1.0,
                    line={"color": "#E11D48", "width": 2, "dash": "dot"},
                ),
                dict(
                    type="line",
                    name="ks_optimal_split",
                    xref="x",
                    yref="paper",
                    x0=best_ks_percentile,
                    x1=best_ks_percentile,
                    y0=0.5,
                    y1=1.0,
                    line={"color": "#7C3AED", "width": 2, "dash": "dash"},
                )
            ],
            "annotations": base_annotations + dynamic_annotations,
        }

    steps = []
    for percentile_value in range(1, 101):
        steps.append(
            {
                "method": "relayout",
                "label": str(percentile_value),
                "args": [slider_layout_for(percentile_value)],
            }
        )

    fig.update_layout(
        template="plotly_white",
        title={
            "text": "Model Performance Evaluation",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22, "color": "#0F172A"},
        },
        width=1100,
        height=900,
        margin={"l": 70, "r": 40, "t": 140, "b": 70},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 12},
        },
        sliders=[
            {
                "active": max(0, min(99, default_percentile - 1)),
                "currentvalue": {"prefix": "Contact percentile: ", "suffix": "%"},
                "pad": {"t": 20},
                "len": 0.92,
                "x": 0.04,
                "steps": steps,
            }
        ],
    )

    fig.update_layout(**slider_layout_for(default_percentile))
    return fig


def EvaluateModel(
    output_html_path: str | Path = "outputs/model_evaluation_report.html",
    seed: int | None = 42,
) -> Path:
    model_score, target_store, _ = create_simulated_tables(seed=seed)
    performance = _prepare_performance_data(model_score=model_score, target_store=target_store)
    metrics = _build_metrics_by_contact_percentile(performance=performance)
    figure = _make_figure(metrics=metrics, default_percentile=20)

    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_html = figure.to_html(full_html=False, include_plotlyjs=True)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Model Evaluation Report</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
      color: #0F172A;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      background: #FFFFFF;
      border: 1px solid #E2E8F0;
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
      padding: 20px 20px 8px 20px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 28px;
      font-weight: 700;
      color: #0F172A;
      letter-spacing: 0.2px;
    }}
    p {{
      margin: 0 0 16px 0;
      color: #334155;
      font-size: 15px;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Model Performance Evaluation</h1>
    <p>
      Gain and cumulative success rate are computed from scored clients and observed events within
      one month after scoring timestamp.
    </p>
    {fig_html}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    report = EvaluateModel()
    print(f"Generated report: {report.resolve()}")
