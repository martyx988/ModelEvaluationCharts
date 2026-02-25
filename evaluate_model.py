from __future__ import annotations

import json
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


def _format_period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start == end:
        return start.strftime("%b %Y")
    return f"{start.strftime('%b %Y')} - {end.strftime('%b %Y')}"


def _resolve_historical_scores(
    model_score: pd.DataFrame,
    latest_fs_time: pd.Timestamp,
    historical_period_start: str | pd.Timestamp | None,
    historical_period_end: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, str]:
    scored = model_score.copy()
    scored["fs_time"] = pd.to_datetime(scored["fs_time"], errors="coerce")
    historical = scored.loc[scored["fs_time"] < latest_fs_time].copy()
    if historical.empty:
        raise ValueError("No historical scores are available before the newest actual model score date.")

    start = pd.to_datetime(historical_period_start) if historical_period_start is not None else historical["fs_time"].min()
    end = pd.to_datetime(historical_period_end) if historical_period_end is not None else historical["fs_time"].max()
    if start > end:
        raise ValueError("historical_period_start must be earlier than or equal to historical_period_end.")

    historical = historical.loc[(historical["fs_time"] >= start) & (historical["fs_time"] <= end)].copy()
    if historical.empty:
        raise ValueError("Historical period filter returned no model scores. Adjust historical period boundaries.")

    actual_start = historical["fs_time"].min()
    actual_end = historical["fs_time"].max()
    return historical, _format_period_label(actual_start, actual_end)


def _required_cutoff_for_desired_rate(metrics: pd.DataFrame, desired_rate: float) -> int:
    eligible = metrics.loc[metrics["cumulative_success_rate_pct"] >= desired_rate, "contacted_percentile"]
    if eligible.empty:
        return 1
    return int(eligible.max())


def _success_bar_colors(metrics: pd.DataFrame, required_cutoff: int) -> list[str]:
    percentiles = metrics["contacted_percentile"].astype(int)
    colors: list[str] = []
    for percentile in percentiles:
        if percentile <= required_cutoff:
            colors.append("rgba(0, 87, 217, 0.80)")
        else:
            colors.append("rgba(148, 163, 184, 0.28)")
    return colors


def _make_figure(
    metrics: pd.DataFrame,
    default_percentile: int = 20,
    desired_success_rate: float | None = None,
) -> go.Figure:
    if desired_success_rate is None:
        desired_success_rate = float(
            metrics.loc[metrics["contacted_percentile"] == default_percentile, "cumulative_success_rate_pct"].iloc[0]
        )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=("Cumulative Gain Chart", "Cumulative Success Rate Chart"),
    )

    x = metrics["contacted_percentile"]
    selected_row = metrics.loc[metrics["contacted_percentile"] == default_percentile].iloc[0]
    selected_gain = float(selected_row["gain_pct"])
    selected_sr = float(selected_row["cumulative_success_rate_pct"])
    required_cutoff = _required_cutoff_for_desired_rate(metrics=metrics, desired_rate=desired_success_rate)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["gain_pct"],
            mode="lines+markers",
            name="Model",
            line={"color": "#0057D9", "width": 3},
            marker={"size": 4},
            hovertemplate="Contacted: %{x}%<br>Gain: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["random_baseline_gain_pct"],
            mode="lines",
            name="Random",
            line={"color": "#94A3B8", "width": 1.5, "dash": "dash"},
            hovertemplate="Contacted: %{x}%<br>Baseline gain: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["ideal_gain_pct"],
            mode="lines",
            name="Ideal",
            line={"color": "#CBD5E1", "width": 1.5, "dash": "dot"},
            hovertemplate="Contacted: %{x}%<br>Ideal gain: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=metrics["cumulative_success_rate_pct"],
            name="Success Rate",
            marker={
                "color": _success_bar_colors(metrics=metrics, required_cutoff=required_cutoff),
                "line": {"color": "rgba(15, 23, 42, 0.12)", "width": 0.6},
            },
            hovertemplate="Contacted: %{x}%<br>Success rate: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text="Contacted Population Percentile (%)",
        row=2,
        col=1,
        range=[1, 100],
        showline=True,
        linewidth=1,
        linecolor="#94A3B8",
        ticks="outside",
        tickcolor="#94A3B8",
        dtick=10,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(
        title_text="Gain / Share (%)",
        row=1,
        col=1,
        range=[0, 105],
        showline=True,
        linewidth=1,
        linecolor="#94A3B8",
        gridcolor="#E2E8F0",
    )
    fig.update_yaxes(zeroline=False)
    fig.update_yaxes(
        title_text="Success Rate (%)",
        row=2,
        col=1,
        range=[0, max(5.0, float(metrics["cumulative_success_rate_pct"].max()) * 1.08)],
        showline=True,
        linewidth=1,
        linecolor="#94A3B8",
        gridcolor="#E2E8F0",
    )

    base_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    best_ks_percentile = int(metrics["best_ks_percentile"].iat[0])
    best_ks_row = metrics.loc[metrics["contacted_percentile"] == best_ks_percentile].iloc[0]
    best_ks_value = float(best_ks_row["ks_pct"])

    annotations = base_annotations + [
        dict(
            x=default_percentile,
            y=selected_gain,
            xref="x",
            yref="y",
            text=f"Top {default_percentile}% -> SR {selected_sr:.1f}%, Gain {selected_gain:.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=26,
            ay=-34,
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="#E11D48",
            borderwidth=1,
            borderpad=4,
            font={"size": 11, "color": "#0F172A"},
        ),
        dict(
            x=best_ks_percentile,
            y=101,
            xref="x",
            yref="y",
            text=f"Optimal cutoff (max KS {best_ks_value:.1f}pp): {best_ks_percentile}%",
            showarrow=False,
            bgcolor="rgba(248,250,252,0.95)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            borderpad=4,
            font={"size": 11, "color": "#334155"},
        ),
        dict(
            x=required_cutoff,
            y=max(5.0, float(metrics["cumulative_success_rate_pct"].max()) * 0.86),
            xref="x2",
            yref="y2",
            text=f"Needed for target SR: {required_cutoff}%",
            showarrow=False,
            bgcolor="rgba(255,251,235,0.96)",
            bordercolor="#D97706",
            borderwidth=1,
            borderpad=4,
            font={"size": 11, "color": "#92400E"},
        ),
    ]
    shapes = [
        dict(
            type="line",
            name="selected_cutoff",
            xref="x",
            yref="paper",
            x0=default_percentile,
            x1=default_percentile,
            y0=0.0,
            y1=1.0,
            line={"color": "#E11D48", "width": 2.2, "dash": "dot"},
        ),
        dict(
            type="line",
            name="ks_optimal_split",
            xref="x",
            yref="paper",
            x0=best_ks_percentile,
            x1=best_ks_percentile,
            y0=0.0,
            y1=1.0,
            line={"color": "#94A3B8", "width": 1.4, "dash": "dash"},
        ),
        dict(
            type="line",
            name="required_cutoff",
            xref="x",
            yref="paper",
            x0=required_cutoff,
            x1=required_cutoff,
            y0=0.0,
            y1=1.0,
            line={"color": "#D97706", "width": 1.8, "dash": "dot"},
        ),
    ]

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title={
            "text": "Model Performance",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 28, "color": "#0F172A", "family": "Segoe UI, Arial, sans-serif"},
        },
        width=680,
        height=940,
        margin={"l": 92, "r": 55, "t": 120, "b": 90},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.06,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 12, "color": "#334155"},
            "bgcolor": "rgba(255,255,255,0)",
            "tracegroupgap": 12,
        },
        annotations=annotations,
        shapes=shapes,
        font={"family": "Segoe UI, Arial, sans-serif", "size": 13, "color": "#0F172A"},
    )
    return fig


def _build_cutoff_summary(metrics: pd.DataFrame, selected_percentile: int) -> dict[str, float | int]:
    row = metrics.loc[metrics["contacted_percentile"] == selected_percentile].iloc[0]
    captured_successes = int(row["cum_successes"])
    total_successes = int(row["total_successes"])
    captured_pct = (captured_successes / total_successes * 100.0) if total_successes > 0 else 0.0
    return {
        "selected_percentile": selected_percentile,
        "contacted_clients": int(row["cum_clients"]),
        "gain_pct": float(row["gain_pct"]),
        "success_rate_pct": float(row["cumulative_success_rate_pct"]),
        "captured_successes": captured_successes,
        "total_successes": total_successes,
        "captured_pct": captured_pct,
    }


def _build_campaign_selection_summary(
    performance: pd.DataFrame,
    metrics: pd.DataFrame,
    campaign_clients: pd.DataFrame,
) -> dict[str, float | int]:
    if "pt_unified_key" not in campaign_clients.columns:
        raise ValueError("campaign_clients must include column: pt_unified_key")
    selected_keys = set(campaign_clients["pt_unified_key"].dropna().astype(str))
    if not selected_keys:
        raise ValueError("campaign_clients.pt_unified_key is empty after removing null values.")

    performance_keys = performance["pt_unified_key"].astype(str)
    selected = performance.loc[performance_keys.isin(selected_keys)].copy()
    if selected.empty:
        raise ValueError("None of campaign_clients.pt_unified_key matched scored clients.")

    n_total = len(performance)
    n_selected = len(selected)
    total_successes = int(performance["is_success"].sum())
    selected_successes = int(selected["is_success"].sum())
    selected_rate = (selected_successes / n_selected * 100.0) if n_selected > 0 else 0.0
    selected_capture = (selected_successes / total_successes * 100.0) if total_successes > 0 else 0.0

    model_top_n = performance.head(n_selected)
    model_top_successes = int(model_top_n["is_success"].sum())
    model_top_rate = (model_top_successes / n_selected * 100.0) if n_selected > 0 else 0.0
    model_top_capture = (model_top_successes / total_successes * 100.0) if total_successes > 0 else 0.0

    overall_rate = (total_successes / n_total * 100.0) if n_total > 0 else 0.0
    volume_pct = n_selected / n_total * 100.0
    volume_percentile = int(min(100, max(1, np.ceil(volume_pct))))

    return {
        "selected_clients": n_selected,
        "selected_rate_pct": selected_rate,
        "selected_capture_pct": selected_capture,
        "model_top_rate_pct": model_top_rate,
        "model_top_capture_pct": model_top_capture,
        "overall_rate_pct": overall_rate,
        "volume_percentile": volume_percentile,
        "model_curve_rate_pct": float(
            metrics.loc[metrics["contacted_percentile"] == volume_percentile, "cumulative_success_rate_pct"].iloc[0]
        ),
    }


def _build_cumulative_estimated_curve(scored_clients: pd.DataFrame) -> pd.DataFrame:
    if scored_clients.empty:
        raise ValueError("Cannot build cumulative estimated curve from empty scored client frame.")
    ordered = scored_clients.sort_values("score", ascending=False, ignore_index=True).copy()
    ordered["cum_mean_score"] = ordered["score"].expanding().mean() * 100.0
    n = len(ordered)
    percentile_marks = np.arange(1, 101)
    idx = np.ceil(percentile_marks * n / 100).astype(int) - 1
    points = ordered.iloc[idx].copy().reset_index(drop=True)
    return pd.DataFrame(
        {
            "contact_share_pct": percentile_marks,
            "estimated_success_rate_pct": points["cum_mean_score"].astype(float),
        }
    )


def _build_actual_estimated_selection_summary(
    actual_model_score: pd.DataFrame,
    campaign_clients: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    if "pt_unified_key" not in campaign_clients.columns:
        raise ValueError("campaign_clients must include column: pt_unified_key")
    selected_keys = set(campaign_clients["pt_unified_key"].dropna().astype(str))
    if not selected_keys:
        raise ValueError("campaign_clients.pt_unified_key is empty after removing null values.")

    scored = actual_model_score.copy()
    scored["pt_unified_key"] = scored["pt_unified_key"].astype(str)
    selected = scored.loc[scored["pt_unified_key"].isin(selected_keys)].copy()
    if selected.empty:
        raise ValueError("None of campaign_clients.pt_unified_key matched scored clients in actual period.")

    n_selected = len(selected)
    model_top = scored.sort_values("score", ascending=False, ignore_index=True).head(n_selected).copy()
    selected_rate_estimated = float(selected["score"].mean() * 100.0)
    model_rate_estimated = float(model_top["score"].mean() * 100.0)
    volume_pct = n_selected / len(scored) * 100.0
    volume_percentile = int(min(100, max(1, np.ceil(volume_pct))))
    selected_curve = _build_cumulative_estimated_curve(selected)
    model_curve = _build_cumulative_estimated_curve(model_top)
    return (
        {
            "selected_clients": n_selected,
            "selected_rate_estimated_pct": selected_rate_estimated,
            "model_guided_rate_estimated_pct": model_rate_estimated,
            "volume_percentile": volume_percentile,
        },
        selected_curve,
        model_curve,
    )


def _historical_portfolio_event_rate(
    historical_scores: pd.DataFrame,
    target_store: pd.DataFrame,
) -> float:
    rates: list[float] = []
    for _, snapshot in historical_scores.groupby("fs_time", sort=True):
        performance = _prepare_performance_data(model_score=snapshot, target_store=target_store)
        rates.append(float(performance["is_success"].mean() * 100.0))
    if not rates:
        raise ValueError("Could not compute historical portfolio event rate from selected historical period.")
    return float(np.mean(rates))


def _make_campaign_selection_figure(
    actual_period_label: str,
    selection_summary: dict[str, float | int],
    selected_curve: pd.DataFrame,
    model_guided_curve: pd.DataFrame,
    historical_portfolio_rate_pct: float,
    historical_period_label: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.18,
        subplot_titles=(
            "Estimated Success Rate Benchmark",
            "Cumulative Estimated Success Rate Within Campaign Volume",
        ),
    )

    benchmark_labels = ["Your Selection", "Model-guided at same volume", "Portfolio average"]
    benchmark_values = [
        float(selection_summary["selected_rate_estimated_pct"]),
        float(selection_summary["model_guided_rate_estimated_pct"]),
        float(historical_portfolio_rate_pct),
    ]
    fig.add_trace(
        go.Bar(
            x=benchmark_labels,
            y=benchmark_values,
            marker={"color": ["#2563EB", "#0EA5E9", "#94A3B8"]},
            text=[f"{v:.1f}%" for v in benchmark_values],
            textposition="outside",
            hovertemplate="%{x}<br>Success rate: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=selected_curve["contact_share_pct"],
            y=selected_curve["estimated_success_rate_pct"],
            mode="lines",
            line={"color": "#2563EB", "width": 2.5},
            name=f"Selected clients (estimated, {actual_period_label})",
            hovertemplate="Within selected set: %{x}%<br>Estimated SR: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=model_guided_curve["contact_share_pct"],
            y=model_guided_curve["estimated_success_rate_pct"],
            mode="lines",
            line={"color": "#0EA5E9", "width": 2.0, "dash": "dash"},
            name=f"Model-guided same volume (estimated, {actual_period_label})",
            hovertemplate="Within model-guided set: %{x}%<br>Estimated SR: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    selected_full = float(selection_summary["selected_rate_estimated_pct"])
    model_full = float(selection_summary["model_guided_rate_estimated_pct"])
    fig.update_layout(
        annotations=[
            dict(
                x=100,
                y=selected_full,
                xref="x2",
                yref="y2",
                text=f"Selected full-volume estimate: {selected_full:.1f}%",
                showarrow=False,
                bgcolor="rgba(37,99,235,0.10)",
                bordercolor="#2563EB",
                borderwidth=1,
                borderpad=3,
                font={"size": 10, "color": "#1E3A8A"},
            ),
            dict(
                x=100,
                y=model_full,
                xref="x2",
                yref="y2",
                text=f"Model-guided full-volume estimate: {model_full:.1f}%",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(14,165,233,0.12)",
                bordercolor="#0EA5E9",
                borderwidth=1,
                borderpad=3,
                font={"size": 10, "color": "#0C4A6E"},
            ),
        ],
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Segoe UI, Arial, sans-serif", "size": 12, "color": "#0F172A"},
        width=680,
        height=620,
        margin={"l": 72, "r": 30, "t": 80, "b": 60},
    )
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1, rangemode="tozero", gridcolor="#E2E8F0")
    fig.update_yaxes(title_text="Estimated Success Rate (%)", row=2, col=1, rangemode="tozero", gridcolor="#E2E8F0")
    fig.update_xaxes(title_text="Contacted Share Within Campaign Volume (%)", row=2, col=1, dtick=10)
    return fig


def EvaluateModel(
    output_html_path: str | Path = "outputs/model_evaluation_report.html",
    seed: int | None = 42,
    include_campaign_selection: bool = False,
    campaign_clients: pd.DataFrame | None = None,
    historical_period_start: str | pd.Timestamp | None = None,
    historical_period_end: str | pd.Timestamp | None = None,
) -> Path:
    model_score, target_store, _ = create_simulated_tables(seed=seed)
    model_score = model_score.copy()
    model_score["fs_time"] = pd.to_datetime(model_score["fs_time"], errors="coerce")
    if model_score["fs_time"].isna().any():
        raise ValueError("model_score.fs_time contains invalid datetime values.")
    latest_fs_time = model_score["fs_time"].max()
    latest_model_score = model_score.loc[model_score["fs_time"] == latest_fs_time].copy()
    performance = _prepare_performance_data(model_score=latest_model_score, target_store=target_store)
    metrics = _build_metrics_by_contact_percentile(performance=performance)
    selected_percentile = int(metrics["best_ks_percentile"].iat[0])
    desired_success_rate = float(
        metrics.loc[metrics["contacted_percentile"] == selected_percentile, "cumulative_success_rate_pct"].iloc[0]
    )
    required_cutoff = _required_cutoff_for_desired_rate(metrics=metrics, desired_rate=desired_success_rate)
    figure = _make_figure(
        metrics=metrics,
        default_percentile=selected_percentile,
        desired_success_rate=desired_success_rate,
    )
    summary = _build_cutoff_summary(metrics=metrics, selected_percentile=selected_percentile)
    cutoff_points = [
        {
            "p": int(row["contacted_percentile"]),
            "clients": int(row["cum_clients"]),
            "gain": float(row["gain_pct"]),
            "sr": float(row["cumulative_success_rate_pct"]),
            "captured": int(row["cum_successes"]),
            "total": int(row["total_successes"]),
            "captured_pct": (
                float(row["cum_successes"]) / float(row["total_successes"]) * 100.0
                if float(row["total_successes"]) > 0
                else 0.0
            ),
        }
        for _, row in metrics.iterrows()
    ]
    best_ks_percentile = int(metrics["best_ks_percentile"].iat[0])
    campaign_section_html = ""
    if include_campaign_selection:
        if campaign_clients is None:
            raise ValueError("campaign_clients must be provided when include_campaign_selection=True.")
        selection_summary, selected_curve, model_guided_curve = _build_actual_estimated_selection_summary(
            actual_model_score=latest_model_score,
            campaign_clients=campaign_clients,
        )
        historical_scores, historical_period_label = _resolve_historical_scores(
            model_score=model_score,
            latest_fs_time=latest_fs_time,
            historical_period_start=historical_period_start,
            historical_period_end=historical_period_end,
        )
        historical_portfolio_rate_pct = _historical_portfolio_event_rate(
            historical_scores=historical_scores,
            target_store=target_store,
        )
        actual_period_label = _format_period_label(latest_fs_time, latest_fs_time)
        campaign_fig = _make_campaign_selection_figure(
            actual_period_label=actual_period_label,
            selection_summary=selection_summary,
            selected_curve=selected_curve,
            model_guided_curve=model_guided_curve,
            historical_portfolio_rate_pct=historical_portfolio_rate_pct,
            historical_period_label=historical_period_label,
        )
        campaign_fig_html = campaign_fig.to_html(full_html=False, include_plotlyjs=False, div_id="campaign-figure")
        campaign_section_html = f"""
    <section class="campaign-section">
      <h2>Campaign Selection Potential</h2>
      <p>
        Estimated from actual model scores (newest snapshot: {actual_period_label}).
        Targets are not used for the first two metrics.
      </p>
      <p><strong>Portfolio baseline period:</strong> {historical_period_label} (historical observed events)</p>
      <div class="campaign-kpis">
        <div class="campaign-kpi"><strong>Your selection (estimated)</strong><span>{selection_summary["selected_rate_estimated_pct"]:.1f}%</span></div>
        <div class="campaign-kpi"><strong>Model-guided at same volume (estimated)</strong><span>{selection_summary["model_guided_rate_estimated_pct"]:.1f}%</span></div>
        <div class="campaign-kpi"><strong>Portfolio average (historical)</strong><span>{historical_portfolio_rate_pct:.1f}%</span></div>
      </div>
      <ul class="campaign-tips">
        <li>Calibration chart (predicted vs observed rate by score band).</li>
        <li>Lift chart at campaign-volume breakpoints (10%, 20%, ...).</li>
        <li>Expected successes by budget/capacity scenarios.</li>
      </ul>
      {campaign_fig_html}
    </section>
"""

    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_html = figure.to_html(full_html=False, include_plotlyjs=True, div_id="model-figure")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Model Evaluation Report</title>
  <style>
    :root {{
      --bg-top: #F8FAFC;
      --bg-bottom: #EEF2FF;
      --panel: #FFFFFF;
      --text-main: #0F172A;
      --text-muted: #475569;
      --line-soft: #E2E8F0;
      --accent: #0057D9;
    }}
    body {{
      margin: 0;
      padding: 32px 24px;
      background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
      color: var(--text-main);
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    .wrap {{
      max-width: 1360px;
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--line-soft);
      border-radius: 14px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
      padding: 26px 24px 14px 24px;
    }}
    .content-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(360px, 1fr);
      gap: 16px;
      align-items: stretch;
    }}
    .figure-panel {{
      min-width: 0;
    }}
    .guide-panel {{
      border: 1px solid var(--line-soft);
      border-radius: 12px;
      background: #FAFCFF;
      padding: 14px 14px 12px 14px;
      display: grid;
      grid-template-rows: 1fr 1fr;
      gap: 12px;
      min-height: 940px;
    }}
    .guide-section {{
      border: 1px solid #E6EEF8;
      border-radius: 10px;
      background: #FFFFFF;
      padding: 12px 11px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}
    .guide-panel h3 {{
      margin: 0 0 8px 0;
      font-size: 16px;
      color: var(--text-main);
    }}
    .guide-panel p {{
      margin: 0 0 10px 0;
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.45;
    }}
    .guide-section strong {{
      color: var(--text-main);
      margin-top: 4px;
      margin-bottom: 2px;
      font-size: 13px;
      display: block;
    }}
    .header {{
      border-bottom: 1px solid var(--line-soft);
      padding-bottom: 12px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 30px;
      font-weight: 700;
      color: var(--text-main);
      letter-spacing: 0.2px;
    }}
    p {{
      margin: 0;
      color: var(--text-muted);
      font-size: 15px;
      line-height: 1.5;
    }}
    .meta {{
      margin-top: 8px;
      font-size: 13px;
      color: var(--text-muted);
    }}
    .subtitle {{
      margin-top: 4px;
      font-size: 14px;
      color: var(--text-muted);
    }}
    .kpis {{
      margin: 16px 0 8px 0;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }}
    .kpi {{
      border: 1px solid var(--line-soft);
      border-radius: 10px;
      padding: 10px 12px;
      background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    }}
    .kpi-label {{
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .kpi-value {{
      font-size: 20px;
      font-weight: 700;
      color: var(--text-main);
      line-height: 1.2;
    }}
    .footnote {{
      margin-top: 10px;
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.5;
      border-top: 1px solid var(--line-soft);
      padding-top: 10px;
    }}
    .controls {{
      margin: 10px 0 12px 0;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .controls label {{
      font-size: 13px;
      color: var(--text-muted);
      font-weight: 600;
    }}
    #cutoff-slider {{
      width: min(460px, 90vw);
      accent-color: var(--accent);
    }}
    #cutoff-value {{
      font-size: 13px;
      color: var(--text-main);
      font-weight: 700;
      background: #EFF6FF;
      border: 1px solid #BFDBFE;
      border-radius: 999px;
      padding: 2px 9px;
    }}
    #desired-rate-slider {{
      width: min(300px, 80vw);
      accent-color: #D97706;
    }}
    #desired-rate-value, #required-cutoff-value {{
      font-size: 13px;
      color: #7C2D12;
      font-weight: 700;
      background: #FFF7ED;
      border: 1px solid #FDBA74;
      border-radius: 999px;
      padding: 2px 9px;
    }}
    .accent {{
      color: var(--accent);
      font-weight: 600;
    }}
    .campaign-section {{
      margin-top: 20px;
      border-top: 1px solid var(--line-soft);
      padding-top: 14px;
    }}
    .campaign-section h2 {{
      margin: 0 0 6px 0;
      font-size: 22px;
      color: var(--text-main);
    }}
    .campaign-kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin: 8px 0 8px 0;
    }}
    .campaign-kpi {{
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 8px 10px;
      background: #FCFEFF;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      color: var(--text-muted);
    }}
    .campaign-kpi strong {{
      color: var(--text-main);
      font-weight: 600;
    }}
    .campaign-kpi span {{
      color: #1D4ED8;
      font-weight: 700;
    }}
    .campaign-tips {{
      margin: 8px 0 10px 18px;
      color: var(--text-muted);
      font-size: 13px;
      line-height: 1.4;
      padding: 0;
    }}
    @media (max-width: 1100px) {{
      .content-grid {{
        grid-template-columns: 1fr;
      }}
      .guide-panel {{
        min-height: auto;
        grid-template-rows: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>Model Performance Evaluation</h1>
      <p>
        Cumulative gain and cumulative success rate
      </p>
      <div class="subtitle">Presentation view with selected and optimal cutoff markers.</div>
      <div class="meta">
        The dashed vertical marker indicates the <span class="accent">optimal cutoff by maximum KS separation</span>.
      </div>
    </div>
    <div class="kpis">
      <div class="kpi">
        <div class="kpi-label">Selected cutoff</div>
        <div class="kpi-value" id="kpi-cutoff">Top {summary["selected_percentile"]}% ({summary["contacted_clients"]:,})</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Lift / Gain</div>
        <div class="kpi-value" id="kpi-gain">{summary["gain_pct"]:.1f}%</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Success rate @ cutoff</div>
        <div class="kpi-value" id="kpi-sr">{summary["success_rate_pct"]:.1f}%</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Captured successes</div>
        <div class="kpi-value" id="kpi-captured">{summary["captured_successes"]:,} / {summary["total_successes"]:,} ({summary["captured_pct"]:.1f}%)</div>
      </div>
    </div>
    <div class="controls">
      <label for="cutoff-slider">Selected cutoff percentile</label>
      <input id="cutoff-slider" type="range" min="1" max="100" step="1" value="{summary["selected_percentile"]}" />
      <span id="cutoff-value">{summary["selected_percentile"]}%</span>
      <label for="desired-rate-slider">Desired success rate</label>
      <input
        id="desired-rate-slider"
        type="range"
        min="{metrics['cumulative_success_rate_pct'].min():.1f}"
        max="{metrics['cumulative_success_rate_pct'].max():.1f}"
        step="0.1"
        value="{desired_success_rate:.1f}"
      />
      <span id="desired-rate-value">{desired_success_rate:.1f}%</span>
      <span id="required-cutoff-value">Required cutoff: {required_cutoff}%</span>
    </div>
    <div class="content-grid">
      <div class="figure-panel">
        {fig_html}
        <div class="footnote">
          Definitions: Gain = cumulative share of all successes captured up to the selected percentile.
          Success rate = cumulative successes divided by cumulative contacted clients.
          Success is defined as an observed event from scoring time through one calendar month.
        </div>
      </div>
      <aside class="guide-panel">
        <section class="guide-section">
          <h3>How to read these charts: top chart</h3>
          <p>
            The gain chart answers: <span class="accent">"How much of all potential successes will we capture?"</span>
            at each targeting depth.
          </p>
          <strong>Model vs baselines</strong>
          <p>
            Blue above gray means model ranking adds value over random outreach. Earlier separation means
            better prioritization.
          </p>
          <strong>Business decision</strong>
          <p>
            Use the selected cutoff line to balance campaign size vs captured success share.
          </p>
        </section>
        <section class="guide-section">
          <h3>How to read the bottom chart</h3>
          <p>
            Bars show cumulative success rate by cutoff depth. This indicates expected conversion quality
            in the contacted population.
          </p>
          <strong>Desired success rate control</strong>
          <p>
            Set a target rate and use the reported required cutoff to choose the largest audience
            that still meets your quality target.
          </p>
          <strong>Bar colors</strong>
          <p>
            Blue bars are inside the chosen percentile range for your desired success rate.
            Gray-shadow bars are outside the chosen range.
          </p>
        </section>
      </aside>
    </div>
    {campaign_section_html}
  </div>
  <script>
    const cutoffData = {json.dumps(cutoff_points)};
    const bestKsPercentile = {best_ks_percentile};

    function formatInt(x) {{
      return Number(x).toLocaleString("en-US");
    }}

    function pointFor(p) {{
      return cutoffData[p - 1];
    }}

    function updateKpis(point) {{
      document.getElementById("kpi-cutoff").textContent = `Top ${{point.p}}% (${{formatInt(point.clients)}})`;
      document.getElementById("kpi-gain").textContent = `${{point.gain.toFixed(1)}}%`;
      document.getElementById("kpi-sr").textContent = `${{point.sr.toFixed(1)}}%`;
      document.getElementById("kpi-captured").textContent =
        `${{formatInt(point.captured)}} / ${{formatInt(point.total)}} (${{point.captured_pct.toFixed(1)}}%)`;
      document.getElementById("cutoff-value").textContent = `${{point.p}}%`;
    }}

    function requiredCutoffForDesired(desiredRate) {{
      let required = 1;
      for (const point of cutoffData) {{
        if (point.sr >= desiredRate) {{
          required = point.p;
        }}
      }}
      return required;
    }}

    function barColorForRequired(point, required) {{
      if (point.p <= required) {{
        return "rgba(0,87,217,0.80)";
      }}
      return "rgba(148,163,184,0.28)";
    }}

    function updateDesiredRateUi(desiredRate, forcedRequired = null) {{
      const required = forcedRequired === null ? requiredCutoffForDesired(desiredRate) : forcedRequired;
      document.getElementById("desired-rate-value").textContent = `${{desiredRate.toFixed(1)}}%`;
      document.getElementById("required-cutoff-value").textContent = `Required cutoff: ${{required}}%`;

      const gd = document.getElementById("model-figure");
      if (!gd || !gd.layout || !gd.layout.annotations || gd.layout.annotations.length < 5) {{
        return false;
      }}
      const colors = cutoffData.map((point) => barColorForRequired(point, required));
      Plotly.restyle(gd, {{
        "marker.color": [colors]
      }}, [3]);
      Plotly.relayout(gd, {{
        "shapes[2].x0": required,
        "shapes[2].x1": required,
        "annotations[4].x": required,
        "annotations[4].text": `Needed for target SR: ${{required}}%`
      }});
      return true;
    }}

    function updateFigure(point) {{
      const gd = document.getElementById("model-figure");
      if (!gd || !gd.layout || !gd.layout.annotations || gd.layout.annotations.length < 5) {{
        return false;
      }}
      Plotly.relayout(gd, {{
        "shapes[0].x0": point.p,
        "shapes[0].x1": point.p,
        "annotations[2].x": point.p,
        "annotations[2].y": point.gain,
        "annotations[2].text": `Top ${{point.p}}% -> SR ${{point.sr.toFixed(1)}}%, Gain ${{point.gain.toFixed(1)}}%`,
        "annotations[3].x": bestKsPercentile
      }});
      return true;
    }}

    function applyCutoff(percentile) {{
      const point = pointFor(percentile);
      if (!point) {{
        return;
      }}
      const required = point.p;
      desiredRateSlider.value = point.sr.toFixed(1);
      updateKpis(point);
      updateFigure(point);
      updateDesiredRateUi(Number(desiredRateSlider.value), required);
    }}

    function applyDesiredRate(desiredRate) {{
      const required = requiredCutoffForDesired(desiredRate);
      slider.value = String(required);
      updateDesiredRateUi(desiredRate);
      applyCutoff(required);
    }}

    const slider = document.getElementById("cutoff-slider");
    const desiredRateSlider = document.getElementById("desired-rate-slider");
    slider.addEventListener("input", (event) => {{
      applyCutoff(Number(event.target.value));
    }});
    desiredRateSlider.addEventListener("input", (event) => {{
      applyDesiredRate(Number(event.target.value));
    }});

    let retries = 0;
    const init = () => {{
      const ok = updateFigure(pointFor(Number(slider.value))) && updateDesiredRateUi(Number(desiredRateSlider.value));
      applyCutoff(Number(slider.value));
      if (!ok && retries < 20) {{
        retries += 1;
        setTimeout(init, 100);
      }}
    }};
    init();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    report = EvaluateModel()
    print(f"Generated report: {report.resolve()}")
