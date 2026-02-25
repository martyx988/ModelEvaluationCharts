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
    historical_period: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, str]:
    scored = model_score.copy()
    scored["fs_time"] = pd.to_datetime(scored["fs_time"], errors="coerce")
    historical_all = scored.loc[scored["fs_time"] < latest_fs_time].copy()
    if historical_all.empty:
        raise ValueError("No historical scores are available before the newest actual model score date.")

    if historical_period is None:
        selected = historical_all.loc[historical_all["fs_time"] == historical_all["fs_time"].max()].copy()
        ts = selected["fs_time"].iloc[0]
        return selected, _format_period_label(ts, ts)

    selected_period = pd.to_datetime(historical_period).to_period("M")
    historical = historical_all.loc[historical_all["fs_time"].dt.to_period("M") == selected_period].copy()
    if historical.empty:
        available = sorted(historical_all["fs_time"].dt.strftime("%Y-%m-%d").unique())
        raise ValueError(
            "No historical scores found for requested period. "
            f"Requested={selected_period}; available snapshots={available}"
        )
    ts = historical["fs_time"].iloc[0]
    return historical, _format_period_label(ts, ts)


def _resolve_historical_period_input(
    historical_period: str | pd.Timestamp | None,
    historical_period_start: str | pd.Timestamp | None,
    historical_period_end: str | pd.Timestamp | None,
) -> str | pd.Timestamp | None:
    has_legacy_args = historical_period_start is not None or historical_period_end is not None
    if historical_period is not None and has_legacy_args:
        raise ValueError("Use either historical_period or historical_period_start/end, not both.")
    if not has_legacy_args:
        return historical_period

    try:
        start_ts = pd.to_datetime(historical_period_start) if historical_period_start is not None else None
        end_ts = pd.to_datetime(historical_period_end) if historical_period_end is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("historical_period_start/end must be parseable dates.") from exc

    if start_ts is not None and end_ts is not None and start_ts.to_period("M") != end_ts.to_period("M"):
        raise ValueError("historical_period_start and historical_period_end must fall in the same calendar month.")

    resolved = end_ts if end_ts is not None else start_ts
    return resolved


def _resolve_report_language(language: str) -> dict[str, str]:
    translations: dict[str, dict[str, str]] = {
        "en": {
            "html_lang": "en",
            "report_title": "Model Evaluation Report",
            "header_title": "Model Performance Evaluation",
            "header_subtitle": "Cumulative gain and cumulative success rate with synchronized campaign scenario views.",
            "kpi_selected_cutoff": "Selected cutoff",
            "kpi_lift_gain": "Lift / Gain",
            "kpi_success_rate": "Success rate @ cutoff",
            "kpi_captured": "Captured successes",
            "label_cutoff_slider": "Selected cutoff percentile",
            "label_desired_rate_slider": "Desired success rate",
            "required_cutoff_prefix": "Required cutoff",
            "top_word": "Top",
            "abbr_sr": "SR",
            "abbr_gain": "Gain",
            "campaign_section_title": "Campaign Selection Potential",
            "campaign_intro": "Expected successes are estimated from model score sums on the campaign filtered base.",
            "campaign_distribution_title": "Campaign Client Distribution by Score Percentile",
            "campaign_rate_compare_title": "Default Success Rate Comparison",
            "campaign_gain_title": "Campaign Base Cumulative Gain",
            "campaign_success_title": "Campaign Base Cumulative Success Rate",
            "top_gain_title": "Cumulative Gain Chart",
            "top_success_title": "Cumulative Success Rate Chart",
            "tooltip_how_to_read": "How to read this chart",
            "number_locale": "en-US",
        },
        "cs": {
            "html_lang": "cs",
            "report_title": "Vyhodnocení modelu",
            "header_title": "Vyhodnocení výkonnosti modelu",
            "header_subtitle": "Kumulativní zisk a kumulativní úspěšnost se synchronizovanými pohledy scénáře kampaně.",
            "kpi_selected_cutoff": "Výběrový cut-off",
            "kpi_lift_gain": "Lift / Gain",
            "kpi_success_rate": "Úspěšnost @ cut-off",
            "kpi_captured": "Zachycené úspěchy",
            "label_cutoff_slider": "Výběrový percentil cut-off",
            "label_desired_rate_slider": "Požadovaná úspěšnost",
            "required_cutoff_prefix": "Požadovaný cut-off",
            "top_word": "Top",
            "abbr_sr": "ÚSP",
            "abbr_gain": "zisk",
            "campaign_section_title": "Potenciál výběru kampaně",
            "campaign_intro": "Očekávané úspěchy jsou odhadnuty ze součtu skóre v kampaní filtrované bázi.",
            "campaign_distribution_title": "Rozložení klientů kampaně podle percentilu skóre",
            "campaign_rate_compare_title": "Porovnání výchozí úspěšnosti",
            "campaign_gain_title": "Kumulativní zisk kampaně",
            "campaign_success_title": "Kumulativní úspěšnost kampaně",
            "top_gain_title": "Graf kumulativního zisku",
            "top_success_title": "Graf kumulativní úspěšnosti",
            "tooltip_how_to_read": "Jak číst tento graf",
            "number_locale": "cs-CZ",
        },
    }
    if language not in translations:
        raise ValueError("language must be one of: en, cs")
    return translations[language]


def _resolve_chart_text(language: str) -> dict[str, str]:
    texts: dict[str, dict[str, str]] = {
        "en": {
            "subplot_gain": "Cumulative Gain Chart",
            "subplot_success": "Cumulative Success Rate Chart",
            "trace_model": "Model",
            "trace_random": "Random",
            "trace_ideal": "Ideal",
            "trace_success_selected": "Success Rate (Selected Range)",
            "trace_success_outside": "Success Rate (Outside Range)",
            "trace_success": "Success Rate",
            "x_contacted_pct": "Contacted Population Percentile (%)",
            "y_gain_share": "Gain / Share (%)",
            "y_success_rate": "Success Rate (%)",
            "annot_top": "Top {p}% -> SR {sr:.1f}%, Gain {gain:.1f}%",
            "annot_ks": "Optimal cutoff (max KS {ks:.1f}pp): {p}%",
            "annot_required": "Needed for target SR: {p}%",
            "hover_gain": "Contacted: %{x}%<br>Gain: %{y:.2f}%<extra></extra>",
            "hover_baseline_gain": "Contacted: %{x}%<br>Baseline gain: %{y:.2f}%<extra></extra>",
            "hover_ideal_gain": "Contacted: %{x}%<br>Ideal gain: %{y:.2f}%<extra></extra>",
            "hover_success_rate": "Contacted: %{x}%<br>Success rate: %{y:.2f}%<extra></extra>",
            "dist_all_clients": "All scored clients",
            "dist_campaign_clients": "Campaign clients",
            "dist_hover_all": "Percentile: %{x}<br>All clients: %{y}<extra></extra>",
            "dist_hover_campaign": "Percentile: %{x}<br>Campaign clients: %{y}<extra></extra>",
            "dist_x_title": "Model Score Percentile",
            "dist_y_title": "Client Count",
            "rate_bar_1": "Whole Base\n(Default)",
            "rate_bar_2": "Campaign Clients\n(Default)",
            "rate_bar_3": "Top-N by Score\n(Same Volume)",
            "rate_hover": "%{x}<br>Expected success rate: %{y:.2f}%<extra></extra>",
            "rate_y_title": "Expected Success Rate (%)",
            "figure_title": "Model Performance",
        },
        "cs": {
            "subplot_gain": "Graf kumulativního zisku",
            "subplot_success": "Graf kumulativní úspěšnosti",
            "trace_model": "Model",
            "trace_random": "Náhoda",
            "trace_ideal": "Ideál",
            "trace_success_selected": "Úspěšnost (vybraný rozsah)",
            "trace_success_outside": "Úspěšnost (mimo rozsah)",
            "trace_success": "Úspěšnost",
            "x_contacted_pct": "Percentil oslovené populace (%)",
            "y_gain_share": "Zisk / podíl (%)",
            "y_success_rate": "Úspěšnost (%)",
            "annot_top": "Top {p}% -> ÚSP {sr:.1f} %, zisk {gain:.1f} %",
            "annot_ks": "Optimální cut-off (max KS {ks:.1f} b.): {p} %",
            "annot_required": "Nutné pro cílovou úspěšnost: {p} %",
            "hover_gain": "Osloveno: %{x}%<br>Zisk: %{y:.2f}%<extra></extra>",
            "hover_baseline_gain": "Osloveno: %{x}%<br>Základní zisk: %{y:.2f}%<extra></extra>",
            "hover_ideal_gain": "Osloveno: %{x}%<br>Ideální zisk: %{y:.2f}%<extra></extra>",
            "hover_success_rate": "Osloveno: %{x}%<br>Úspěšnost: %{y:.2f}%<extra></extra>",
            "dist_all_clients": "Všichni skórovaní klienti",
            "dist_campaign_clients": "Klienti kampaně",
            "dist_hover_all": "Percentil: %{x}<br>Všichni klienti: %{y}<extra></extra>",
            "dist_hover_campaign": "Percentil: %{x}<br>Klienti kampaně: %{y}<extra></extra>",
            "dist_x_title": "Percentil skóre modelu",
            "dist_y_title": "Počet klientů",
            "rate_bar_1": "Celá báze\n(výchozí)",
            "rate_bar_2": "Klienti kampaně\n(výchozí)",
            "rate_bar_3": "Top-N dle skóre\n(stejný objem)",
            "rate_hover": "%{x}<br>Očekávaná úspěšnost: %{y:.2f}%<extra></extra>",
            "rate_y_title": "Očekávaná úspěšnost (%)",
            "figure_title": "Výkonnost modelu",
        },
    }
    if language not in texts:
        raise ValueError("language must be one of: en, cs")
    return texts[language]


def _required_cutoff_for_desired_rate(metrics: pd.DataFrame, desired_rate: float) -> int:
    eligible = metrics.loc[metrics["cumulative_success_rate_pct"] >= desired_rate, "contacted_percentile"]
    if eligible.empty:
        return 1
    return int(eligible.max())


def _success_bar_colors(metrics: pd.DataFrame, required_cutoff: int) -> list[str]:
    colors: list[str] = []
    for percentile in metrics["contacted_percentile"].astype(int):
        if percentile <= required_cutoff:
            colors.append("rgba(0, 87, 217, 0.80)")
        else:
            colors.append("rgba(148, 163, 184, 0.28)")
    return colors


def _success_rate_segments(
    metrics: pd.DataFrame,
    required_cutoff: int,
) -> tuple[list[float | None], list[float | None]]:
    selected: list[float | None] = []
    outside: list[float | None] = []
    for percentile, sr in zip(
        metrics["contacted_percentile"].astype(int),
        metrics["cumulative_success_rate_pct"].astype(float),
    ):
        if percentile <= required_cutoff:
            selected.append(sr)
            outside.append(None)
        else:
            selected.append(None)
            outside.append(sr)
    return selected, outside


def _make_figure(
    metrics: pd.DataFrame,
    default_percentile: int = 20,
    desired_success_rate: float | None = None,
    chart_text: dict[str, str] | None = None,
) -> go.Figure:
    if chart_text is None:
        chart_text = _resolve_chart_text("en")
    if desired_success_rate is None:
        desired_success_rate = float(
            metrics.loc[metrics["contacted_percentile"] == default_percentile, "cumulative_success_rate_pct"].iloc[0]
        )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(chart_text["subplot_gain"], chart_text["subplot_success"]),
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
            name=chart_text["trace_model"],
            line={"color": "#0057D9", "width": 3},
            marker={"size": 4},
            hovertemplate=chart_text["hover_gain"],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["random_baseline_gain_pct"],
            mode="lines",
            name=chart_text["trace_random"],
            line={"color": "#94A3B8", "width": 1.5, "dash": "dash"},
            hovertemplate=chart_text["hover_baseline_gain"],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["ideal_gain_pct"],
            mode="lines",
            name=chart_text["trace_ideal"],
            line={"color": "#CBD5E1", "width": 1.5, "dash": "dot"},
            hovertemplate=chart_text["hover_ideal_gain"],
        ),
        row=1,
        col=1,
    )
    selected_sr_segment, outside_sr_segment = _success_rate_segments(metrics=metrics, required_cutoff=required_cutoff)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=selected_sr_segment,
            mode="lines",
            name=chart_text["trace_success_selected"],
            line={"color": "#0057D9", "width": 3},
            hovertemplate=chart_text["hover_success_rate"],
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=outside_sr_segment,
            mode="lines",
            name=chart_text["trace_success_outside"],
            line={"color": "rgba(148,163,184,0.48)", "width": 2},
            hovertemplate=chart_text["hover_success_rate"],
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text=chart_text["x_contacted_pct"],
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
        title_text=chart_text["y_gain_share"],
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
        title_text=chart_text["y_success_rate"],
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
            text=chart_text["annot_top"].format(p=default_percentile, sr=selected_sr, gain=selected_gain),
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
            text=chart_text["annot_ks"].format(ks=best_ks_value, p=best_ks_percentile),
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
            text=chart_text["annot_required"].format(p=required_cutoff),
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
            "text": chart_text["figure_title"],
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


def _make_gain_figure(
    metrics: pd.DataFrame,
    default_percentile: int,
    chart_text: dict[str, str] | None = None,
) -> go.Figure:
    if chart_text is None:
        chart_text = _resolve_chart_text("en")
    x = metrics["contacted_percentile"]
    selected_row = metrics.loc[metrics["contacted_percentile"] == default_percentile].iloc[0]
    selected_gain = float(selected_row["gain_pct"])
    selected_sr = float(selected_row["cumulative_success_rate_pct"])
    best_ks_percentile = int(metrics["best_ks_percentile"].iat[0])
    best_ks_row = metrics.loc[metrics["contacted_percentile"] == best_ks_percentile].iloc[0]
    best_ks_value = float(best_ks_row["ks_pct"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["gain_pct"],
            mode="lines+markers",
            name=chart_text["trace_model"],
            line={"color": "#0057D9", "width": 3},
            marker={"size": 4},
            hovertemplate=chart_text["hover_gain"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["random_baseline_gain_pct"],
            mode="lines",
            name=chart_text["trace_random"],
            line={"color": "#94A3B8", "width": 1.5, "dash": "dash"},
            hovertemplate=chart_text["hover_baseline_gain"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=metrics["ideal_gain_pct"],
            mode="lines",
            name=chart_text["trace_ideal"],
            line={"color": "#CBD5E1", "width": 1.5, "dash": "dot"},
            hovertemplate=chart_text["hover_ideal_gain"],
        )
    )
    fig.update_xaxes(
        title_text=chart_text["x_contacted_pct"],
        range=[1, 100],
        showline=True,
        linewidth=1,
        linecolor="#94A3B8",
        ticks="outside",
        tickcolor="#94A3B8",
        dtick=10,
    )
    fig.update_yaxes(
        title_text=chart_text["y_gain_share"],
        range=[0, 105],
        showline=True,
        linewidth=1,
        linecolor="#94A3B8",
        gridcolor="#E2E8F0",
        zeroline=False,
    )
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=430,
        margin={"l": 72, "r": 24, "t": 56, "b": 60},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11, "color": "#334155"},
            "bgcolor": "rgba(255,255,255,0)",
        },
        annotations=[
            dict(
                x=default_percentile,
                y=selected_gain,
                xref="x",
                yref="y",
                text=chart_text["annot_top"].format(p=default_percentile, sr=selected_sr, gain=selected_gain),
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
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
                text=chart_text["annot_ks"].format(ks=best_ks_value, p=best_ks_percentile),
                showarrow=False,
                bgcolor="rgba(248,250,252,0.95)",
                bordercolor="#CBD5E1",
                borderwidth=1,
                borderpad=4,
                font={"size": 11, "color": "#334155"},
            ),
        ],
        shapes=[
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
        ],
        font={"family": "Segoe UI, Arial, sans-serif", "size": 13, "color": "#0F172A"},
    )
    return fig


def _make_success_rate_figure(
    metrics: pd.DataFrame,
    required_cutoff: int,
    chart_text: dict[str, str] | None = None,
) -> go.Figure:
    if chart_text is None:
        chart_text = _resolve_chart_text("en")
    fig = go.Figure()
    x = metrics["contacted_percentile"]
    fig.add_trace(
        go.Bar(
            x=x,
            y=metrics["cumulative_success_rate_pct"],
            name=chart_text["trace_success"],
            marker={
                "color": _success_bar_colors(metrics=metrics, required_cutoff=required_cutoff),
                "line": {"color": "rgba(15, 23, 42, 0.12)", "width": 0.6},
            },
            hovertemplate=chart_text["hover_success_rate"],
        )
    )
    y_top = max(5.0, float(metrics["cumulative_success_rate_pct"].max()) * 1.08)
    fig.update_xaxes(
        title_text=chart_text["x_contacted_pct"],
        range=[1, 100],
        showline=True,
        linewidth=1,
        linecolor="#94A3B8",
        ticks="outside",
        tickcolor="#94A3B8",
        dtick=10,
    )
    fig.update_yaxes(
        title_text=chart_text["y_success_rate"],
        range=[0, y_top],
        showline=True,
        linewidth=1,
        linecolor="#94A3B8",
        gridcolor="#E2E8F0",
        zeroline=False,
    )
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=430,
        margin={"l": 72, "r": 24, "t": 56, "b": 60},
        hovermode="x unified",
        showlegend=False,
        annotations=[
            dict(
                x=required_cutoff,
                y=max(5.0, float(metrics["cumulative_success_rate_pct"].max()) * 0.86),
                xref="x",
                yref="y",
                text=chart_text["annot_required"].format(p=required_cutoff),
                showarrow=False,
                bgcolor="rgba(255,251,235,0.96)",
                bordercolor="#D97706",
                borderwidth=1,
                borderpad=4,
                font={"size": 11, "color": "#92400E"},
            )
        ],
        shapes=[
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
            )
        ],
        font={"family": "Segoe UI, Arial, sans-serif", "size": 13, "color": "#0F172A"},
    )
    return fig


def _prepare_campaign_estimated_performance(
    latest_model_score: pd.DataFrame,
    campaign_clients: pd.DataFrame,
) -> pd.DataFrame:
    if "pt_unified_key" not in campaign_clients.columns:
        raise ValueError("campaign_clients must include column: pt_unified_key")
    required_cols = {"pt_unified_key", "score", "percentile"}
    missing = required_cols - set(latest_model_score.columns)
    if missing:
        raise ValueError(f"model_score is missing required columns for campaign estimation: {sorted(missing)}")

    selected_keys = set(campaign_clients["pt_unified_key"].dropna().astype(str))
    if not selected_keys:
        raise ValueError("campaign_clients.pt_unified_key is empty after removing null values.")

    scored = latest_model_score.copy()
    scored["pt_unified_key"] = scored["pt_unified_key"].astype(str)
    selected = scored.loc[scored["pt_unified_key"].isin(selected_keys)].copy()
    if selected.empty:
        raise ValueError("None of campaign_clients.pt_unified_key matched scored clients.")
    return selected


def _build_estimated_metrics_by_contact_percentile(campaign_scored: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"pt_unified_key", "score", "percentile"}
    missing = required_cols - set(campaign_scored.columns)
    if missing:
        raise ValueError(f"campaign_scored is missing required columns: {sorted(missing)}")

    scored = campaign_scored.copy()
    scored["score"] = pd.to_numeric(scored["score"], errors="coerce")
    if scored["score"].isna().any():
        raise ValueError("campaign_scored.score contains non-numeric values.")
    scored["score"] = scored["score"].clip(lower=0.0, upper=1.0)
    scored = scored.sort_values("score", ascending=False, ignore_index=True)

    n_clients = len(scored)
    if n_clients == 0:
        raise ValueError("No campaign-scored clients available to evaluate.")

    scored["cum_successes"] = scored["score"].cumsum()
    scored["cum_non_successes"] = (1.0 - scored["score"]).cumsum()
    scored["cum_clients"] = np.arange(1, n_clients + 1)

    total_successes = float(scored["score"].sum())
    total_non_successes = float(n_clients - total_successes)
    percentile_marks = np.arange(1, 101)
    idx = np.ceil(percentile_marks * n_clients / 100).astype(int) - 1
    points = scored.iloc[idx].copy().reset_index(drop=True)

    points["contacted_percentile"] = percentile_marks
    points["gain_pct"] = np.where(
        total_successes > 0.0,
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
        total_non_successes > 0.0,
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


def _campaign_covers_whole_base(latest_model_score: pd.DataFrame, campaign_scored: pd.DataFrame) -> bool:
    required = {"pt_unified_key"}
    missing_latest = required - set(latest_model_score.columns)
    missing_campaign = required - set(campaign_scored.columns)
    if missing_latest:
        raise ValueError(f"latest_model_score is missing required columns: {sorted(missing_latest)}")
    if missing_campaign:
        raise ValueError(f"campaign_scored is missing required columns: {sorted(missing_campaign)}")

    all_keys = set(latest_model_score["pt_unified_key"].dropna().astype(str))
    campaign_keys = set(campaign_scored["pt_unified_key"].dropna().astype(str))
    return all_keys == campaign_keys


def _resolve_campaign_metrics_for_report(
    latest_model_score: pd.DataFrame,
    campaign_scored: pd.DataFrame,
    actual_metrics: pd.DataFrame,
    estimated_metrics: pd.DataFrame,
) -> pd.DataFrame:
    # If campaign selection is the full base, align bottom charts to top observed metrics.
    if _campaign_covers_whole_base(latest_model_score=latest_model_score, campaign_scored=campaign_scored):
        return actual_metrics.copy()
    return estimated_metrics


def _build_campaign_percentile_distribution(
    latest_model_score: pd.DataFrame,
    campaign_scored: pd.DataFrame,
) -> pd.DataFrame:
    if "percentile" not in latest_model_score.columns:
        raise ValueError("model_score is missing required column: percentile")
    if "percentile" not in campaign_scored.columns:
        raise ValueError("campaign_scored is missing required column: percentile")

    all_counts = latest_model_score.groupby("percentile", as_index=False).size().rename(columns={"size": "all_clients"})
    selected_counts = (
        campaign_scored.groupby("percentile", as_index=False).size().rename(columns={"size": "campaign_clients"})
    )
    distribution = all_counts.merge(selected_counts, on="percentile", how="left")
    distribution["campaign_clients"] = distribution["campaign_clients"].fillna(0).astype(int)
    distribution = distribution.sort_values("percentile", ascending=True, ignore_index=True)
    return distribution


def _make_campaign_distribution_figure(
    distribution: pd.DataFrame,
    chart_text: dict[str, str] | None = None,
) -> go.Figure:
    if chart_text is None:
        chart_text = _resolve_chart_text("en")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=distribution["percentile"],
            y=distribution["all_clients"],
            name=chart_text["dist_all_clients"],
            marker={"color": "rgba(148, 163, 184, 0.60)"},
            hovertemplate=chart_text["dist_hover_all"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=distribution["percentile"],
            y=distribution["campaign_clients"],
            name=chart_text["dist_campaign_clients"],
            marker={"color": "rgba(0, 87, 217, 0.85)"},
            hovertemplate=chart_text["dist_hover_campaign"],
        )
    )
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=280,
        margin={"l": 70, "r": 20, "t": 28, "b": 56},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11, "color": "#334155"},
        },
        font={"family": "Segoe UI, Arial, sans-serif", "size": 12, "color": "#0F172A"},
    )
    fig.update_xaxes(
        title_text=chart_text["dist_x_title"],
        dtick=5,
        showline=True,
        linewidth=1,
        linecolor="#94A3B8",
        tickangle=0,
    )
    fig.update_yaxes(
        title_text=chart_text["dist_y_title"],
        rangemode="tozero",
        showline=True,
        linewidth=1,
        linecolor="#94A3B8",
        gridcolor="#E2E8F0",
    )
    return fig


def _make_campaign_rate_comparison_figure(
    whole_base_default_sr_pct: float,
    campaign_default_sr_pct: float,
    top_equal_volume_sr_pct: float,
    chart_text: dict[str, str] | None = None,
) -> go.Figure:
    if chart_text is None:
        chart_text = _resolve_chart_text("en")
    labels = [
        chart_text["rate_bar_1"],
        chart_text["rate_bar_2"],
        chart_text["rate_bar_3"],
    ]
    values = [
        whole_base_default_sr_pct,
        campaign_default_sr_pct,
        top_equal_volume_sr_pct,
    ]
    colors = ["#94A3B8", "#0057D9", "#0EA5A4"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker={"color": colors},
                text=[f"{v:.1f}%" for v in values],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=chart_text["rate_hover"],
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=280,
        margin={"l": 56, "r": 14, "t": 28, "b": 56},
        showlegend=False,
        font={"family": "Segoe UI, Arial, sans-serif", "size": 12, "color": "#0F172A"},
    )
    fig.update_yaxes(
        title_text=chart_text["rate_y_title"],
        rangemode="tozero",
        gridcolor="#E2E8F0",
        showline=True,
        linewidth=1,
        linecolor="#94A3B8",
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="#94A3B8")
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
            "Cumulative Success Rate Within Campaign Volume",
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
            cliponaxis=False,
            hovertemplate="%{x}<br>Success rate: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=model_guided_curve["contact_share_pct"],
            y=model_guided_curve["estimated_success_rate_pct"],
            name=f"Model-guided cumulative SR ({actual_period_label})",
            marker={
                "color": "rgba(37,99,235,0.82)",
                "line": {"color": "rgba(255,255,255,0.65)", "width": 0.5},
            },
            hovertemplate="Model-guided within campaign volume: %{x}%<br>Cumulative SR: %{y:.2f}%<extra></extra>",
            opacity=0.95,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=selected_curve["contact_share_pct"],
            y=selected_curve["estimated_success_rate_pct"],
            mode="lines",
            line={"color": "#0EA5E9", "width": 2.0, "dash": "dash"},
            name=f"Selected cumulative SR ({actual_period_label})",
            hovertemplate="Selected clients within campaign volume: %{x}%<br>Cumulative SR: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    selected_full = float(selection_summary["selected_rate_estimated_pct"])
    model_full = float(selection_summary["model_guided_rate_estimated_pct"])
    fig.update_layout(
        barmode="overlay",
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
        margin={"l": 72, "r": 30, "t": 96, "b": 60},
    )
    top_max = max(benchmark_values) if benchmark_values else 0.0
    fig.update_yaxes(
        title_text="Success Rate (%)",
        row=1,
        col=1,
        range=[0, max(5.0, top_max * 1.18)],
        gridcolor="#E2E8F0",
    )
    fig.update_yaxes(title_text="Estimated Success Rate (%)", row=2, col=1, rangemode="tozero", gridcolor="#E2E8F0")
    fig.update_xaxes(title_text="Contacted Share Within Campaign Volume (%)", row=2, col=1, dtick=10, range=[1, 100])
    return fig


def EvaluateModel(
    output_html_path: str | Path = "outputs/model_evaluation_report.html",
    seed: int | None = 42,
    include_campaign_selection: bool = False,
    campaign_clients: pd.DataFrame | None = None,
    historical_period: str | pd.Timestamp | None = None,
    historical_period_start: str | pd.Timestamp | None = None,
    historical_period_end: str | pd.Timestamp | None = None,
    language: str = "en",
) -> Path:
    historical_period = _resolve_historical_period_input(
        historical_period=historical_period,
        historical_period_start=historical_period_start,
        historical_period_end=historical_period_end,
    )
    labels = _resolve_report_language(language=language)
    chart_text = _resolve_chart_text(language=language)
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
    gain_figure = _make_gain_figure(metrics=metrics, default_percentile=selected_percentile, chart_text=chart_text)
    success_figure = _make_success_rate_figure(
        metrics=metrics,
        required_cutoff=required_cutoff,
        chart_text=chart_text,
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
    campaign_cutoff_points: list[dict[str, float | int]] = []
    campaign_best_ks_percentile = selected_percentile
    if include_campaign_selection:
        if campaign_clients is None:
            raise ValueError("campaign_clients must be provided when include_campaign_selection=True.")
        campaign_scored = _prepare_campaign_estimated_performance(
            latest_model_score=latest_model_score,
            campaign_clients=campaign_clients,
        )
        campaign_estimated_metrics = _build_estimated_metrics_by_contact_percentile(campaign_scored=campaign_scored)
        campaign_metrics = _resolve_campaign_metrics_for_report(
            latest_model_score=latest_model_score,
            campaign_scored=campaign_scored,
            actual_metrics=metrics,
            estimated_metrics=campaign_estimated_metrics,
        )
        campaign_distribution = _build_campaign_percentile_distribution(
            latest_model_score=latest_model_score,
            campaign_scored=campaign_scored,
        )
        whole_base_scores = pd.to_numeric(latest_model_score["score"], errors="coerce").clip(lower=0.0, upper=1.0)
        if whole_base_scores.isna().any():
            raise ValueError("latest_model_score.score contains non-numeric values.")
        campaign_scores = pd.to_numeric(campaign_scored["score"], errors="coerce").clip(lower=0.0, upper=1.0)
        if campaign_scores.isna().any():
            raise ValueError("campaign_scored.score contains non-numeric values.")
        whole_base_default_sr_pct = float(whole_base_scores.mean() * 100.0)
        campaign_default_sr_pct = float(campaign_scores.mean() * 100.0)
        top_equal_volume = (
            latest_model_score[["pt_unified_key", "score"]]
            .copy()
            .assign(score=lambda df: pd.to_numeric(df["score"], errors="coerce").clip(lower=0.0, upper=1.0))
            .dropna(subset=["score"])
            .sort_values("score", ascending=False, ignore_index=True)
            .head(len(campaign_scored))
        )
        if top_equal_volume.empty:
            raise ValueError("Unable to compute top-equal-volume success rate because no scored clients are available.")
        top_equal_volume_sr_pct = float(top_equal_volume["score"].mean() * 100.0)
        campaign_required_cutoff = _required_cutoff_for_desired_rate(
            metrics=campaign_metrics,
            desired_rate=desired_success_rate,
        )
        campaign_gain_figure = _make_gain_figure(
            metrics=campaign_metrics,
            default_percentile=selected_percentile,
            chart_text=chart_text,
        )
        campaign_success_figure = _make_success_rate_figure(
            metrics=campaign_metrics,
            required_cutoff=campaign_required_cutoff,
            chart_text=chart_text,
        )
        campaign_distribution_figure = _make_campaign_distribution_figure(
            distribution=campaign_distribution,
            chart_text=chart_text,
        )
        campaign_rate_compare_figure = _make_campaign_rate_comparison_figure(
            whole_base_default_sr_pct=whole_base_default_sr_pct,
            campaign_default_sr_pct=campaign_default_sr_pct,
            top_equal_volume_sr_pct=top_equal_volume_sr_pct,
            chart_text=chart_text,
        )

        campaign_distribution_html = campaign_distribution_figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="campaign-distribution-figure",
        )
        campaign_rate_compare_html = campaign_rate_compare_figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="campaign-rate-compare-figure",
        )
        campaign_gain_html = campaign_gain_figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="campaign-gain-figure",
        )
        campaign_success_html = campaign_success_figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="campaign-success-figure",
        )
        campaign_cutoff_points = [
            {
                "p": int(row["contacted_percentile"]),
                "clients": int(row["cum_clients"]),
                "gain": float(row["gain_pct"]),
                "sr": float(row["cumulative_success_rate_pct"]),
                "captured": float(row["cum_successes"]),
                "total": float(row["total_successes"]),
                "captured_pct": (
                    float(row["cum_successes"]) / float(row["total_successes"]) * 100.0
                    if float(row["total_successes"]) > 0
                    else 0.0
                ),
            }
            for _, row in campaign_metrics.iterrows()
        ]
        campaign_best_ks_percentile = int(campaign_metrics["best_ks_percentile"].iat[0])
        campaign_section_html = f"""
    <section class="campaign-section">
      <h2>{labels["campaign_section_title"]}</h2>
      <p>
        {labels["campaign_intro"]}
      </p>
      <div class="campaign-top-row">
        <div class="plot-card" id="campaign-distribution-card">
          <div class="plot-card-head">
            <h3>{labels["campaign_distribution_title"]}</h3>
            <div class="tooltip-wrap">
              <button class="tooltip-btn" type="button" aria-describedby="tooltip-campaign-distribution">?</button>
              <div class="tooltip-body" id="tooltip-campaign-distribution">
                <strong>{labels["tooltip_how_to_read"]}</strong>
                <p>Each percentile bucket shows where campaign clients sit in the model ranking.</p>
                <ul>
                  <li><strong>Gray bars:</strong> full scored population per percentile.</li>
                  <li><strong>Blue bars:</strong> your campaign clients in the same percentile.</li>
                  <li>More blue mass in higher percentiles means stronger targeting quality.</li>
                </ul>
              </div>
            </div>
          </div>
          {campaign_distribution_html}
        </div>
        <div class="plot-card" id="campaign-rate-compare-card">
          <div class="plot-card-head">
            <h3>{labels["campaign_rate_compare_title"]}</h3>
            <div class="tooltip-wrap">
              <button class="tooltip-btn" type="button" aria-describedby="tooltip-campaign-rate-compare">?</button>
              <div class="tooltip-body" id="tooltip-campaign-rate-compare">
                <strong>{labels["tooltip_how_to_read"]}</strong>
                <p>Bars compare expected success rate under three business scenarios.</p>
                <ul>
                  <li><strong>Whole Base (Default):</strong> baseline expected quality at the default cutoff.</li>
                  <li><strong>Campaign Clients (Default):</strong> expected quality of your selected audience.</li>
                  <li><strong>Top-N by Score:</strong> the model-optimal audience at the same campaign size.</li>
                </ul>
              </div>
            </div>
          </div>
          {campaign_rate_compare_html}
        </div>
      </div>
      <div class="chart-grid chart-grid-2">
        <div class="plot-card" id="campaign-gain-card">
          <div class="plot-card-head">
            <h3>{labels["campaign_gain_title"]}</h3>
            <div class="tooltip-wrap">
              <button class="tooltip-btn" type="button" aria-describedby="tooltip-campaign-gain">?</button>
              <div class="tooltip-body" id="tooltip-campaign-gain">
                <strong>{labels["tooltip_how_to_read"]}</strong>
                <p>Shows how much expected success volume is captured as contact depth increases.</p>
                <ul>
                  <li>Blue model line above gray random baseline indicates ranking value.</li>
                  <li>The red marker tracks the currently selected cutoff from the slider.</li>
                </ul>
              </div>
            </div>
          </div>
          {campaign_gain_html}
        </div>
        <div class="plot-card" id="campaign-success-card">
          <div class="plot-card-head">
            <h3>{labels["campaign_success_title"]}</h3>
            <div class="tooltip-wrap">
              <button class="tooltip-btn" type="button" aria-describedby="tooltip-campaign-success">?</button>
              <div class="tooltip-body" id="tooltip-campaign-success">
                <strong>{labels["tooltip_how_to_read"]}</strong>
                <p>Bars show cumulative expected conversion quality inside the campaign-only base.</p>
                <ul>
                  <li>Blue bars are within the cutoff required to hit desired success rate.</li>
                  <li>Gray bars are outside that quality threshold.</li>
                </ul>
              </div>
            </div>
          </div>
          {campaign_success_html}
        </div>
      </div>
    </section>
"""

    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_gain_html = gain_figure.to_html(full_html=False, include_plotlyjs=True, div_id="top-gain-figure")
    top_success_html = success_figure.to_html(full_html=False, include_plotlyjs=False, div_id="top-success-figure")
    html = f"""<!DOCTYPE html>
<html lang="{labels["html_lang"]}">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{labels["report_title"]}</title>
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
      display: block;
    }}
    .chart-grid {{
      display: grid;
      gap: 12px;
      align-items: stretch;
    }}
    .chart-grid-2 {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .plot-card {{
      border: 1px solid var(--line-soft);
      border-radius: 12px;
      background: #FFFFFF;
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
      padding: 8px 10px 6px 10px;
      min-width: 0;
    }}
    .campaign-top-row {{
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
      gap: 12px;
      margin: 8px 0 12px 0;
      align-items: stretch;
    }}
    .plot-card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 4px 6px 0 6px;
    }}
    .plot-card-head h3 {{
      margin: 0;
      font-size: 16px;
      color: var(--text-main);
      font-weight: 700;
    }}
    .tooltip-wrap {{
      position: relative;
      display: inline-flex;
      align-items: center;
    }}
    .tooltip-btn {{
      width: 24px;
      height: 24px;
      border-radius: 50%;
      border: 1px solid #CBD5E1;
      background: #F8FAFC;
      color: #0F172A;
      font-size: 13px;
      font-weight: 700;
      cursor: default;
      line-height: 1;
      padding: 0;
    }}
    .tooltip-body {{
      position: absolute;
      right: 0;
      top: 30px;
      min-width: 260px;
      max-width: 320px;
      background: #0F172A;
      color: #F8FAFC;
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.35;
      display: none;
      z-index: 12;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.3);
    }}
    .tooltip-body strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      color: #FFFFFF;
    }}
    .tooltip-body p {{
      margin: 0 0 6px 0;
      font-size: 12px;
      color: #E2E8F0;
      line-height: 1.35;
    }}
    .tooltip-body ul {{
      margin: 0;
      padding-left: 16px;
    }}
    .tooltip-body li {{
      margin: 0 0 4px 0;
      color: #E2E8F0;
      line-height: 1.3;
      font-size: 12px;
    }}
    .tooltip-wrap:hover .tooltip-body {{
      display: block;
    }}
    .tooltip-wrap:focus-within .tooltip-body {{
      display: block;
    }}
    .figure-panel {{
      min-width: 0;
    }}
    .guide-panel {{
      display: none;
    }}
    .header {{
      border-bottom: 1px solid var(--line-soft);
      padding-bottom: 12px;
      gap: 16px;
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
    @media (max-width: 1100px) {{
      .campaign-top-row {{
        grid-template-columns: 1fr;
      }}
      .chart-grid-2 {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>{labels["header_title"]}</h1>
      <p>
        {labels["header_subtitle"]}
      </p>
    </div>
    <div class="kpis">
      <div class="kpi">
        <div class="kpi-label">{labels["kpi_selected_cutoff"]}</div>
        <div class="kpi-value" id="kpi-cutoff">{labels["top_word"]} {summary["selected_percentile"]}% ({summary["contacted_clients"]:,})</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">{labels["kpi_lift_gain"]}</div>
        <div class="kpi-value" id="kpi-gain">{summary["gain_pct"]:.1f}%</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">{labels["kpi_success_rate"]}</div>
        <div class="kpi-value" id="kpi-sr">{summary["success_rate_pct"]:.1f}%</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">{labels["kpi_captured"]}</div>
        <div class="kpi-value" id="kpi-captured">{summary["captured_successes"]:,} / {summary["total_successes"]:,} ({summary["captured_pct"]:.1f}%)</div>
      </div>
    </div>
    <div class="controls">
      <label for="cutoff-slider">{labels["label_cutoff_slider"]}</label>
      <input id="cutoff-slider" type="range" min="1" max="100" step="1" value="{summary["selected_percentile"]}" />
      <span id="cutoff-value">{summary["selected_percentile"]}%</span>
      <label for="desired-rate-slider">{labels["label_desired_rate_slider"]}</label>
      <input
        id="desired-rate-slider"
        type="range"
        min="{metrics['cumulative_success_rate_pct'].min():.1f}"
        max="{metrics['cumulative_success_rate_pct'].max():.1f}"
        step="0.1"
        value="{desired_success_rate:.1f}"
      />
      <span id="desired-rate-value">{desired_success_rate:.1f}%</span>
      <span id="required-cutoff-value">{labels["required_cutoff_prefix"]}: {required_cutoff}%</span>
    </div>
    <div class="content-grid">
      <div class="chart-grid chart-grid-2">
        <div class="plot-card" id="top-gain-card">
          <div class="plot-card-head">
            <h3>{labels["top_gain_title"]}</h3>
            <div class="tooltip-wrap">
              <button class="tooltip-btn" type="button" aria-describedby="tooltip-top-gain">?</button>
              <div class="tooltip-body" id="tooltip-top-gain">
                <strong>{labels["tooltip_how_to_read"]}</strong>
                <p>The gain chart answers how much of total potential you capture at each targeting depth.</p>
                <ul>
                  <li><strong>Model vs Random:</strong> earlier blue-vs-gray separation means better prioritization.</li>
                  <li><strong>Red cutoff line:</strong> your current operating point from the slider.</li>
                  <li>Use it to balance campaign reach versus captured value.</li>
                </ul>
              </div>
            </div>
          </div>
          {top_gain_html}
        </div>
        <div class="plot-card" id="top-success-card">
          <div class="plot-card-head">
            <h3>{labels["top_success_title"]}</h3>
            <div class="tooltip-wrap">
              <button class="tooltip-btn" type="button" aria-describedby="tooltip-top-success">?</button>
              <div class="tooltip-body" id="tooltip-top-success">
                <strong>{labels["tooltip_how_to_read"]}</strong>
                <p>This chart shows cumulative quality of the contacted population.</p>
                <ul>
                  <li>Set a desired success rate target with the slider.</li>
                  <li>The required cutoff marker shows the largest audience that still meets that target.</li>
                  <li>Blue bars are within target; gray bars are outside target.</li>
                </ul>
              </div>
            </div>
          </div>
          {top_success_html}
        </div>
      </div>
      <div class="footnote">
        Definitions: Gain = cumulative share of all successes captured up to the selected percentile.
        Success rate = cumulative successes divided by cumulative contacted clients.
        Success is defined as an observed event from scoring time through one calendar month.
      </div>
    </div>
    {campaign_section_html}
  </div>
    <script>
    const cutoffData = {json.dumps(cutoff_points)};
    const campaignCutoffData = {json.dumps(campaign_cutoff_points)};
    const bestKsPercentile = {best_ks_percentile};
    const campaignBestKsPercentile = {campaign_best_ks_percentile};
    const numberLocale = "{labels["number_locale"]}";
    const labelTop = "{labels["top_word"]}";
    const labelSr = "{labels["abbr_sr"]}";
    const labelGain = "{labels["abbr_gain"]}";
    const requiredCutoffPrefix = "{labels["required_cutoff_prefix"]}";
    const neededForTargetPrefix = "{labels["required_cutoff_prefix"]}";

    function formatInt(x) {{
      return Number(x).toLocaleString(numberLocale);
    }}

    function pointFor(p, points) {{
      return points[p - 1];
    }}

    function updateKpis(point) {{
      document.getElementById("kpi-cutoff").textContent = `${{labelTop}} ${{point.p}}% (${{formatInt(point.clients)}})`;
      document.getElementById("kpi-gain").textContent = `${{point.gain.toFixed(1)}}%`;
      document.getElementById("kpi-sr").textContent = `${{point.sr.toFixed(1)}}%`;
      document.getElementById("kpi-captured").textContent =
        `${{formatInt(point.captured)}} / ${{formatInt(point.total)}} (${{point.captured_pct.toFixed(1)}}%)`;
      document.getElementById("cutoff-value").textContent = `${{point.p}}%`;
    }}

    function requiredCutoffForDesired(desiredRate, points) {{
      let required = 1;
      for (const point of points) {{
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

    function updateGainFigure(divId, point, ksPercentile) {{
      const gd = document.getElementById(divId);
      if (!gd || !gd.layout || !gd.layout.shapes || gd.layout.shapes.length < 2) {{
        return false;
      }}
      Plotly.relayout(gd, {{
        "shapes[0].x0": point.p,
        "shapes[0].x1": point.p,
        "annotations[0].x": point.p,
        "annotations[0].y": point.gain,
        "annotations[0].text": `${{labelTop}} ${{point.p}}% -> ${{labelSr}} ${{point.sr.toFixed(1)}}%, ${{labelGain}} ${{point.gain.toFixed(1)}}%`,
        "annotations[1].x": ksPercentile
      }});
      return true;
    }}

    function updateSuccessFigure(divId, points, required) {{
      const gd = document.getElementById(divId);
      if (!gd || !gd.layout || !gd.layout.shapes || gd.layout.shapes.length < 1) {{
        return false;
      }}
      const colors = points.map((point) => {{
        if (point.p <= required) {{
          return "rgba(0,87,217,0.80)";
        }}
        return "rgba(148,163,184,0.28)";
      }});
      Plotly.restyle(gd, {{
        "marker.color": [colors]
      }}, [0]);
      Plotly.relayout(gd, {{
        "shapes[0].x0": required,
        "shapes[0].x1": required,
        "annotations[0].x": required,
        "annotations[0].text": `${{neededForTargetPrefix}}: ${{required}}%`
      }});
      return true;
    }}

    function updateDesiredRateUi(desiredRate, forcedRequired = null) {{
      const required = forcedRequired === null ? requiredCutoffForDesired(desiredRate, cutoffData) : forcedRequired;
      document.getElementById("desired-rate-value").textContent = `${{desiredRate.toFixed(1)}}%`;
      document.getElementById("required-cutoff-value").textContent = `${{requiredCutoffPrefix}}: ${{required}}%`;
      const okTop = updateSuccessFigure("top-success-figure", cutoffData, required);
      const okCampaign = campaignCutoffData.length > 0
        ? updateSuccessFigure("campaign-success-figure", campaignCutoffData, required)
        : true;
      return okTop && okCampaign;
    }}

    function applyCutoff(percentile) {{
      const point = pointFor(percentile, cutoffData);
      if (!point) {{
        return;
      }}
      const required = point.p;
      desiredRateSlider.value = point.sr.toFixed(1);
      updateKpis(point);
      updateGainFigure("top-gain-figure", point, bestKsPercentile);
      if (campaignCutoffData.length > 0) {{
        const campaignPoint = pointFor(percentile, campaignCutoffData);
        if (campaignPoint) {{
          updateGainFigure("campaign-gain-figure", campaignPoint, campaignBestKsPercentile);
        }}
      }}
      updateDesiredRateUi(Number(desiredRateSlider.value), required);
    }}

    function applyDesiredRate(desiredRate) {{
      const required = requiredCutoffForDesired(desiredRate, cutoffData);
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
      const ok = updateGainFigure("top-gain-figure", pointFor(Number(slider.value), cutoffData), bestKsPercentile)
        && updateDesiredRateUi(Number(desiredRateSlider.value));
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
