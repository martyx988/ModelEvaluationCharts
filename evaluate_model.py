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


def EvaluateModel(
    output_html_path: str | Path = "outputs/model_evaluation_report.html",
    seed: int | None = 42,
) -> Path:
    model_score, target_store, _ = create_simulated_tables(seed=seed)
    performance = _prepare_performance_data(model_score=model_score, target_store=target_store)
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

    function updateDesiredRateUi(desiredRate) {{
      const required = requiredCutoffForDesired(desiredRate);
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
      desiredRateSlider.value = point.sr.toFixed(1);
      updateKpis(point);
      updateFigure(point);
      updateDesiredRateUi(Number(desiredRateSlider.value));
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
