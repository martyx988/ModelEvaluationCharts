from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from evaluate_model import (
    EvaluateModel,
    _build_estimated_metrics_by_contact_percentile,
    _build_metrics_by_contact_percentile,
    _make_campaign_rate_comparison_figure,
    _make_gain_figure,
    _resolve_chart_text,
    _resolve_campaign_metrics_for_report,
    _make_figure,
    _prepare_campaign_estimated_performance,
    _prepare_performance_data,
)
from simulated_data import create_simulated_tables


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


def test_figure_has_presentation_trace_set_and_cutoff_markers() -> None:
    metrics = _build_metrics_by_contact_percentile(_performance_fixture())
    figure = _make_figure(metrics, default_percentile=20)

    trace_names = {trace.name for trace in figure.data}
    assert "Model" in trace_names
    assert "Random" in trace_names
    assert "Ideal" in trace_names
    assert "Success Rate (Selected Range)" in trace_names
    assert "Success Rate (Outside Range)" in trace_names
    assert "Cumulative Non-Success Share" not in trace_names
    assert "KS" not in trace_names
    success_trace = next(trace for trace in figure.data if trace.name == "Success Rate (Selected Range)")
    assert success_trace.type == "scatter"

    layout_dict = figure.to_plotly_json().get("layout", {})
    assert "sliders" not in layout_dict or not layout_dict["sliders"]
    shapes = layout_dict.get("shapes", [])
    assert any(shape.get("name") == "selected_cutoff" for shape in shapes)
    assert any(shape.get("name") == "ks_optimal_split" for shape in shapes)


def test_generated_report_contains_interactive_cutoff_control(tmp_path) -> None:
    output = tmp_path / "report.html"
    EvaluateModel(output_html_path=output, seed=42)
    html = output.read_text(encoding="utf-8")
    assert 'id="cutoff-slider"' in html
    assert 'id="desired-rate-slider"' in html
    assert 'id="required-cutoff-value"' in html
    assert "Plotly.relayout" in html
    assert "desiredRateSlider.value = point.sr.toFixed(1)" in html
    assert "slider.value = String(required)" in html
    assert "point.p <= required" in html
    assert "const required = point.p;" in html
    assert 'id="top-gain-card"' in html
    assert 'id="top-success-card"' in html
    assert 'id="tooltip-top-gain"' in html
    assert 'id="tooltip-top-success"' in html
    assert "plot-card-head" in html


def test_generated_report_includes_campaign_selection_section_when_enabled(tmp_path) -> None:
    model_score, _, _ = create_simulated_tables(seed=42)
    campaign_clients = model_score[["pt_unified_key"]].head(50).copy()
    output = tmp_path / "report_selection.html"
    EvaluateModel(
        output_html_path=output,
        seed=42,
        include_campaign_selection=True,
        campaign_clients=campaign_clients,
    )
    html = output.read_text(encoding="utf-8")
    assert "Campaign Selection Potential" in html
    assert "Expected successes are estimated from model score sums" in html
    assert 'id="campaign-distribution-figure"' in html
    assert 'id="campaign-rate-compare-figure"' in html
    assert 'id="campaign-gain-figure"' in html
    assert 'id="campaign-success-figure"' in html
    assert 'id="tooltip-campaign-distribution"' in html
    assert 'id="tooltip-campaign-rate-compare"' in html
    assert 'id="tooltip-campaign-gain"' in html
    assert 'id="tooltip-campaign-success"' in html
    assert "How to read this chart" in html


def test_simulated_tables_include_latest_january_scores() -> None:
    model_score, _, _ = create_simulated_tables(seed=42)
    fs_time = pd.to_datetime(model_score["fs_time"])
    assert fs_time.nunique() > 1
    assert fs_time.max() == pd.Timestamp("2026-01-31")


def test_simulated_scores_mass_aligns_with_target_event_volume() -> None:
    model_score, target_store, _ = create_simulated_tables(seed=42)
    total_expected_successes = float(pd.to_numeric(model_score["score"], errors="coerce").sum())
    total_events = len(target_store)
    assert abs(total_expected_successes - total_events) < 20


def test_simulated_campaign_expected_rate_ordering() -> None:
    model_score, _, campaign_clients = create_simulated_tables(seed=42)
    fs_time = pd.to_datetime(model_score["fs_time"])
    latest = model_score.loc[fs_time == fs_time.max(), ["pt_unified_key", "score"]].copy()
    latest["score"] = pd.to_numeric(latest["score"], errors="coerce").clip(0.0, 1.0)
    whole_rate = float(latest["score"].mean() * 100.0)

    selected = latest.loc[latest["pt_unified_key"].isin(set(campaign_clients["pt_unified_key"]))].copy()
    campaign_rate = float(selected["score"].mean() * 100.0)

    top_same_volume = latest.sort_values("score", ascending=False).head(len(selected)).copy()
    top_rate = float(top_same_volume["score"].mean() * 100.0)

    assert whole_rate <= campaign_rate <= top_rate


def test_campaign_selection_report_displays_actual_and_historical_periods(tmp_path) -> None:
    model_score, _, _ = create_simulated_tables(seed=42)
    campaign_clients = model_score[["pt_unified_key"]].drop_duplicates().head(120).copy()
    output = tmp_path / "report_selection_periods.html"
    EvaluateModel(
        output_html_path=output,
        seed=42,
        include_campaign_selection=True,
        campaign_clients=campaign_clients,
        historical_period="2025-12-31",
    )
    html = output.read_text(encoding="utf-8")
    assert "Campaign Selection Potential" in html
    assert "Expected successes are estimated from model score sums" in html


def test_evaluatemodel_accepts_legacy_historical_period_start_end_kwargs(tmp_path) -> None:
    model_score, _, _ = create_simulated_tables(seed=42)
    campaign_clients = model_score[["pt_unified_key"]].drop_duplicates().head(120).copy()
    output = tmp_path / "legacy_period_args.html"

    EvaluateModel(
        output_html_path=output,
        seed=42,
        include_campaign_selection=True,
        campaign_clients=campaign_clients,
        historical_period_start="2025-12-01",
        historical_period_end="2025-12-31",
    )

    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "Campaign Selection Potential" in html


def test_evaluatemodel_rejects_conflicting_historical_args(tmp_path) -> None:
    model_score, _, _ = create_simulated_tables(seed=42)
    campaign_clients = model_score[["pt_unified_key"]].drop_duplicates().head(120).copy()
    output = tmp_path / "conflict_period_args.html"

    with pytest.raises(ValueError, match="Use either historical_period or historical_period_start/end"):
        EvaluateModel(
            output_html_path=output,
            seed=42,
            include_campaign_selection=True,
            campaign_clients=campaign_clients,
            historical_period="2025-12-31",
            historical_period_start="2025-12-01",
        )


def test_evaluatemodel_rejects_legacy_period_month_mismatch(tmp_path) -> None:
    model_score, _, _ = create_simulated_tables(seed=42)
    campaign_clients = model_score[["pt_unified_key"]].drop_duplicates().head(120).copy()
    output = tmp_path / "mismatch_period_args.html"

    with pytest.raises(ValueError, match="must fall in the same calendar month"):
        EvaluateModel(
            output_html_path=output,
            seed=42,
            include_campaign_selection=True,
            campaign_clients=campaign_clients,
            historical_period_start="2025-11-30",
            historical_period_end="2025-12-01",
        )


def test_all_client_campaign_curves_align_with_top_curves_for_simulated_data() -> None:
    model_score, target_store, _ = create_simulated_tables(seed=42)
    scored = model_score.copy()
    scored["fs_time"] = pd.to_datetime(scored["fs_time"])
    latest = scored.loc[scored["fs_time"] == scored["fs_time"].max()].copy()

    top_performance = _prepare_performance_data(model_score=latest, target_store=target_store)
    top_metrics = _build_metrics_by_contact_percentile(performance=top_performance)

    all_clients = latest[["pt_unified_key"]].copy()
    campaign_scored = _prepare_campaign_estimated_performance(
        latest_model_score=latest,
        campaign_clients=all_clients,
    )
    campaign_metrics = _build_estimated_metrics_by_contact_percentile(campaign_scored=campaign_scored)

    merged = top_metrics[
        ["contacted_percentile", "gain_pct", "cumulative_success_rate_pct"]
    ].merge(
        campaign_metrics[["contacted_percentile", "gain_pct", "cumulative_success_rate_pct"]],
        on="contacted_percentile",
        suffixes=("_actual", "_estimated"),
    )

    max_gain_diff = (merged["gain_pct_actual"] - merged["gain_pct_estimated"]).abs().max()
    max_sr_diff = (
        merged["cumulative_success_rate_pct_actual"] - merged["cumulative_success_rate_pct_estimated"]
    ).abs().max()
    assert max_gain_diff < 8.0
    assert max_sr_diff < 8.0


def test_all_client_campaign_ks_cutoff_matches_top_metrics() -> None:
    model_score, target_store, _ = create_simulated_tables(seed=42)
    scored = model_score.copy()
    scored["fs_time"] = pd.to_datetime(scored["fs_time"])
    latest = scored.loc[scored["fs_time"] == scored["fs_time"].max()].copy()

    top_performance = _prepare_performance_data(model_score=latest, target_store=target_store)
    top_metrics = _build_metrics_by_contact_percentile(performance=top_performance)

    all_clients = latest[["pt_unified_key"]].copy()
    campaign_scored = _prepare_campaign_estimated_performance(
        latest_model_score=latest,
        campaign_clients=all_clients,
    )
    campaign_estimated_metrics = _build_estimated_metrics_by_contact_percentile(campaign_scored=campaign_scored)
    campaign_metrics = _resolve_campaign_metrics_for_report(
        latest_model_score=latest,
        campaign_scored=campaign_scored,
        actual_metrics=top_metrics,
        estimated_metrics=campaign_estimated_metrics,
    )

    assert int(campaign_metrics["best_ks_percentile"].iat[0]) == int(top_metrics["best_ks_percentile"].iat[0])


def test_campaign_rate_comparison_labels_render_inside_bars() -> None:
    fig = _make_campaign_rate_comparison_figure(
        whole_base_default_sr_pct=10.0,
        campaign_default_sr_pct=12.5,
        top_equal_volume_sr_pct=15.2,
    )
    bar = fig.data[0]
    assert bar.type == "bar"
    assert bar.textposition == "inside"


def test_full_base_campaign_report_chart_values_remain_close(tmp_path) -> None:
    model_score, _, _ = create_simulated_tables(seed=42)
    scored = model_score.copy()
    scored["fs_time"] = pd.to_datetime(scored["fs_time"])
    latest = scored.loc[scored["fs_time"] == scored["fs_time"].max()].copy()
    campaign_clients = latest[["pt_unified_key"]].drop_duplicates().copy()
    output = tmp_path / "full_base_campaign_report.html"

    EvaluateModel(
        output_html_path=output,
        seed=42,
        include_campaign_selection=True,
        campaign_clients=campaign_clients,
    )

    html = output.read_text(encoding="utf-8")
    match = re.search(
        r"const cutoffData = (\[.*?\]);\s*const campaignCutoffData = (\[.*?\]);",
        html,
        flags=re.S,
    )
    assert match is not None

    top_points = json.loads(match.group(1))
    campaign_points = json.loads(match.group(2))
    assert len(top_points) == len(campaign_points) == 100

    max_gain_diff = max(abs(float(t["gain"]) - float(c["gain"])) for t, c in zip(top_points, campaign_points))
    max_sr_diff = max(abs(float(t["sr"]) - float(c["sr"])) for t, c in zip(top_points, campaign_points))

    assert max_gain_diff <= 1.0
    assert max_sr_diff <= 1.0


def test_generated_report_supports_czech_language_output(tmp_path) -> None:
    model_score, _, _ = create_simulated_tables(seed=42)
    campaign_clients = model_score[["pt_unified_key"]].drop_duplicates().head(200).copy()
    output = tmp_path / "report_cs.html"
    EvaluateModel(
        output_html_path=output,
        seed=42,
        language="cs",
        include_campaign_selection=True,
        campaign_clients=campaign_clients,
    )
    html = output.read_text(encoding="utf-8")
    assert '<html lang="cs">' in html
    assert "Vyhodnocení výkonnosti modelu" in html
    assert "Výběrový percentil cut-off" in html
    assert "Porovnání výchozí úspěšnosti" in html
    assert "Graf kumulativního zisku" in html
    assert "Definice:" in html
    assert "Jak číst tento graf" in html
    assert "Každý percentilový koš ukazuje" in html
    assert "Graf zisku odpovídá na to" in html
    assert 'const numberLocale = "cs-CZ"' in html
    assert "toLocaleString(numberLocale)" in html


def test_czech_chart_labels_are_applied_to_figure_builders() -> None:
    metrics = _build_metrics_by_contact_percentile(_performance_fixture())
    chart_text = _resolve_chart_text("cs")
    fig = _make_gain_figure(metrics=metrics, default_percentile=20, chart_text=chart_text)

    assert fig.layout.xaxis.title.text == "Percentil oslovené populace (%)"
    assert fig.layout.yaxis.title.text == "Zisk / podíl (%)"
    trace_names = {trace.name for trace in fig.data}
    assert "Náhoda" in trace_names
    assert "Ideál" in trace_names


def test_evaluatemodel_rejects_unsupported_language(tmp_path) -> None:
    output = tmp_path / "report_bad_lang.html"
    with pytest.raises(ValueError, match="language must be one of: en, cs"):
        EvaluateModel(output_html_path=output, seed=42, language="de")
