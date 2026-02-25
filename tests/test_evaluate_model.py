from __future__ import annotations

import pandas as pd

from evaluate_model import EvaluateModel, _build_metrics_by_contact_percentile, _make_figure


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
    assert "Success Rate" in trace_names
    assert "Cumulative Non-Success Share" not in trace_names
    assert "KS" not in trace_names
    success_trace = next(trace for trace in figure.data if trace.name == "Success Rate")
    assert success_trace.type == "bar"

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
    assert "How to read these charts" in html
