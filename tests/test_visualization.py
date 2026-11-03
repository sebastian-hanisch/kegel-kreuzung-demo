"""Abbildungen: Achsen gesperrt, Spuren stimmen mit den Zahlen überein."""

import pytest

import kkr_constants as C
import kkr_evaluation as ev
import kkr_visualization as V

S = ev.Settings(dmax=16, seed=7)


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(ax.get("fixedrange") is True for ax in axes)


@pytest.fixture(scope="module")
def rows():
    return ev.scaling_rows(S)


def test_time_chart_has_two_log_lines_and_the_crossing_marker(rows):
    fig = V.build_time(rows, 1e-6, 16)
    assert _locked(fig) and [t.name for t in fig.data] == ["Barriere (eigene, dicht)", "ADMM (dünnes A)"] and fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    assert list(fig.data[0].x) == [4, 8, 16] and list(fig.data[0].y) == [1e3 * r["bar"][1e-6]["seconds"] for r in rows] and len(fig.layout.shapes) == 1 and fig.layout.shapes[0].x0 == 16
    assert len(V.build_time(rows, 1e-6, None).layout.shapes) == 0


def test_step_cost_chart_has_two_measured_and_two_reference_lines(rows):
    fig = V.build_step_cost(rows)
    assert _locked(fig) and [t.name for t in fig.data] == ["Newton-Schritt der Barriere", "ADMM-Schritt (dünnes A)", "d³", "d²"]
    d3 = fig.data[2]
    assert d3.y[-1] == pytest.approx(fig.data[0].y[-1]) and d3.y[0] == pytest.approx(fig.data[0].y[-1] * (4 / 16) ** 3)


def test_steps_chart_has_a_panel_per_method_and_three_targets(rows):
    fig = V.build_steps(rows)
    assert _locked(fig) and len(fig.data) == 6 and len(fig.layout.annotations) == 2 and sum(1 for t in fig.data if t.showlegend is not False) == 3
    assert list(fig.data[0].y) == [r["bar"][1e-3]["steps"] for r in rows] and list(fig.data[1].y) == [r["admm"][1e-3]["iterations"] for r in rows]


def test_structure_chart_has_dense_and_sparse_bars(rows):
    st = ev.structure_rows(S, rows)
    fig = V.build_structure(st)
    assert _locked(fig) and len(fig.data) == 4 and fig.layout.barmode == "group" and list(fig.data[0].y) == [r["dense_step_ms"] for r in st] and list(fig.data[3].x) == ["4", "8", "16"]


def test_solver_chart_has_the_frozen_series():
    fig = V.build_solvers()
    assert _locked(fig) and len(fig.data) == 3 and list(fig.data[0].x) == [r["d"] for r in C.SOLVER_TABLE] and list(fig.data[0].y) == [r["clarabel_ms"] for r in C.SOLVER_TABLE] and fig.layout.yaxis.type == "log"
