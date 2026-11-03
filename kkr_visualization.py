"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kkr_constants as C
from kkr_evaluation import sci


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _dtick(fig, ds):
    fig.update_xaxes(type="log", tickvals=list(ds), ticktext=[str(d) for d in ds])


# ----------------------------------------------------------------------------- Experiment 3 (Hauptabbildung)

def build_time(rows: list, target: float, crossing) -> go.Figure:
    ds = [r["d"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds, y=[1e3 * r["bar"][target]["seconds"] for r in rows], mode="lines+markers", name="Barriere (eigene, dicht)", line=dict(color=C.COLORS["barrier"], width=3),
                             hovertemplate="d = %{x}: %{y:.1f} ms<extra>Barriere</extra>"))
    fig.add_trace(go.Scatter(x=ds, y=[1e3 * r["admm"][target]["seconds"] for r in rows], mode="lines+markers", name="ADMM (dünnes A)", line=dict(color=C.COLORS["admm"], width=3),
                             hovertemplate="d = %{x}: %{y:.1f} ms<extra>ADMM</extra>"))
    if crossing is not None:
        fig.add_vline(x=crossing, line=dict(color="#888888", dash="dot"), annotation_text="ADMM ab hier vorn", annotation_position="top")
    _dtick(fig, ds)
    fig.update_xaxes(title_text="Dimension d (n = 4d Punkte)")
    fig.update_yaxes(title_text=f"Zeit bis Wertfehler {sci(target, 0)} (ms)", type="log")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 1

def build_step_cost(rows: list) -> go.Figure:
    ds = [r["d"] for r in rows]
    nw, ad = [r["newton_step_ms"] for r in rows], [r["admm_step_ms"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds, y=nw, mode="lines+markers", name="Newton-Schritt der Barriere", line=dict(color=C.COLORS["barrier"], width=3), hovertemplate="d = %{x}: %{y:.3f} ms<extra></extra>"))
    fig.add_trace(go.Scatter(x=ds, y=ad, mode="lines+markers", name="ADMM-Schritt (dünnes A)", line=dict(color=C.COLORS["admm"], width=3), hovertemplate="d = %{x}: %{y:.3f} ms<extra></extra>"))
    last = rows[-1]
    fig.add_trace(go.Scatter(x=ds, y=[nw[-1] * (d / ds[-1]) ** 3 for d in ds], mode="lines", name="d³", line=dict(color=C.COLORS["ref2"], dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ds, y=[ad[-1] * (d / ds[-1]) ** 2 for d in ds], mode="lines", name="d²", line=dict(color=C.COLORS["ref2"], dash="dash"), hoverinfo="skip"))
    _dtick(fig, ds)
    fig.update_xaxes(title_text="Dimension d")
    fig.update_yaxes(title_text="Zeit je Schritt (ms)", type="log")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 2

def build_steps(rows: list) -> go.Figure:
    ds = [r["d"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Barriere: Newton-Schritte", "ADMM: Schritte"), shared_yaxes=False)
    dash = {C.TARGETS[0]: "dot", C.TARGETS[1]: "dash", C.TARGETS[2]: "solid"}
    for t in C.TARGETS:
        label = f"Wertfehler {sci(t, 0)}"
        fig.add_trace(go.Scatter(x=ds, y=[r["bar"][t]["steps"] for r in rows], mode="lines+markers", name=label, line=dict(color=C.COLORS["barrier"], width=3, dash=dash[t]), legendgroup=label,
                                 hovertemplate="d = %{x}: %{y} Schritte<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ds, y=[r["admm"][t]["iterations"] for r in rows], mode="lines+markers", name=label, line=dict(color=C.COLORS["admm"], width=3, dash=dash[t]), legendgroup=label,
                                 showlegend=False, hovertemplate="d = %{x}: %{y} Schritte<extra></extra>"), row=1, col=2)
    for col in (1, 2):
        fig.update_xaxes(type="log", tickvals=ds, ticktext=[str(d) for d in ds], title_text="Dimension d", row=1, col=col)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 4

def build_structure(rows: list) -> go.Figure:
    ds = [str(r["d"]) for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Zeit je ADMM-Schritt (ms)", f"Zeit bis Wertfehler {sci(C.STRUCTURE_TARGET, 0)} (ms)"))
    fig.add_trace(go.Bar(x=ds, y=[r["dense_step_ms"] for r in rows], name="dichtes A", marker_color=C.COLORS["dense"]), row=1, col=1)
    fig.add_trace(go.Bar(x=ds, y=[r["sparse_step_ms"] for r in rows], name="dünnes A", marker_color=C.COLORS["sparse"]), row=1, col=1)
    fig.add_trace(go.Bar(x=ds, y=[1e3 * r["dense_seconds"] for r in rows], name="dichtes A", marker_color=C.COLORS["dense"], showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=ds, y=[1e3 * r["sparse_seconds"] for r in rows], name="dünnes A", marker_color=C.COLORS["sparse"], showlegend=False), row=1, col=2)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Dimension d")
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 5

def build_solvers(table=C.SOLVER_TABLE) -> go.Figure:
    ds = [r["d"] for r in table]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds, y=[r["clarabel_ms"] for r in table], mode="lines+markers", name="Clarabel (Innere Punkte, Standard)", line=dict(color=C.COLORS["clarabel"], width=3),
                             hovertemplate="d = %{x}: %{y:.1f} ms<extra>Clarabel</extra>"))
    fig.add_trace(go.Scatter(x=ds, y=[r["scs6_ms"] for r in table], mode="lines+markers", name="SCS (ADMM), eps 10⁻⁶", line=dict(color=C.COLORS["scs"], width=3),
                             hovertemplate="d = %{x}: %{y:.1f} ms<extra>SCS</extra>"))
    fig.add_trace(go.Scatter(x=ds, y=[r["scs3_ms"] for r in table], mode="lines+markers", name="SCS (ADMM), eps 10⁻³", line=dict(color=C.COLORS["scs"], width=2, dash="dot"),
                             hovertemplate="d = %{x}: %{y:.1f} ms<extra>SCS</extra>"))
    _dtick(fig, ds)
    fig.update_xaxes(title_text="Dimension d (n = 4d Punkte)")
    fig.update_yaxes(title_text="Löserzeit (ms)", type="log")
    return _base(fig)
