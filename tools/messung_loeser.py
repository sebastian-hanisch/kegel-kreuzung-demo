"""Experiment 5 (eingefrorene Tabelle in kkr_constants.SOLVER_TABLE) neu messen: Clarabel (Innere Punkte, dünn besetzt) gegen SCS (ADMM mit Einbettung) auf der Kugel von n = 4d Punkten in d Dimensionen.

    pip install -r requirements-dev.txt
    python tools/messung_loeser.py            # d = 8, 16, 32, 64, 128 (bei d = 256 braucht Clarabel über eine Minute)
    python tools/messung_loeser.py 8,16,256   # eigene Dimensionen

Gemeldet wird die Löserzeit aus cvxpy (`solver_stats.solve_time`, ms), die Schrittzahl und der Wertfehler gegen Clarabel mit Toleranz 1e-10. Die Zeiten hängen vom Rechner ab; die eingefrorene Tabelle
stammt vom Entwicklungsrechner (2026-11-02, cvxpy 1.9.3). Nicht Teil der App und nicht Teil der Laufzeit-Abhängigkeiten."""

import sys
import warnings
from pathlib import Path

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kkr_family as FAM  # noqa: E402

warnings.filterwarnings("ignore")


def build(pts):
    n, d = pts.shape
    c, R = cp.Variable(d), cp.Variable()
    return cp.Problem(cp.Minimize(R), [cp.SOC(R * np.ones(n), pts - np.ones((n, 1)) @ cp.reshape(c, (1, d), order="C"), axis=1)]), R


def main(dims):
    for d in dims:
        pts = FAM.cloud(d, seed=35)
        prob, R = build(pts)
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
        ref = float(R.value)
        print(f"== d = {d}, n = {len(pts)}, R* = {ref:.9f}", flush=True)
        for label, kw in (("Clarabel Standard (1e-8)", dict(solver=cp.CLARABEL)), ("SCS eps 1e-3", dict(solver=cp.SCS, eps=1e-3)), ("SCS eps 1e-6", dict(solver=cp.SCS, eps=1e-6, max_iters=200000))):
            prob, R = build(pts)
            prob.solve(**kw)
            st = prob.solver_stats
            print(f"  {label:26s}: {st.solve_time * 1e3:9.1f} ms, {st.num_iters} Schritte, Wertfehler {abs(float(R.value) - ref) / ref:.1e}", flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [8, 16, 32, 64, 128])
