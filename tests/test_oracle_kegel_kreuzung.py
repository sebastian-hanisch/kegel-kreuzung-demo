"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): die Familie und die Schrittzahlen gegen cvxpy.

- cvxpy mit Clarabel (Innere Punkte, dünn besetzt) löst dieselbe Kugel mit `cp.SOC`: R* auf 10⁻⁶ für d = 4, 16, 32; die Barriere des Kerns und ADMM mit dünnem A müssen es treffen
- die Schrittzahlen als Struktur: Clarabel unter 25, SCS (ADMM) über 100; die eingefrorene Tabelle der echten Löser stimmt im Aufbau mit einem frischen Lauf bei d = 16 überein

Fehlt cvxpy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import numpy as np
import pytest

import kegel_kern as K
import kkr_admm as AD
import kkr_constants as C
import kkr_family as FAM


def _cp():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    return cp


def _build(cp, pts):
    n, d = pts.shape
    c, R = cp.Variable(d), cp.Variable()
    return cp.Problem(cp.Minimize(R), [cp.SOC(R * np.ones(n), pts - np.ones((n, 1)) @ cp.reshape(c, (1, d), order="C"), axis=1)]), R


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
@pytest.mark.parametrize("d", [4, 16, 32])
def test_the_radius_equals_the_cvxpy_radius(d):
    cp = _cp()
    pts = FAM.cloud(d, seed=35)
    prob, R = _build(cp, pts)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    ref = float(R.value)
    assert K.solve_barrier(FAM.dense_problem(pts), tol=1e-10).value == pytest.approx(ref, rel=1e-6)
    r = AD.admm(FAM.sparse_problem(pts), rho=0.03, tol=1e-8, max_iter=20000)
    assert r.converged and r.value == pytest.approx(ref, rel=1e-5)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_step_counts_of_the_real_solvers_have_the_claimed_structure():
    cp = _cp()
    if "SCS" not in cp.installed_solvers():
        pytest.skip("SCS fehlt")
    row = next(r for r in C.SOLVER_TABLE if r["d"] == 16)
    pts = FAM.cloud(16, seed=35)
    prob, R = _build(cp, pts)
    prob.solve(solver=cp.CLARABEL)
    clarabel = prob.solver_stats.num_iters
    prob, R = _build(cp, pts)
    prob.solve(solver=cp.SCS, eps=1e-6, max_iters=200000)
    scs = prob.solver_stats.num_iters
    assert clarabel < 25 and scs > 100 and abs(clarabel - row["clarabel_steps"]) <= 4 and 0.5 * row["scs6_steps"] <= scs <= 2.0 * row["scs6_steps"]
