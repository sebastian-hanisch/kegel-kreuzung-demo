"""Randfälle, an denen der Vergleich erfahrungsgemäß bricht (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import kegel_kern as K
import kkr_admm as AD
import kkr_constants as C
import kkr_evaluation as ev
import kkr_family as FAM


def test_dimension_one_and_two_work():
    for d in (1, 2):
        row = ev.measure_dimension(d, seed=3, targets=(1e-3, 1e-6))
        assert row["n"] == 8 and row["variables"] == d + 1 and row["rows"] == 8 * (d + 1) and all(row["admm"][t]["converged"] for t in (1e-3, 1e-6))
        assert row["bar"][1e-6]["error"] <= 1e-6 and row["newton_ops"] == 8 * (d + 1) ** 2


def test_the_smallest_number_of_points_still_gives_a_program():
    """n = d + 1 Punkte (ein Simplex): R* ist der Umkreisradius, Barriere und ADMM stimmen überein."""
    pts = FAM.cloud(5, n=6, seed=2)
    ref = K.solve_barrier(FAM.dense_problem(pts), tol=1e-11).value
    r = AD.admm(FAM.sparse_problem(pts), rho=0.1, tol=1e-8, max_iter=20000, ref_value=ref)
    assert r.converged and ref > 0


def test_duplicate_points_do_not_change_the_radius():
    pts = FAM.cloud(4, n=16, seed=4)
    twice = np.concatenate([pts, pts])
    a = K.solve_barrier(FAM.dense_problem(pts), tol=1e-11).value
    b = K.solve_barrier(FAM.dense_problem(twice), tol=1e-11).value
    assert b == pytest.approx(a, rel=1e-8)
    r = AD.admm(FAM.sparse_problem(twice), rho=0.1, tol=1e-8, max_iter=20000, ref_value=a)
    assert r.converged


def test_the_balancing_handles_data_of_very_different_scale_on_the_family():
    for f in (1e-3, 1e3):
        pts = f * FAM.cloud(8, seed=5)
        ref = K.solve_barrier(FAM.dense_problem(pts), tol=1e-11).value
        r = AD.admm(FAM.sparse_problem(pts), rho=1.0, tol=1e-6, max_iter=20000, adaptive=True, ref_value=ref)
        assert r.converged and r.rho_changes >= 1, f


def test_a_target_below_the_rounding_is_not_reached():
    pts = FAM.cloud(4, seed=2)
    ref = K.solve_barrier(FAM.dense_problem(pts), tol=1e-12).value
    r = AD.admm(FAM.sparse_problem(pts), rho=0.1, tol=1e-15, max_iter=3000, ref_value=ref)
    assert not r.converged and r.iterations == 3000


def test_the_control_ranges_keep_two_dimensions_for_the_slopes():
    assert min(C.DMAX_CHOICES) >= 2 * C.SLOPE_FROM // 2 and len([d for d in C.DIMS if d >= C.SLOPE_FROM and d <= min(C.DMAX_CHOICES)]) >= 1
    assert len(ev.dims_for(32)) == 4 and ev.dims_for(C.DMAX_CHOICES[-1])[-1] == C.DMAX_CHOICES[-1]


def test_another_seed_keeps_the_crossing_in_the_same_band():
    rows = ev.scaling_rows(ev.Settings(dmax=32, seed=1))
    assert ev.crossing(rows, 1e-6) in (16, 32) and ev.crossing(rows, 1e-9) in (16, 32)
    assert all(r["admm"][t]["converged"] for r in rows for t in C.TARGETS)


def test_the_view_works_at_the_smallest_range():
    s = ev.Settings(dmax=16, seed=7)
    rows = ev.scaling_rows(s)
    v = ev.view(s, rows)
    assert [r["d"] for r in rows] == [4, 8, 16] and v["last"]["d"] == 16 and ev.verdict_time(s, v, rows)[0] in {"success", "info"}
    assert ev.verdict_cost(rows, v["slopes"])[0] == "info" and ev.verdict_steps(rows)[0] == "info"
