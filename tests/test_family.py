"""Die Familie der Kugelprogramme (kkr_family): Handinstanz, dichte gleich dünne Fassung, Zählung der Operationen, Wolke."""

import numpy as np
import pytest

import kegel_kern as K
import kkr_admm as AD
import kkr_family as FAM

LINE = np.array([[-2.0], [-1.0], [1.0], [2.0]])        # vier Punkte auf der Geraden: Mittelpunkt 0, Radius 2


def test_hand_instance_four_points_on_a_line():
    """d = 1, n = 4: Variablen (c, R), 4 Lorentz-Blöcke der Größe 2, 8 Zeilen; R* = 2 (der Abstand der äußeren Punkte vom Mittelpunkt 0)."""
    prob = FAM.dense_problem(LINE)
    assert prob.A.shape == (8, 2) and prob.cones == (("q", 2),) * 4
    assert prob.A.tolist() == [[0, -1], [1, 0]] * 4 and prob.b.tolist() == [0, -2, 0, -1, 0, 1, 0, 2] and prob.c.tolist() == [0, 1]
    res = K.solve_barrier(prob, tol=1e-12)
    assert res.value == pytest.approx(2.0, abs=1e-9) and res.x[0] == pytest.approx(0.0, abs=1e-8)


def test_the_sparse_matrix_equals_the_dense_one_entry_by_entry():
    for d in (1, 3, 8):
        pts = FAM.cloud(d, seed=3)
        dense, sparse = FAM.dense_problem(pts), FAM.sparse_problem(pts)
        assert np.array_equal(sparse.A.toarray(), dense.A) and np.array_equal(sparse.b, dense.b) and np.array_equal(sparse.c, dense.c)
        assert sparse.A.nnz == len(pts) * (d + 1) and sparse.groups == [("q", d + 1, len(pts))] and AD.spec_of(dense) == sparse.groups


def test_operation_counts_by_hand():
    """n = 4, d = 1: Newton n (d + 1)² = 16, ADMM n (d + 1) = 8; das Verhältnis ist d + 1."""
    assert FAM.newton_operations(4, 1) == 16 and FAM.admm_operations(4, 1) == 8
    for d in (2, 16, 64):
        n = FAM.default_n(d)
        assert FAM.newton_operations(n, d) / FAM.admm_operations(n, d) == d + 1
    assert FAM.sparse_problem(LINE).A.nnz == FAM.admm_operations(4, 1)


def test_admm_on_the_hand_instance_dense_and_sparse():
    for prob in (FAM.dense_problem(LINE), FAM.sparse_problem(LINE)):
        r = AD.admm(prob, rho=0.1, tol=1e-9, max_iter=3000)
        assert r.converged and r.value == pytest.approx(2.0, abs=1e-7) and r.x[0] == pytest.approx(0.0, abs=1e-6)


def test_dense_and_sparse_admm_produce_the_same_iterates():
    pts = FAM.cloud(8, seed=5)
    a = AD.admm(FAM.dense_problem(pts), rho=0.03, tol=0.0, max_iter=40)
    b = AD.admm(FAM.sparse_problem(pts), rho=0.03, tol=0.0, max_iter=40)
    assert np.abs(a.x - b.x).max() < 1e-12 and np.abs(a.u - b.u).max() < 1e-12 and a.primal == pytest.approx(b.primal, rel=1e-9)


def test_cloud_shape_mean_determinism_and_spread():
    pts = FAM.cloud(16, seed=1)
    assert pts.shape == (64, 16) and np.abs(pts.mean(axis=0)).max() < 1e-12 and np.array_equal(pts, FAM.cloud(16, seed=1)) and not np.array_equal(pts, FAM.cloud(16, seed=2))
    assert FAM.default_n(1) == 8 and FAM.default_n(2) == 8 and FAM.default_n(4) == 16 and FAM.default_n(64) == 256
    big = FAM.cloud(32, n=4000, seed=0)
    assert big.std(axis=0)[0] == pytest.approx(1.0, rel=0.08) and big.std(axis=0)[-1] == pytest.approx(3.0, rel=0.08)


def test_the_optimal_radius_grows_with_the_dimension():
    """Band: R* wächst mit √d bei normalverteilten Punkten (die Standardabweichungen wachsen von 1 auf 3)."""
    r = {d: K.solve_barrier(FAM.dense_problem(FAM.cloud(d, seed=35)), tol=1e-9).value for d in (4, 16, 32)}
    assert r[4] < r[16] < r[32] and 1.5 <= r[16] / r[4] <= 2.8 and 1.2 <= r[32] / r[16] <= 1.9


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        FAM.cloud(0)
    with pytest.raises(ValueError):
        FAM.cloud(3, n=1)
