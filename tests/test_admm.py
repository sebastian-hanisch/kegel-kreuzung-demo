"""ADMM (kkr_admm, aus Stück 9 übernommen): Handbeispiele, Fixpunkt, Fehlerfälle, dünn besetztes A."""

import numpy as np
import pytest

import kegel_kern as K
import kkr_admm as AD
import kkr_family as FAM


def test_the_lorentz_hand_program_step_by_step(tiny_cones):
    """min t mit (t; 3, 4) im Lorentz-Kegel, ρ = 1, Start x = 0, u = 0 (von Hand, wie Stück 9): Schritt 1: x = 1,5, s = (3,25; 1,95; 2,6), u = (1,75; −1,05; −1,4), Residuen √6,125 und 0,75; Schritt 2: x = 4."""
    prob, opt = tiny_cones["soc"]
    r1 = AD.admm(prob, rho=1.0, max_iter=1, tol=0.0)
    assert r1.x == pytest.approx([1.5]) and r1.s == pytest.approx([3.25, 1.95, 2.6]) and r1.u == pytest.approx([1.75, -1.05, -1.4])
    assert r1.primal == pytest.approx(np.sqrt(6.125)) and r1.dual == pytest.approx(0.75)
    assert AD.admm(prob, rho=1.0, max_iter=2, tol=0.0).x == pytest.approx([4.0])
    r = AD.admm(prob, rho=1.0, max_iter=400, tol=1e-12)
    assert r.converged and r.x == pytest.approx([5.0], abs=1e-9) and r.u == pytest.approx([1.0, -0.6, -0.8], abs=1e-9) and r.value == pytest.approx(opt, abs=1e-9)


def test_the_orthant_hand_program(tiny_cones):
    prob, opt = tiny_cones["lp"]
    r2 = AD.admm(prob, rho=1.0, max_iter=2, tol=0.0)
    assert r2.x == pytest.approx([1.0]) and r2.s == pytest.approx([0.0]) and r2.u == pytest.approx([1.0]) and AD.admm(prob, rho=1.0, tol=1e-9).value == pytest.approx(opt)


def test_the_result_matches_the_barrier_on_a_family_member_for_dense_and_sparse_a():
    pts = FAM.cloud(8, seed=7)
    dense, sparse = FAM.dense_problem(pts), FAM.sparse_problem(pts)
    ref = K.solve_barrier(dense, tol=1e-12).value
    for prob in (dense, sparse):
        r = AD.admm(prob, rho=0.03, tol=1e-9, max_iter=8000, ref_value=ref)
        assert r.converged and abs(r.value - ref) <= 1e-9 * ref
    r = AD.admm(sparse, rho=0.03, tol=1e-8, max_iter=8000)
    assert r.converged and r.value == pytest.approx(ref, rel=1e-6)


def test_the_reference_stop_counts_the_value_error_and_the_primal_residual():
    pts = FAM.cloud(4, seed=2)
    prob = FAM.sparse_problem(pts)
    ref = K.solve_barrier(FAM.dense_problem(pts), tol=1e-12).value
    r = AD.admm(prob, rho=0.1, tol=1e-6, max_iter=5000, ref_value=ref)
    assert r.converged and abs(r.value - ref) <= 1e-6 * ref and r.primal <= 1e-6 * (1 + np.linalg.norm(prob.b))


def test_rank_deficient_a_and_bad_rho_are_rejected(tiny_cones):
    prob, _ = tiny_cones["soc"]
    bad = K.ConeProblem(np.array([1.0, 1.0]), np.array([[-1.0, -1.0], [0.0, 0.0], [0.0, 0.0]]), np.array([0.0, 3.0, 4.0]), (("q", 3),), np.array([6.0, 0.0]))
    with pytest.raises(ValueError, match="Spaltenrang"):
        AD.admm(bad)
    for rho in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            AD.admm(prob, rho=rho)


def test_a_sparse_matrix_without_full_column_rank_is_rejected_too():
    import scipy.sparse as sp

    A = sp.csr_matrix(np.array([[-1.0, -1.0], [0.0, 0.0], [0.0, 0.0]]))
    prob = FAM.SparseProblem(A, np.array([0.0, 3.0, 4.0]), np.array([1.0, 1.0]), [("q", 3, 1)])
    with pytest.raises(ValueError, match="Spaltenrang"):
        AD.admm(prob)


def test_the_balancing_changes_rho_on_a_sparse_program():
    prob = FAM.sparse_problem(FAM.cloud(8, seed=7))
    r = AD.admm(prob, rho=100.0, tol=1e-6, max_iter=600, adaptive=True)
    assert r.rho_changes > 0 and r.rho < 100.0 and len(r.history["value"]) == r.iterations


def test_spec_of_groups_equal_cones():
    assert AD.spec_of(FAM.dense_problem(FAM.cloud(3, seed=1))) == [("q", 4, 12)] and AD.spec_of(FAM.sparse_problem(FAM.cloud(3, seed=1))) == [("q", 4, 12)]
