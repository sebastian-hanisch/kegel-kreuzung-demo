"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Zeiten werden nicht auf Millisekunden getestet, nur ihre Struktur (Steigungen, Reihenfolge, Kreuzung in einem Band). Die Schrittzahlen sind deterministisch bis auf die Rundung der BLAS-Routinen: Bänder mit etwa 10 % Luft."""

import hashlib
from pathlib import Path

import pytest

import kkr_constants as C
import kkr_evaluation as ev

S = ev.Settings(dmax=64)


@pytest.fixture(scope="module")
def rows():
    return ev.scaling_rows(S)


@pytest.fixture(scope="module")
def by(rows):
    return {r["d"]: r for r in rows}


@pytest.fixture(scope="module")
def structure(rows):
    return {r["d"]: r for r in ev.structure_rows(S, rows)}


def test_the_barrier_needs_almost_the_same_steps_for_every_dimension(rows, by):
    """README: Newton-Schritte der Barriere bis Wertfehler 10⁻³ / 10⁻⁶ / 10⁻⁹: d = 4: 18 / 24 / 28; d = 16: 23 / 33 / 38; d = 64: 24 / 37 / 44; über alle d zwischen 18 und 44 (Band 15 bis 50);
    von 10⁻³ bis 10⁻⁹ nur das 1,8fache bei d = 64."""
    steps = [r["bar"][t]["steps"] for r in rows for t in C.TARGETS]
    assert 15 <= min(steps) and max(steps) <= 50
    assert [by[4]["bar"][t]["steps"] for t in C.TARGETS] == pytest.approx([18, 24, 28], abs=4) and [by[64]["bar"][t]["steps"] for t in C.TARGETS] == pytest.approx([24, 37, 44], abs=5)
    assert 1.5 <= by[64]["bar"][1e-9]["steps"] / by[64]["bar"][1e-3]["steps"] <= 2.4
    assert all(r["bar"][t]["error"] <= t for r in rows for t in C.TARGETS)


def test_admm_needs_about_ten_times_the_steps_for_nine_digits_instead_of_three(rows, by):
    """README: ADMM-Schritte bis 10⁻³ / 10⁻⁶ / 10⁻⁹ bei d = 64: 138 / 789 / 1 441 (das 10fache); bei 10⁻⁶ wächst es von 206 (d = 4) auf 789 (d = 64)."""
    a = by[64]["admm"]
    assert all(a[t]["converged"] and a[t]["error"] <= t * 1.0001 for t in C.TARGETS)
    assert 6.0 <= a[1e-9]["iterations"] / a[1e-3]["iterations"] <= 14.0
    assert [a[t]["iterations"] for t in C.TARGETS] == pytest.approx([138, 789, 1441], rel=0.2)
    assert 170 <= by[4]["admm"][1e-6]["iterations"] <= 250 and 2.5 <= a[1e-6]["iterations"] / by[4]["admm"][1e-6]["iterations"] <= 5.0
    its = [r["admm"][t]["iterations"] for r in rows for t in C.TARGETS]
    assert 55 <= min(its) <= 95 and 1200 <= max(its) <= 1700                                                  # README: 70 bis 1 440 Schritte
    assert all(r["rho"] in C.RHO_GRID for r in rows) and by[64]["rho"] <= by[4]["rho"]                       # das beste ρ wandert mit der Skala nach unten


def test_the_newton_step_grows_with_slope_three_the_admm_step_much_less(rows, by):
    """README: Zeit je Newton-Schritt von 0,21 ms (d = 4) auf 47,6 ms (d = 64), Steigung (d ≥ 16) 3,2 (Band 2,5 bis 3,8); je ADMM-Schritt von 0,13 auf 0,32 ms, Steigung 0,6 (Band unter 1,2):
    gezählt n (d + 1)² gegen n (d + 1), Verhältnis d + 1."""
    sl = ev.slopes(rows)
    assert 2.5 <= sl["newton"] <= 3.8 and sl["admm"] < 1.2 and sl["newton"] - sl["admm"] > 1.8
    steps = [r["newton_step_ms"] for r in rows]
    assert steps == sorted(steps) and 5.0 <= by[64]["newton_step_ms"] / by[32]["newton_step_ms"] <= 14.0
    assert all(r["newton_ops"] == r["n"] * (r["d"] + 1) ** 2 and r["admm_ops"] == r["n"] * (r["d"] + 1) and r["newton_ops"] // r["admm_ops"] == r["d"] + 1 for r in rows)
    assert [r["n"] for r in rows] == [16, 32, 64, 128, 256] and [r["variables"] for r in rows] == [5, 9, 17, 33, 65] and by[64]["rows"] == 256 * 65 and by[64]["nnz"] == 256 * 65


def test_the_crossing_is_in_the_size_not_in_the_accuracy(rows, by):
    """README: die eigene dichte Barriere liegt bei d = 4 und 8 vorn, ADMM ab d = 16 bis 32 (bei jedem der drei Ziele), bei d = 64 um den Faktor 3 bis 30 (Band 1,5 bis 40); Zeiten schwanken: weite Bänder."""
    for t in C.TARGETS:
        assert ev.crossing(rows, t) in (16, 32), t
        assert by[4]["bar"][t]["seconds"] < by[4]["admm"][t]["seconds"] and by[64]["bar"][t]["seconds"] > 1.5 * by[64]["admm"][t]["seconds"]
        assert 1.5 <= by[64]["bar"][t]["seconds"] / by[64]["admm"][t]["seconds"] <= 40.0
    v = ev.view(S, rows)
    assert v["crossing"] in (16, 32) and v["last"]["d"] == 64 and 1.5 <= v["factor"] <= 40.0
    small = ev.view(ev.Settings(dmax=16), rows[:3])
    assert small["crossing"] in (None, 16) and small["slopes"]["newton"] is None


def test_the_same_admm_is_much_cheaper_with_a_sparse_matrix(structure):
    """README (Wertfehler 10⁻⁶): dichtes A kostet bei d = 64 2,4 ms je Schritt, dünnes A 0,39 ms (Band: Faktor 3 bis 12); bei d = 4 ist das dichte A noch billiger; Schritte und Ergebnisse gleich."""
    assert all(r["iterations_dense"] == r["iterations_sparse"] and r["max_diff"] < 1e-9 and r["converged"] for r in structure.values())
    assert 3.0 <= structure[64]["dense_step_ms"] / structure[64]["sparse_step_ms"] <= 12.0 and structure[4]["dense_step_ms"] < structure[4]["sparse_step_ms"]
    assert 0.6 <= structure[32]["dense_step_ms"] / structure[32]["sparse_step_ms"] <= 1.8                  # bei d = 32 etwa gleich (README)
    assert structure[64]["nnz"] == 16640 and structure[64]["dense_entries"] == 16640 * 65


def test_the_frozen_solver_table_has_the_claimed_structure():
    """README: Clarabel braucht 10 bis 19 Schritte, SCS 175 bis 875; bis d = 128 liegt Clarabel vorn (bei eps 10⁻⁶), bei d = 256 SCS mit 7,2 s gegen 103 s (Faktor 14)."""
    t = C.SOLVER_TABLE
    assert [r["d"] for r in t] == [8, 16, 32, 64, 128, 256] and all(r["n"] == 4 * r["d"] for r in t)
    assert min(r["clarabel_steps"] for r in t) == 10 and max(r["clarabel_steps"] for r in t) == 19
    assert min(r["scs3_steps"] for r in t) == 175 and max(r["scs6_steps"] for r in t) == 875
    assert [r["d"] for r in t if r["scs6_ms"] < r["clarabel_ms"]] == [256]
    last = t[-1]
    assert 12 <= last["clarabel_ms"] / last["scs6_ms"] <= 16 and all(r["scs3_err"] < 1e-4 and r["scs6_err"] < 1e-7 + 1e-9 for r in t)
    assert ev.verdict_solvers()[0] == "info" and "bei d = 256" in ev.verdict_solvers()[1]


def test_the_kernel_is_byte_identical_in_the_series():
    """README: kegel_kern.py ist wortgleich mit Stück 2 bis 9: Prüfsumme wie im Schwester-Repo, sofern es nebenan liegt."""
    here = Path(__file__).resolve().parent.parent / "kegel_kern.py"
    sibling = here.parent.parent / "kegel-admm-demo" / "kegel_kern.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    assert hashlib.sha256(here.read_bytes()).hexdigest() == hashlib.sha256(sibling.read_bytes()).hexdigest()
