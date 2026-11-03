"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np
import pytest

import kkr_constants as C
import kkr_evaluation as ev


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.de(-1.0, 0) == "−1" and ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(float("inf")) == "unendlich" and ev.sci(1e-6, 0) == "1·10⁻⁶"
    assert ev.ms_text(0.0004) == "0,40" and ev.ms_text(0.0129) == "12,9" and ev.ms_text(0.1594) == "159" and ev.ms_text(103.2287) == "103 229"
    assert ev.per_step_text(0.526) == "0,526" and ev.per_step_text(4.853) == "4,85" and ev.per_step_text(47.57) == "47,6" and ev.steps_text(1) == "1 Schritt" and ev.steps_text(33) == "33 Schritte"
    assert ev.dims_for(16) == (4, 8, 16) and ev.dims_for(64) == C.DIMS and ev.dims_for(3) == ()
    assert ev.value_error(7.7, 7.0) == pytest.approx(0.1)


def _row(d, bar, admm, newton_ms, admm_ms, scale=1.0):
    return {"d": d, "n": 4 * d, "rho": 0.03,
            "bar": {t: {"steps": s, "seconds": sec, "error": t / 2, "tol": t} for t, (s, sec) in zip(C.TARGETS, bar)},
            "admm": {t: {"iterations": i, "converged": True, "seconds": scale * sec, "error": t / 2, "rho": 0.03} for t, (i, sec) in zip(C.TARGETS, admm)},
            "newton_step_ms": newton_ms, "admm_step_ms": admm_ms}


def _rows(scale=1.0):
    return [_row(16, [(23, 0.0129), (33, 0.0175), (38, 0.020)], [(113, 0.0168), (475, 0.069), (837, 0.122)], 0.526, 0.146, scale),
            _row(32, [(23, 0.1173), (33, 0.1594), (38, 0.1844)], [(91, 0.0169), (372, 0.0678), (667, 0.1203)], 4.853, 0.18, scale)]


def test_crossing_is_the_first_dimension_where_admm_is_faster():
    rows = _rows()
    assert ev.crossing(rows, 1e-3) == 32 and ev.crossing(rows, 1e-6) == 32 and ev.crossing(rows[:1], 1e-6) is None and ev.crossing(_rows(scale=10.0), 1e-6) is None
    rows[1]["admm"][1e-6]["converged"] = False
    assert ev.crossing(rows, 1e-6) is None                                                       # ein nicht gelöster Lauf zählt nicht als schneller


def test_time_verdict_names_the_crossing_the_factor_and_the_targets():
    s = ev.Settings(dmax=32, target=1e-6)
    rows = _rows()
    v = ev.view(s, rows)
    assert v["crossing"] == 32 and v["factor"] == pytest.approx(0.1594 / 0.0678) and v["crossings"] == {1e-3: 32, 1e-6: 32, 1e-9: 32}
    kind, text = ev.verdict_time(s, v, rows)
    assert kind == "success"
    for part in ("(d = 16, 32)", "braucht die Barriere bei d = 32 bis Wertfehler 1,0·10⁻⁶ 159 ms (33 Newton-Schritte), ADMM mit dünnem A 67,8 ms (372 Schritte)", "**ab d = 32 liegt ADMM vorn**, bei d = 32 um den Faktor 2,4",
                 "Ziel 1,0·10⁻³: ab d = 32; Ziel 1,0·10⁻⁶: ab d = 32; Ziel 1,0·10⁻⁹: ab d = 32"):
        assert part in text, part


def test_time_verdict_without_a_crossing_says_it_lies_to_the_right():
    s = ev.Settings(dmax=32, target=1e-6)
    rows = _rows(scale=10.0)
    kind, text = ev.verdict_time(s, ev.view(s, rows), rows)
    assert kind == "info" and "die Barriere überall vorn (Faktor 0,2 bei d = 32)" in text and "die Kreuzung liegt rechts von d = 32" in text


def test_slopes_need_two_dimensions_from_sixteen():
    sl = ev.slopes(_rows())
    assert sl["newton"] == pytest.approx(np.log(4.853 / 0.526) / np.log(2)) and sl["admm"] == pytest.approx(np.log(0.18 / 0.146) / np.log(2))
    assert ev.slopes(_rows()[:1]) == {"newton": None, "admm": None} and ev.slope([1, 2], [1, 8]) == pytest.approx(3.0) and ev.slope([1], [1]) is None


def test_cost_verdict_names_the_step_times_the_slopes_and_the_counts():
    rows = _rows()
    kind, text = ev.verdict_cost(rows, ev.slopes(rows))
    assert kind == "info"
    for part in ("Ein Newton-Schritt der Barriere kostet bei d = 16 0,526 ms und bei d = 32 4,85 ms, ein ADMM-Schritt mit dünnem A 0,146 und 0,180 ms", "Zwischen d = 16 und d = 32 wächst die Zeit je Newton-Schritt mit der Steigung 3,2 gegen d, die je ADMM-Schritt mit 0,3",
                 "n (d + 1)² gegen n (d + 1) Einträge, Verhältnis d + 1 = 33"):
        assert part in text, part
    assert "mindestens zwei Dimensionen ab d = 16" in ev.verdict_cost(rows[:1], ev.slopes(rows[:1]))[1]


def test_steps_verdict_shows_the_ratio_of_nine_digits_to_three_and_no_accuracy_crossing():
    kind, text = ev.verdict_steps(_rows())
    assert kind == "info"
    for part in ("zwischen 23 und 38 Newton-Schritte (bei d = 32: 23, 33, 38 für Wertfehler 1·10⁻³, 1·10⁻⁶, 1·10⁻⁹)", "ADMM braucht bei d = 32 91, 372, 667 Schritte, von 10⁻³ bis 10⁻⁹ das 7fache",
                 "wächst es von 475 auf 372 bei 10⁻⁶", "bei 10⁻³ nicht billiger als bei 10⁻⁹ (23 gegen 38 Schritte)"):
        assert part in text, part


def _structure(diff=0.0):
    return [{"d": 32, "n": 128, "iterations_dense": 372, "iterations_sparse": 372, "dense_seconds": 0.0644, "sparse_seconds": 0.0688, "max_diff": diff, "dense_step_ms": 0.173, "sparse_step_ms": 0.185, "nnz": 4224,
             "dense_entries": 139392, "converged": True}]


def test_structure_verdict_says_the_gain_comes_from_the_matrix():
    kind, text = ev.verdict_structure(_structure(2e-13))
    assert kind == "info"
    for part in ("bei d = 32 mit dichtem A 0,173 ms je Schritt und mit dünnem A 0,185 ms (0,9-mal)", "bis Wertfehler 1·10⁻⁶ insgesamt 64,4 ms gegen 68,8 ms", "größter Unterschied der Ergebnisse 2,0·10⁻¹³, Schrittzahl 372 und 372",
                 "(4224 von 139392 Einträgen ungleich null)"):
        assert part in text, part


def test_solver_verdict_reads_the_frozen_table():
    kind, text = ev.verdict_solvers()
    assert kind == "info"
    for part in ("Clarabel (Innere Punkte) 10 bis 19 Schritte, SCS (ADMM mit Einbettung) 175 bis 875", "Bei d = 128 liegt Clarabel mit 655 ms noch vor SCS (1 467 ms bei eps 10⁻⁶)", "bei d = 256 braucht Clarabel 103 229 ms und SCS 7 208 ms",
                 "Die Kreuzung der echten Löser liegt bei d = 256"):
        assert part in text, part
    flipped = tuple({**r, "scs6_ms": 9e9} for r in C.SOLVER_TABLE)
    assert "rechts der Tabelle" in ev.verdict_solvers(flipped)[1]
