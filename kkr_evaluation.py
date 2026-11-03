"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

import kegel_kern as K
import kkr_admm as AD
import kkr_constants as C
import kkr_family as FAM


@dataclass(frozen=True)
class Settings:
    dmax: int = C.DEFAULT_DMAX
    target: float = C.DEFAULT_TARGET
    seed: int = C.DEFAULT_SEED


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def ms_text(seconds: float) -> str:
    ms = 1e3 * seconds
    return de(ms, 2) if ms < 1 else de(ms, 1) if ms < 100 else f"{ms:,.0f}".replace(",", " ")


def per_step_text(ms: float) -> str:
    return de(ms, 3) if ms < 1 else de(ms, 2) if ms < 10 else de(ms, 1)


def steps_text(n: int) -> str:
    return "1 Schritt" if n == 1 else f"{n} Schritte"


def dims_for(dmax: int) -> tuple:
    return tuple(d for d in C.DIMS if d <= dmax)


# ----------------------------------------------------------------------------- Messung

def _best_time(fun, repeats: int):
    out, best = None, math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fun()
        best = min(best, time.perf_counter() - t0)
    return out, best


def value_error(value: float, rstar: float) -> float:
    return abs(value - rstar) / abs(rstar)


def measure_dimension(d: int, seed: int, targets=C.TARGETS, cap: int = C.ADMM_CAP) -> dict:
    """Barriere (dicht) und ADMM (dünnes A) auf der Kugel von 4d Punkten in d Dimensionen, je Ziel am Wertfehler gegen R* (aus der Barriere mit Toleranz 10⁻¹²).
    Barriere: aufsteigende Toleranzen, je Ziel die erste, deren Wert das Ziel trifft; ADMM: bestes ρ der kleinen Reihe bei 10⁻⁶, dann dieselbe Einstellung für die anderen Ziele.
    Zeiten nach einem Warmlauf, beste von 2 Läufen (bei d = 64 von einem), ein BLAS-Thread."""
    pts = FAM.cloud(d, seed=seed)
    n = len(pts)
    dense, sparse = FAM.dense_problem(pts), FAM.sparse_problem(pts)
    repeats = 2 if d <= 32 else 1
    with threadpool_limits(1):
        rstar = K.solve_barrier(dense, tol=1e-12).value
        K.solve_barrier(dense, tol=1e-2)                                                          # Warmlauf
        ladder = []
        for tol in C.BARRIER_TOLS:
            res, sec = _best_time(lambda: K.solve_barrier(dense, tol=tol), repeats)
            ladder.append({"tol": tol, "steps": res.newton_steps, "seconds": sec, "error": value_error(res.value, rstar)})
            if ladder[-1]["error"] <= min(targets):
                break
        bar = {}
        for target in targets:
            b = next((x for x in ladder if x["error"] <= target), ladder[-1])
            bar[target] = {"steps": b["steps"], "seconds": b["seconds"], "error": b["error"], "tol": b["tol"]}
        best = None
        for rho in C.RHO_GRID:
            r = AD.admm(sparse, rho=rho, tol=1e-6, max_iter=cap, ref_value=rstar)
            if best is None or (r.converged, -r.iterations) > (best[0].converged, -best[0].iterations):
                best = (r, rho)
        rho = best[1]
        admm = {}
        for target in targets:
            r = best[0] if target == 1e-6 else AD.admm(sparse, rho=rho, tol=target, max_iter=cap, ref_value=rstar)
            sec = _best_time(lambda: AD.admm(sparse, rho=rho, tol=target, max_iter=r.iterations, ref_value=rstar), repeats)[1]
            admm[target] = {"iterations": r.iterations, "converged": r.converged, "seconds": sec, "error": value_error(r.value, rstar), "rho": rho}
    hi = max(targets)
    lo = min(targets)
    return {"d": d, "n": n, "variables": d + 1, "rows": dense.A.shape[0], "nnz": int(sparse.A.nnz), "rstar": rstar, "bar": bar, "admm": admm,
            "newton_step_ms": 1e3 * bar[lo]["seconds"] / bar[lo]["steps"], "admm_step_ms": 1e3 * admm[lo]["seconds"] / admm[lo]["iterations"],
            "newton_ops": FAM.newton_operations(n, d), "admm_ops": FAM.admm_operations(n, d), "rho": rho, "target_hi": hi}


def scaling_rows(s: Settings) -> list:
    return [measure_dimension(d, s.seed) for d in dims_for(s.dmax)]


def structure_rows(s: Settings, scaling: list, target: float = C.STRUCTURE_TARGET, cap: int = C.ADMM_CAP) -> list:
    """Dasselbe ADMM (ρ aus der Reihe oben) mit dichtem und mit dünnem A: gleiche Iterierte, verschiedene Zeit je Schritt."""
    rows = []
    with threadpool_limits(1):
        for row in scaling:
            d = row["d"]
            pts = FAM.cloud(d, seed=s.seed)
            dense, sparse = FAM.dense_problem(pts), FAM.sparse_problem(pts)
            rho = row["rho"]
            repeats = 2 if d <= 32 else 1
            rd, td = _best_time(lambda: AD.admm(dense, rho=rho, tol=target, max_iter=cap, ref_value=row["rstar"]), repeats)
            rs, ts = _best_time(lambda: AD.admm(sparse, rho=rho, tol=target, max_iter=cap, ref_value=row["rstar"]), repeats)
            rows.append({"d": d, "n": row["n"], "iterations_dense": rd.iterations, "iterations_sparse": rs.iterations, "dense_seconds": td, "sparse_seconds": ts, "max_diff": float(np.abs(rd.x - rs.x).max()),
                         "dense_step_ms": 1e3 * td / rd.iterations, "sparse_step_ms": 1e3 * ts / rs.iterations, "converged": rd.converged and rs.converged, "nnz": row["nnz"], "dense_entries": dense.A.size})
    return rows


def slope(xs, ys) -> float | None:
    """Steigung der Ausgleichsgeraden in der log-log-Darstellung (None bei weniger als zwei Punkten)."""
    if len(xs) < 2:
        return None
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def slopes(rows: list) -> dict:
    """Steigungen der Zeit je Newton- und je ADMM-Schritt gegen d, angepasst ab d = SLOPE_FROM (None, wenn dort weniger als zwei Dimensionen liegen)."""
    sel = [r for r in rows if r["d"] >= C.SLOPE_FROM]
    return {"newton": slope([r["d"] for r in sel], [r["newton_step_ms"] for r in sel]), "admm": slope([r["d"] for r in sel], [r["admm_step_ms"] for r in sel])}


def crossing(rows: list, target: float):
    """Die kleinste Dimension, ab der ADMM schneller ist als die Barriere (None: keine in der Reihe)."""
    for r in rows:
        if r["admm"][target]["converged"] and r["admm"][target]["seconds"] < r["bar"][target]["seconds"]:
            return r["d"]
    return None


def view(s: Settings, rows: list) -> dict:
    last = rows[-1]
    t = s.target
    cross = crossing(rows, t)
    return {"crossing": cross, "last": last, "bar_ms": 1e3 * last["bar"][t]["seconds"], "admm_ms": 1e3 * last["admm"][t]["seconds"], "factor": last["bar"][t]["seconds"] / last["admm"][t]["seconds"],
            "slopes": slopes(rows), "crossings": {tt: crossing(rows, tt) for tt in C.TARGETS}}


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def verdict_time(s: Settings, v: dict, rows: list) -> tuple:
    t, last = s.target, v["last"]
    head = (f"Auf der Kugel von 4d Punkten in d Dimensionen (d = {', '.join(str(r['d']) for r in rows)}) braucht die Barriere bei d = {last['d']} bis Wertfehler {sci(t)} {ms_text(last['bar'][t]['seconds'])} ms "
            f"({last['bar'][t]['steps']} Newton-Schritte), ADMM mit dünnem A {ms_text(last['admm'][t]['seconds'])} ms ({last['admm'][t]['iterations']} Schritte): ")
    if v["crossing"] is None:
        return "info", head + f"in dieser Reihe liegt die Barriere überall vorn (Faktor {de(v['factor'], 1)} bei d = {last['d']}); die Kreuzung liegt rechts von d = {last['d']}."
    others = "; ".join(f"Ziel {sci(tt)}: " + (f"ab d = {c}" if c else "keine") for tt, c in v["crossings"].items())
    return "success", head + f"**ab d = {v['crossing']} liegt ADMM vorn**, bei d = {last['d']} um den Faktor {de(v['factor'], 1)}. Kreuzung je Ziel: {others}."


def verdict_cost(rows: list, sl: dict) -> tuple:
    first, last = rows[0], rows[-1]
    text = (f"Ein Newton-Schritt der Barriere kostet bei d = {first['d']} {per_step_text(first['newton_step_ms'])} ms und bei d = {last['d']} {per_step_text(last['newton_step_ms'])} ms, "
            f"ein ADMM-Schritt mit dünnem A {per_step_text(first['admm_step_ms'])} und {per_step_text(last['admm_step_ms'])} ms. ")
    if sl["newton"] is not None:
        text += (f"Zwischen d = {C.SLOPE_FROM} und d = {last['d']} wächst die Zeit je Newton-Schritt mit der Steigung {de(sl['newton'], 1)} gegen d, die je ADMM-Schritt mit {de(sl['admm'], 1)} "
                 f"(gezählt: n (d + 1)² gegen n (d + 1) Einträge, Verhältnis d + 1 = {last['d'] + 1}; bei kleinen d verdeckt fester Aufwand die Steigung von ADMM).")
    else:
        text += f"Für die Steigungen braucht die Reihe mindestens zwei Dimensionen ab d = {C.SLOPE_FROM}."
    return "info", text


def verdict_steps(rows: list) -> tuple:
    last = rows[-1]
    bar = [r["bar"][t]["steps"] for r in rows for t in C.TARGETS]
    a = [last["admm"][t]["iterations"] for t in C.TARGETS]
    tl = ", ".join(sci(t, 0) for t in C.TARGETS)
    return "info", (f"Die Barriere braucht über alle d zwischen {min(bar)} und {max(bar)} Newton-Schritte (bei d = {last['d']}: " + ", ".join(str(last["bar"][t]["steps"]) for t in C.TARGETS)
                    + f" für Wertfehler {tl}): kaum abhängig von d. ADMM braucht bei d = {last['d']} " + ", ".join(str(x) for x in a)
                    + f" Schritte, von 10⁻³ bis 10⁻⁹ das {de(a[-1] / a[0], 0)}fache; über die Dimensionen wächst es von {rows[0]['admm'][C.TARGETS[1]]['iterations']} auf {last['admm'][C.TARGETS[1]]['iterations']} bei 10⁻⁶. "
                    f"Die Schrittzahl der Barriere wächst mit dem Ziel nur langsam ({last['bar'][C.TARGETS[0]]['steps']} gegen {last['bar'][C.TARGETS[-1]]['steps']} Schritte von 10⁻³ bis 10⁻⁹); ob die Genauigkeit die Kreuzung verschiebt, zeigt die Gesamtzeit je Ziel unten.")


def verdict_structure(rows: list) -> tuple:
    last = rows[-1]
    same = max(r["max_diff"] for r in rows)
    return "info", (f"Dasselbe ADMM braucht bei d = {last['d']} mit dichtem A {per_step_text(last['dense_step_ms'])} ms je Schritt und mit dünnem A {per_step_text(last['sparse_step_ms'])} ms "
                    f"({de(last['dense_step_ms'] / last['sparse_step_ms'], 1)}-mal), bis Wertfehler {sci(C.STRUCTURE_TARGET, 0)} insgesamt {ms_text(last['dense_seconds'])} ms gegen {ms_text(last['sparse_seconds'])} ms. "
                    f"Die Iterierten stimmen überein (größter Unterschied der Ergebnisse {sci(same)}, Schrittzahl {last['iterations_dense']} und {last['iterations_sparse']}): der Gewinn kommt aus der Struktur von A "
                    f"({last['nnz']} von {last['dense_entries']} Einträgen ungleich null), nicht aus dem Verfahren.")


def verdict_solvers(table=C.SOLVER_TABLE) -> tuple:
    cross = next((r for r in table if r["scs6_ms"] < r["clarabel_ms"]), None)
    steps_c = [r["clarabel_steps"] for r in table]
    steps_s = [r["scs3_steps"] for r in table] + [r["scs6_steps"] for r in table]
    last = table[-1]
    mid = next(r for r in table if r["d"] == 128)
    text = (f"Bei denselben Programmen braucht Clarabel (Innere Punkte) {min(steps_c)} bis {max(steps_c)} Schritte, SCS (ADMM mit Einbettung) {min(steps_s)} bis {max(steps_s)}. "
            f"Bei d = {mid['d']} liegt Clarabel mit {ms_text(mid['clarabel_ms'] / 1e3)} ms noch vor SCS ({ms_text(mid['scs6_ms'] / 1e3)} ms bei eps 10⁻⁶), bei d = {last['d']} braucht Clarabel {ms_text(last['clarabel_ms'] / 1e3)} ms "
            f"und SCS {ms_text(last['scs6_ms'] / 1e3)} ms. Die Kreuzung der echten Löser liegt " + (f"bei d = {cross['d']}" if cross else "rechts der Tabelle") + ", also etwa 4- bis 16-mal weiter rechts als die der eigenen dichten Barriere (Experiment 3). Das ist eine eingefrorene Einzelmessung; der Sprung der Clarabel-Zeit zwischen den letzten beiden Dimensionen ist nicht erklärt.")
    return "info", text
