"""Die Familie der Kugelprogramme: die kleinste umschließende Kugel von n = 4d Punkten in d Dimensionen als Lorentz-Programm.

  Variablen (c, R): d + 1.   Je Punkt i ein Lorentz-Block s_i = (R; x_i − c) ∈ K_{d+1}:  n (d + 1) Zeilen, n Kegel der Größe d + 1.
  dichte Fassung   ConeProblem des Kerns (A als dichte Matrix, wie in Stück 2 bis 9): die Barriere braucht sie so.
  dünne Fassung    dieselbe Matrix als scipy.sparse (n (d + 1) Einträge ungleich null: ein −1 für R und d Einsen für c je Punkt): ADMM braucht nur A x und Aᵀ v.
Operationszählung (exakt, kein Messwert): ein Newton-Schritt der Barriere formt je Kegel einen dichten (d + 1) × (d + 1)-Block der Hesse-Matrix: n (d + 1)² Einträge;
ein ADMM-Schritt rechnet A x, Aᵀ v und eine Projektion je Kegel: etwa n (d + 1) Einträge. Das Verhältnis ist d + 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

import kegel_kern as K

POINTS_PER_DIMENSION = 4                       # n = 4 d (mindestens 8)
SPREAD = (1.0, 3.0)                            # Standardabweichungen der Koordinaten: linear von 1 bis 3 (die Wolke ist in die Länge gezogen)


def default_n(d: int) -> int:
    return max(8, POINTS_PER_DIMENSION * d)


def cloud(d: int, n: int | None = None, seed: int = 35) -> np.ndarray:
    """n normalverteilte Punkte in d Dimensionen, Koordinate j mit Standardabweichung zwischen 1 und 3, um den Mittelwert verschoben. Deterministisch aus (d, n, seed)."""
    if d < 1:
        raise ValueError("mindestens eine Dimension")
    n = default_n(d) if n is None else n
    if n < 2:
        raise ValueError("mindestens zwei Punkte")
    z = np.random.default_rng(seed).normal(size=(n, d)) * np.linspace(SPREAD[0], SPREAD[1], d)
    return z - z.mean(axis=0)


def dense_problem(points: np.ndarray) -> K.ConeProblem:
    """Je Punkt s = (R; x_i − c) im Lorentz-Kegel der Größe d + 1 (wie `build_soc_ball` in Stück 9), dichtes A."""
    n, d = points.shape
    A = np.zeros((n * (d + 1), d + 1))
    b = np.zeros(n * (d + 1))
    for i in range(n):
        r0 = i * (d + 1)
        A[r0, d] = -1.0
        A[r0 + 1:r0 + 1 + d, :d] = np.eye(d)
        b[r0 + 1:r0 + 1 + d] = points[i]
    c = np.zeros(d + 1)
    c[d] = 1.0
    c0 = points.mean(axis=0)
    x0 = np.concatenate([c0, [1.5 * float(np.linalg.norm(points - c0, axis=1).max()) + 1.0]])
    return K.ConeProblem(c, A, b, (("q", d + 1),) * n, x0)


@dataclass(frozen=True, eq=False)
class SparseProblem:
    """Dasselbe Programm mit dünn besetztem A; `groups` ist die Kegelzerlegung für kkr_admm.spec_of."""
    A: sp.csr_matrix
    b: np.ndarray
    c: np.ndarray
    groups: list


def sparse_problem(points: np.ndarray) -> SparseProblem:
    n, d = points.shape
    i = np.arange(n)
    ii, jj = np.meshgrid(i, np.arange(d), indexing="ij")
    rows = np.concatenate([i * (d + 1), (ii * (d + 1) + 1 + jj).ravel()])
    cols = np.concatenate([np.full(n, d), jj.ravel()])
    vals = np.concatenate([-np.ones(n), np.ones(n * d)])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n * (d + 1), d + 1))
    b = np.zeros((n, d + 1))
    b[:, 1:] = points
    c = np.zeros(d + 1)
    c[d] = 1.0
    return SparseProblem(A, b.reshape(-1), c, [("q", d + 1, n)])


def newton_operations(n: int, d: int) -> int:
    return n * (d + 1) ** 2


def admm_operations(n: int, d: int) -> int:
    return n * (d + 1)
