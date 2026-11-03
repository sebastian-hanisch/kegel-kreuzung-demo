"""Konstanten: Regler-Grenzen, feste Reihen der Experimente, Farben, Presets und die eingefrorene Löser-Tabelle (Experiment 5)."""

DIMS = (4, 8, 16, 32, 64)                              # Dimensionen der Reihe; Regler d_max schneidet ab
DMAX_CHOICES = (16, 32, 64)
DEFAULT_DMAX = 32
TARGETS = (1e-3, 1e-6, 1e-9)                           # Wertfehler gegen R*, bei dem die Verfahren verglichen werden
DEFAULT_TARGET = 1e-6
RHO_GRID = (0.003, 0.01, 0.03, 0.1)                    # kleine Reihe für das beste ρ von ADMM je Dimension und Ziel
ADMM_CAP = 8000
BARRIER_TOLS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
SEED_MAX, DEFAULT_SEED = 999999, 35
STRUCTURE_TARGET = 1e-6                                # Ziel des Vergleichs dicht gegen dünn (Experiment 4)
SLOPE_FROM = 16                                        # Steigungen werden ab dieser Dimension angepasst

COLORS = {"barrier": "#888888", "admm": "#2F6B65", "dense": "#b08a2e", "sparse": "#2F6B65", "ref2": "#c9c9c9", "clarabel": "#888888", "scs": "#2F6B65"}
COLOR_OK = "#2F6B65"
COLOR_WARN = "#b08a2e"
COLOR_BAD = "#b34a4a"

_BASE = {"dmax": DEFAULT_DMAX, "target": DEFAULT_TARGET, "seed": DEFAULT_SEED}
PRESETS = {
    "Standardfall: Dimensionen bis 32, Ziel 10⁻⁶": dict(_BASE),
    "Kleine Programme (bis d = 16): die Barriere liegt vorn": dict(_BASE, dmax=16),
    "Bis d = 64: ADMM liegt deutlich vorn": dict(_BASE, dmax=64),
    "Grobes Ziel 10⁻³: die Kreuzung kommt kaum früher": dict(_BASE, target=1e-3),
    "Hohes Ziel 10⁻⁹: kaum später": dict(_BASE, target=1e-9),
}
PRESET_HELP = {
    "Standardfall: Dimensionen bis 32, Ziel 10⁻⁶": "Die Kugel von 4d Punkten in d Dimensionen für d = 4, 8, 16, 32; Barriere und ADMM bis Wertfehler 10⁻⁶. Etwa bei d = 16 bis 32 kreuzen sich die Zeiten.",
    "Kleine Programme (bis d = 16): die Barriere liegt vorn": "Nur d = 4, 8, 16: hier ist die Barriere bei Ziel 10⁻⁶ schneller (wie in Stück 9); die Kreuzung liegt noch rechts.",
    "Bis d = 64: ADMM liegt deutlich vorn": "Bis d = 64 (die eigene dichte Barriere braucht dafür etwa 15 bis 20 Sekunden): ADMM ist dort um ein Vielfaches schneller.",
    "Grobes Ziel 10⁻³: die Kreuzung kommt kaum früher": "Wertfehler 10⁻³ statt 10⁻⁶: die Kreuzung liegt auch hier bei d = 16 bis 32; die Genauigkeit verschiebt sie um weniger als einen Rasterschritt (der Vorsprung von ADMM ist bei d = 64 aber größer).",
    "Hohes Ziel 10⁻⁹: kaum später": "Wertfehler 10⁻⁹: die Kreuzung verschiebt sich um weniger als einen Rasterschritt, der Vorsprung von ADMM bei d = 64 schrumpft aber von etwa dem 25- bis 29fachen (10⁻³) auf das 3- bis 5fache.",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["dmax"] in DMAX_CHOICES and p["target"] in TARGETS for p in PRESETS.values())

# Eingefrorene Messung (Experiment 5): cvxpy 1.9.3, Clarabel (Innere Punkte, dünn besetzt) gegen SCS (ADMM mit Einbettung), dieselbe Kugel von n = 4d Punkten, Seed 35, Entwicklungsrechner 2026-11-02.
# Zeit = solver_stats.solve_time in ms; Wertfehler gegen Clarabel mit Toleranz 1e-10. Nicht live gerechnet: tools/messung_loeser.py im Repo wiederholt sie (dev-Abhängigkeit cvxpy).
SOLVER_TABLE = (
    {"d": 8, "n": 32, "clarabel_ms": 1.2, "clarabel_steps": 13, "scs3_ms": 5.6, "scs3_steps": 175, "scs3_err": 8.5e-7, "scs6_ms": 1.3, "scs6_steps": 275, "scs6_err": 1.7e-9},
    {"d": 16, "n": 64, "clarabel_ms": 3.7, "clarabel_steps": 10, "scs3_ms": 8.3, "scs3_steps": 300, "scs3_err": 1.0e-5, "scs6_ms": 9.5, "scs6_steps": 450, "scs6_err": 5.5e-8},
    {"d": 32, "n": 128, "clarabel_ms": 15.3, "clarabel_steps": 10, "scs3_ms": 20.0, "scs3_steps": 275, "scs3_err": 1.8e-5, "scs6_ms": 29.8, "scs6_steps": 375, "scs6_err": 2.9e-10},
    {"d": 64, "n": 256, "clarabel_ms": 89.8, "clarabel_steps": 10, "scs3_ms": 223.3, "scs3_steps": 425, "scs3_err": 3.9e-6, "scs6_ms": 370.7, "scs6_steps": 550, "scs6_err": 6.5e-8},
    {"d": 128, "n": 512, "clarabel_ms": 654.6, "clarabel_steps": 11, "scs3_ms": 1053.6, "scs3_steps": 525, "scs3_err": 3.8e-6, "scs6_ms": 1466.9, "scs6_steps": 675, "scs6_err": 2.9e-8},
    {"d": 256, "n": 1024, "clarabel_ms": 103228.7, "clarabel_steps": 19, "scs3_ms": 5083.0, "scs3_steps": 625, "scs3_err": 4.9e-5, "scs6_ms": 7208.0, "scs6_steps": 875, "scs6_err": 1.8e-9},
)
