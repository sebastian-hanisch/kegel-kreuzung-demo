"""Splitting gegen Innere Punkte – wo kreuzen sich die Kosten, und hängt es an der Genauigkeit? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 10 der Reihe „Konvexe und konische Optimierung“: der Konvergenzpunkt der beiden Stränge. Die Barriere (Innere Punkte, Stück 5 bis 7) und ADMM (Erste Ordnung, Stück 8 und 9) lösen dieselbe Kugel von 4d Punkten in d Dimensionen;
die Demo misst Kosten je Schritt, Schritte, Gesamtzeit, den Einfluss der Struktur von A und stellt Clarabel gegen SCS.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kkr_constants as C
from kkr_evaluation import (
    Settings,
    de,
    ms_text,
    per_step_text,
    scaling_rows,
    sci,
    structure_rows,
    verdict_cost,
    verdict_solvers,
    verdict_steps,
    verdict_structure,
    verdict_time,
    view,
)
from kkr_layout import dunkles_design, kopf_css
from kkr_pdf_export import build_pdf
from kkr_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from kkr_visualization import build_solvers, build_step_cost, build_steps, build_structure, build_time

st.set_page_config(page_title="Splitting gegen Innere Punkte – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _scaling(dmax, seed):
    return scaling_rows(Settings(dmax=dmax, seed=seed))


@st.cache_data(show_spinner=False)
def _structure(dmax, seed):
    return structure_rows(Settings(dmax=dmax, seed=seed), _scaling(dmax, seed))


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kkr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⚖️ Splitting gegen Innere Punkte – wo kreuzen sich die Kosten, und hängt es an der Genauigkeit?")
st.markdown(
    """
**Die Kreuzung hängt vor allem an der Größe und der Implementierung; die Genauigkeit verschiebt sie auf dem Raster der Dimensionen nicht, schmälert aber den Vorsprung von ADMM.** Dieselbe Kugel von `4d` Punkten in `d` Dimensionen ist ein Lorentz-Programm mit `d + 1` Variablen; die Barriere (Innere Punkte)
braucht dafür etwa 20 bis 45 Newton-Schritte, ADMM einige hundert Schritte. Ein **Newton-Schritt** rechnet mit `n (d + 1)²` Einträgen (die Hesse-Blöcke des Kegels sind dicht; der Code bildet sie mit dichten Produkten, das ist noch teurer), ein **ADMM-Schritt** mit dünn besetztem `A` mit `n (d + 1)`: wächst `d`,
kippt die Gesamtzeit zugunsten von ADMM. Die Genauigkeit verschiebt die Kreuzung auf dem Raster nicht, schmälert aber den Vorsprung von ADMM (bei `d = 64` von etwa dem 25- bis 29fachen bei `10⁻³` auf das 3- bis 5fache bei `10⁻⁹`); ADMM braucht für `10⁻⁹` bei `d = 64` das 10fache der Schritte von `10⁻³`, die Barriere das 1,8fache.
Gegen die echten Löser von cvxpy (Clarabel, SCS) liegt die Kreuzung nach einer eingefrorenen Einzelmessung etwa 4- bis 16-mal weiter rechts.
Stück 10 der Reihe „Konvexe und konische Optimierung“; der Konvergenzpunkt von Innere Punkte und Erste Ordnung.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Die Familie:** die kleinste umschließende Kugel von `n = 4d` Punkten in `d` Dimensionen als Lorentz-Programm (`d + 1` Variablen, `n (d + 1)` Zeilen), dieselbe Aufgabe wie in Stück 2 bis 9, nur mit wachsendem `d`.
2. **Zwei Löser, unverändert im Verfahren:** die Barrierenmethode des Kerns (dichtes `A`, Newton-System je Schritt) und das ADMM aus Stück 9, jetzt mit dünn besetztem `A` (`scipy.sparse`: nur `A x` und `Aᵀ v` ändern sich).
3. **Das Ziel:** Wertfehler `|cᵀx − R*|/R*` unter `10⁻³`, `10⁻⁶` oder `10⁻⁹`; `R*` aus der Barriere mit Toleranz `10⁻¹²` ist der Maßstab für beide. Die Barriere nimmt die kleinste Toleranz, deren Wert das Ziel trifft; ADMM das beste ρ der kleinen Reihe.
4. **Zeiten** nach einem Warmlauf, beste von zwei Läufen, ein BLAS-Thread; sie hängen vom Rechner ab, getestet wird ihre Struktur.
5. **Die Kreuzung** ist die erste Dimension der Reihe, bei der ADMM schneller ist.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    dmax = st.select_slider("Größte Dimension d", options=C.DMAX_CHOICES, key="dmax_select",
                            help="Die Reihe läuft für d = 4, 8, 16, 32, 64 bis zu diesem Wert; bei 64 braucht die eigene dichte Barriere etwa 15 bis 20 Sekunden.")
    target = st.select_slider("Ziel: Wertfehler", options=C.TARGETS, key="target_select", format_func=sci, help="Bis zu welchem Wertfehler gegen R* beide Verfahren laufen (Ergebnis und Experiment 3).")
    seed = st.number_input("Zufalls-Seed der Wolke", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Wolke generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"dmax_select": int(dmax), "target_select": float(target), "seed_input": int(seed)})

settings = Settings(int(dmax), float(target), int(seed))
rows = _scaling(settings.dmax, settings.seed)
v = view(settings, rows)
sl = v["slopes"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m4, m5 = st.columns(2)
m1.metric("ADMM liegt vorn ab d =", str(v["crossing"]) if v["crossing"] else "keine Kreuzung", help="Die erste Dimension der Reihe, bei der ADMM bis zum gewählten Ziel schneller ist als die Barriere.")
m2.metric(f"Barriere bei d = {v['last']['d']}", f"{ms_text(v['bar_ms'] / 1e3)} ms", help=f"{v['last']['bar'][settings.target]['steps']} Newton-Schritte bis Wertfehler {sci(settings.target)}.")
m3.metric(f"ADMM bei d = {v['last']['d']}", f"{ms_text(v['admm_ms'] / 1e3)} ms", help=f"{v['last']['admm'][settings.target]['iterations']} Schritte bis Wertfehler {sci(settings.target)} (ρ = {v['last']['rho']}).")
m4.metric("Faktor Barriere / ADMM", de(v["factor"], 1), help="Größer als 1: ADMM ist schneller.")
m5.metric("Steigung je Schritt (Newton / ADMM)", f"{de(sl['newton'], 1)} / {de(sl['admm'], 1)}" if sl["newton"] is not None else "–",
          help=f"Zeit je Schritt gegen d in der log-log-Darstellung, angepasst ab d = {C.SLOPE_FROM} (ab zwei Dimensionen).")
st.plotly_chart(build_time(rows, settings.target, v["crossing"]), width="stretch", key="main_chart")
st.caption("Zeit bis zum gewählten Wertfehler gegen die Dimension (logarithmisch); die gepunktete Linie markiert die Kreuzung.")

if st.button("📄 PDF des aktuellen Vergleichs erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, rows, v))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kegel-kreuzung.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ab welcher Größe und Genauigkeit gewinnt wer?")
verdict(verdict_time(settings, v, rows))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Kosten je Schritt gegen d")
st.caption("Zeit je Newton-Schritt der Barriere (dichte Hesse-Blöcke) und je ADMM-Schritt (dünnes A) bei `n = 4d` Punkten. Darunter die Zahl der Einträge je Schritt, **gezählt** (exakt, kein Messwert).")
st.plotly_chart(build_step_cost(rows), width="stretch", key="cost_chart")
st.dataframe(pd.DataFrame([{"d": r["d"], "Punkte n": r["n"], "Variablen": r["variables"], "Zeilen": r["rows"], "Einträge von A ungleich null": r["nnz"], "Newton: n (d + 1)²": r["newton_ops"],
                            "ADMM: n (d + 1)": r["admm_ops"], "Verhältnis": r["d"] + 1, "ms je Newton-Schritt": per_step_text(r["newton_step_ms"]), "ms je ADMM-Schritt": per_step_text(r["admm_step_ms"])} for r in rows]),
             hide_index=True, width="stretch")
verdict(verdict_cost(rows, sl))

st.markdown("---")

st.subheader("🔬 Experiment 2: Schritte gegen d")
st.caption("Newton-Schritte der Barriere und Schritte von ADMM bis Wertfehler 10⁻³, 10⁻⁶ und 10⁻⁹ gegen die Dimension; ADMM mit dem besten ρ der Reihe 0,003 / 0,01 / 0,03 / 0,1 bei 10⁻⁶.")
st.plotly_chart(build_steps(rows), width="stretch", key="steps_chart")
st.dataframe(pd.DataFrame([{"d": r["d"], **{f"Barriere {sci(t, 0)}": r["bar"][t]["steps"] for t in C.TARGETS}, **{f"ADMM {sci(t, 0)}": (r["admm"][t]["iterations"] if r["admm"][t]["converged"] else f"≥ {r['admm'][t]['iterations']}")
                                                                                                                      for t in C.TARGETS}, "bestes ρ": r["rho"]} for r in rows]), hide_index=True, width="stretch")
verdict(verdict_steps(rows))

st.markdown("---")

st.subheader("🔬 Experiment 3: Gesamtzeit und Kreuzung")
st.caption("Zeit bis zum Ziel für alle drei Genauigkeiten; ein Faktor über 1 heißt: ADMM ist schneller. Die Kreuzung je Ziel steht im Verdikt.")
st.dataframe(pd.DataFrame([{"d": r["d"], "Ziel": sci(t, 0), "Barriere: Schritte": r["bar"][t]["steps"], "Barriere: ms": ms_text(r["bar"][t]["seconds"]), "ADMM: Schritte": r["admm"][t]["iterations"],
                            "ADMM: ms": ms_text(r["admm"][t]["seconds"]), "Faktor": de(r["bar"][t]["seconds"] / r["admm"][t]["seconds"], 1)} for r in rows for t in C.TARGETS]), hide_index=True, width="stretch")
verdict(verdict_time(settings, v, rows))

st.markdown("---")

st.subheader("🔬 Experiment 4: Struktur ausnutzen")
st.caption(f"Dasselbe ADMM (gleiches ρ, gleicher Abbruch bei Wertfehler {sci(C.STRUCTURE_TARGET, 0)}) mit dichtem und mit dünnem A: dieselben Iterierten, aber die Zeit je Schritt unterscheidet sich.")
structure = _structure(settings.dmax, settings.seed)
st.plotly_chart(build_structure(structure), width="stretch", key="structure_chart")
st.dataframe(pd.DataFrame([{"d": r["d"], "Schritte dicht / dünn": f"{r['iterations_dense']} / {r['iterations_sparse']}", "ms je Schritt dicht": per_step_text(r["dense_step_ms"]), "ms je Schritt dünn": per_step_text(r["sparse_step_ms"]),
                            "Faktor": de(r["dense_step_ms"] / r["sparse_step_ms"], 1), "größter Unterschied der Ergebnisse": sci(r["max_diff"])} for r in structure]), hide_index=True, width="stretch")
verdict(verdict_structure(structure))

st.markdown("---")

st.subheader("🔬 Experiment 5: Die echten Löser")
st.caption("Clarabel (Innere Punkte, dünn besetzt) gegen SCS (ADMM mit Einbettung) auf derselben Kugel, Löserzeit aus cvxpy. **Eingefrorene Messung** (Entwicklungsrechner, 2026-11-02), nicht live: `tools/messung_loeser.py` im Repo wiederholt sie.")
st.plotly_chart(build_solvers(), width="stretch", key="solvers_chart")
st.dataframe(pd.DataFrame([{"d": r["d"], "Punkte n": r["n"], "Clarabel: ms": ms_text(r["clarabel_ms"] / 1e3), "Clarabel: Schritte": r["clarabel_steps"], "SCS 10⁻³: ms": ms_text(r["scs3_ms"] / 1e3), "SCS 10⁻³: Schritte": r["scs3_steps"],
                            "SCS 10⁻³: Wertfehler": sci(r["scs3_err"]), "SCS 10⁻⁶: ms": ms_text(r["scs6_ms"] / 1e3), "SCS 10⁻⁶: Schritte": r["scs6_steps"], "SCS 10⁻⁶: Wertfehler": sci(r["scs6_err"])} for r in C.SOLVER_TABLE]),
             hide_index=True, width="stretch")
verdict(verdict_solvers())

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Splitting gewinnt bei grober, Innere Punkte bei hoher Genauigkeit** | Nur zum Teil: ADMM braucht bei d = 64 für 10⁻⁹ das 10fache der Schritte von 10⁻³ (bei kleinerem d das 3- bis 7fache), die Barriere das 1,8fache; die Kreuzung bleibt auf dem Raster bei d = 16 bis 32, aber der Vorsprung von ADMM bei d = 64 schrumpft von etwa dem 25- bis 29fachen auf das 3- bis 5fache (Experiment 2 und 3). | Stück 16 |
| **Die Kreuzung liegt bei d ≈ 16 bis 32** | Nur für die eigene dichte Barriere gegen das eigene ADMM mit dünnem A. Die echten Löser kreuzen nach einer eingefrorenen Einzelmessung erst bei d ≈ 128 bis 256 (Experiment 5), vermutlich weil Clarabel dünn besetzt rechnet; der Sprung der Clarabel-Zeit von d = 128 auf 256 (158fache Zeit) ist nicht erklärt. | – |
| **Innere Punkte skalieren schlecht** | Vermutlich nur bei dieser Familie: jede Kegelzeile berührt die gemeinsamen Variablen, die Faktorisierung füllt vermutlich auf (nicht gemessen; Clarabel braucht bei d = 256 über 100 Sekunden für 19 Schritte). Bei Programmen mit dünn gekoppelten Kegeln ist das anders. | – |
| **Ein ADMM-Schritt ist immer um den Faktor d billiger** | Bei kleinem d verdeckt fester Aufwand (Python, kleine Matrizen) die Steigung; gemessen ist bis d = 64 eine Steigung unter 1, die Vormessung bis d = 128 etwa 1,5 bis 1,8. | – |
| **ADMM ist ohne Weiteres schnell** | Nur mit dünn besetztem A: dasselbe Verfahren mit dichtem A ist bei d = 64 deutlich langsamer (Experiment 4); der Gewinn kommt aus der Struktur. | – |
| **Die Familie ist typisch** | Eine Familie, ein Kegel (Lorentz), ein Satz gemeinsamer Variablen. Lasso, LP oder Probleme mit einem Kegel je Variable kreuzen woanders; PDLP (nur LP) und GPUs sind hier nicht gemessen. | Stück 16 |
| **Die Zeiten gelten allgemein** | Sie hängen vom Rechner ab; getestet wird nur ihre Struktur (Steigungen, Reihenfolge, Kreuzung in einem weiten Band). Die Tabelle der echten Löser ist eine eingefrorene Messung. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Die Familie.** $\min R$ unter $\|x_i - c\|_2 \le R$ für $i = 1, \dots, n$ mit $x_i \in \mathbb{R}^d$ und $n = 4d$. In Kegelform: Variablen $(c, R) \in \mathbb{R}^{d+1}$, je Punkt der Lorentz-Block $s_i = (R;\, x_i - c) \in K_{d+1}$,
also $A \in \mathbb{R}^{n(d+1) \times (d+1)}$ mit $n(d+1)$ Einträgen ungleich null (ein $-1$ für $R$ und $d$ Einsen für $c$ je Punkt).

**Kosten je Schritt.** Ein Newton-Schritt der Barriere löst $(A^\top H A)\,\Delta x = -g$ mit der blockdiagonalen Hesse-Matrix $H$ der Lorentz-Barriere. Jeder Block von $H$ ist dicht von der Größe $(d+1)^2$: das sind
$n\,(d+1)^2$ Einträge (der Code bildet sie mit dichten Produkten, das kostet mehr Rechenoperationen als diese Zählung). Ein ADMM-Schritt rechnet $A x$ und $A^\top v$ (je $O(\mathrm{nnz}(A)) = O(n\,d)$), eine Projektion je Kegel ($O(d)$) und ein $(d{+}1)\times(d{+}1)$-Produkt mit $(A^\top A)^{-1}$: $O(n\,d)$ insgesamt.
Das Verhältnis ist $d + 1$.

**Gesamtzeit.** Zeit $\approx$ Schritte $\times$ Kosten je Schritt. Für die Barriere sind die Schritte in dieser Messung nahezu unabhängig von $d$ ($\nu = 8d$ wächst, die Schrittzahl kaum; die Worst-Case-Schranke wäre $O(\sqrt{\nu}\log(1/\varepsilon))$, für das hier verwendete Verfahren mit großen Schritten gilt sie nicht), für ADMM wachsen sie langsam mit $d$.
Die Kreuzung ist die Dimension, bei der $\text{Schritte}_{\text{ADMM}} \cdot n d = \text{Schritte}_{\text{Barriere}} \cdot n d^2$ (gezählte Einträge, soweit sie die Zeit bestimmen; bei kleinem $d$ tut es fester Aufwand).

**Literatur.** P. J. Goulart, Y. Chen: *Clarabel: An interior-point solver for conic programs with quadratic objectives*, arXiv:2405.12762 (Preprint; die Zeitschriftenfassung ist hier nicht geprüft).
B. O'Donoghue, E. Chu, N. Parikh, S. Boyd: *Conic Optimization via Operator Splitting and Homogeneous Self-Dual Embedding*, Journal of Optimization Theory and Applications 169 (3), 2016, S. 1042–1068 (SCS).
S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein: *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*, Foundations and Trends in Machine Learning 3 (1), 2011, S. 1–122.
Aussagen der Quellen über „Innere Punkte bei hoher Genauigkeit, Splitting bei grober“ werden hier nicht übernommen, sondern gemessen; Inhalte der Arbeiten (Benchmarks, Beweise) sind nicht nachgelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
