"""PDF-Export des aktuellen Vergleichs (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from kkr_evaluation import Settings, de, ms_text, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "ν": "nu", "λ": "lambda", "β": "beta", "α": "alpha", "ρ": "rho", "μ": "mu", "τ": "tau", "κ": "kappa", "€": "EUR",
            "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "„": '"', "“": '"', "ᵀ": "^T", "ℓ": "l", "∞": "inf", "√": "sqrt", "‖": "|", "°": "o"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, rows: list, view: dict) -> bytes:
    t = s.target
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Splitting gegen Innere Punkte"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    cross = view["crossing"]
    pdf.multi_cell(0, 5, _s(f"Kleinste umschließende Kugel von 4d Punkten in d Dimensionen (Lorentz-Programm), Ziel: Wertfehler {sci(t, 0)} gegen R*. "
                            f"{'ADMM liegt ab d = ' + str(cross) + ' vorn.' if cross else 'In dieser Reihe liegt die Barriere überall vorn.'} Alles simuliert, Zeiten vom Entwicklungsrechner."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head = ("d", "n", "Barriere: Schritte", "Barriere: ms", "ADMM: Schritte", "ADMM: ms", "Faktor")
    widths = (18, 18, 40, 40, 40, 40, 30)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for r in rows:
        b, a = r["bar"][t], r["admm"][t]
        cells = (r["d"], r["n"], b["steps"], ms_text(b["seconds"]), a["iterations"], ms_text(a["seconds"]), de(b["seconds"] / a["seconds"], 1))
        for c, w in zip(cells, widths):
            pdf.cell(w, 5.5, _s(c), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kegel-kreuzung-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
