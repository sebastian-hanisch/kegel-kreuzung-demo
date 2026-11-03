# Splitting gegen Innere Punkte – wo kreuzen sich die Kosten, und hängt es an der Genauigkeit? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-kreuzung-demo.streamlit.app/)** -->

Stück 10 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst. Dies ist der **Konvergenzpunkt** der beiden Stränge: die Barriere (Innere Punkte, Stück 5 bis 7) und ADMM (Erste Ordnung, Stück 8 und 9) lösen dasselbe Programm in wachsender Größe.

## Kernfrage

**Ab welcher Größe und Genauigkeit gewinnt wer?** Die Kugel von `n = 4d` Punkten in `d` Dimensionen ist ein Lorentz-Programm mit `d + 1` Variablen und `n (d + 1)` Zeilen. Die Barriere braucht dafür 18 bis 44 Newton-Schritte (kaum abhängig von `d`), ADMM 70 bis 1 440 Schritte. Ein **Newton-Schritt** rechnet mit `n (d + 1)²` Einträgen (die Hesse-Blöcke des Kegels sind dicht; der Code bildet sie mit dichten Produkten, das ist noch teurer), ein **ADMM-Schritt** mit dünn besetztem `A` mit `n (d + 1)`: gemessen wächst die Zeit je Newton-Schritt mit der Steigung **3,2** gegen `d`, je ADMM-Schritt mit **0,6** (bis `d = 64`; die Vormessung bis `d = 128` ergab 1,5 bis 1,8). Daraus und aus festem Aufwand bei kleinem `d` folgt, dass die **Gesamtzeiten bei `d` ≈ 16 bis 32 kreuzen**; bei `d = 64` ist ADMM etwa 5- bis 7-mal schneller (bis `10⁻⁶`). **Die Genauigkeit verschiebt die Kreuzung auf dem Raster der Dimensionen nicht** (bei allen drei Zielen zwischen `d = 16` und `32`), **schmälert aber den Vorsprung von ADMM:** bei `d = 64` ist ADMM bei `10⁻³` etwa 25- bis 29-mal schneller, bei `10⁻⁹` nur etwa 3- bis 5-mal. ADMM braucht für `10⁻⁹` bei `d = 64` das 10fache der Schritte von `10⁻³` (bei kleinerem `d` das 3- bis 7fache), die Barriere das 1,8fache; die Hook „Splitting früh, Innere Punkte bei hoher Genauigkeit“ zeigt auf dieser Familie in die richtige Richtung, verschiebt die Kreuzung aber um weniger als einen Rasterschritt. Gegen die tatsächlichen Löser von cvxpy (Clarabel, SCS) liegt die Kreuzung nach einer eingefrorenen Einzelmessung bei `d` ≈ 128 bis 256, also etwa 4- bis 16-mal weiter rechts.

Das Beispiel bleibt die kleinste umschließende Kugel einer Punktwolke, jetzt in wachsender Dimension.

## Modell

- **Kern** (`kegel_kern.py`, wortgleich mit Stück 2 bis 9, unverändert): das Lorentz-Programm der Kugel und seine Barrierenmethode (dichtes `A`, Newton-System je Schritt).
- **Familie** (`kkr_family.py`): die Kugel von `4d` Punkten (mindestens 8) in `d` Dimensionen, anisotrop normalverteilt (Standardabweichungen von 1 bis 3), als dichtes `ConeProblem` und als dünn besetztes Programm (`scipy.sparse`, `n (d + 1)` Einträge ungleich null); dazu die **Zählung** der Einträge je Schritt, `n (d + 1)²` gegen `n (d + 1)`.
- **ADMM** (`kkr_admm.py`, aus Stück 9, jetzt auch mit dünn besetztem `A`): dasselbe Verfahren; nur `A x` und `Aᵀ v` ändern sich. **Projektionen** (`kkr_projection.py`) aus Stück 8.
- **Vergleich** (`kkr_evaluation.py`): `R*` aus der Barriere (Toleranz `10⁻¹²`) ist der Maßstab für beide; die Barriere nimmt die kleinste Toleranz einer aufsteigenden Reihe, deren Wert das Ziel trifft; ADMM bricht ab, sobald Wertfehler und primales Residuum unter dem Ziel liegen, mit dem besten ρ aus 0,003 / 0,01 / 0,03 / 0,1 bei `10⁻⁶` (dieselbe Einstellung für die anderen Ziele). Zeiten nach einem Warmlauf, beste von 2 Läufen (bei `d = 64` von einem), ein BLAS-Thread.
- **Echte Löser** (`kkr_constants.SOLVER_TABLE`): eine **eingefrorene** Messung von Clarabel und SCS über cvxpy; `tools/messung_loeser.py` wiederholt sie (nur dev-Abhängigkeit cvxpy).

## Methodik

Fünf Experimente mit live berechnetem Verdikt: (1) Kosten je Schritt gegen `d` mit den Referenzgeraden `d²` und `d³` und der Zählung; (2) Schritte der Barriere und von ADMM bis `10⁻³`, `10⁻⁶`, `10⁻⁹`; (3) Gesamtzeit und Kreuzung je Ziel; (4) dasselbe ADMM mit dichtem und mit dünnem `A`; (5) die eingefrorene Tabelle von Clarabel und SCS. Regler: größte Dimension (16, 32 oder 64), Ziel und Seed der Wolke; ein Schnellstart lädt fünf Szenarien, die Adresszeile spiegelt die Konfiguration. Die Dimensionen sind 4, 8, 16, 32, 64; bei 64 braucht die eigene dichte Barriere etwa 15 bis 20 Sekunden.

## Befunde (gemessen, keine Behauptungen)

Die meisten Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen Rundung und Rechner, steht die gemessene Zahl trotzdem hier; Zeiten in Millisekunden, die Vormessungen und Linux-Werte stehen dort nicht oder nur als weites Band). Zeiten gelten für den Entwicklungsrechner und werden nur in ihrer Struktur getestet. Standardwolke: Seed 35.

| Frage | Befund |
|---|---|
| Newton-Schritte der Barriere (Wertfehler 10⁻³ / 10⁻⁶ / 10⁻⁹) | d = 4: **18 / 24 / 28**; 8: 20 / 35 / 43; 16: 23 / 33 / 38; 32: 23 / 33 / 38; 64: **24 / 37 / 44**. Über alle `d` zwischen 18 und 44; von 10⁻³ bis 10⁻⁹ bei `d = 64` das **1,8fache** |
| Schritte von ADMM (dünnes `A`, bestes ρ bei 10⁻⁶) | d = 4: 71 / 206 / 381 (ρ = 0,1); 8: 174 / 287 / 464 (0,03); 16: 113 / 475 / 837 (0,03); 32: 91 / 372 / 667 (0,01); 64: **138 / 789 / 1 441** (0,01). Von 10⁻³ bis 10⁻⁹ bei `d = 64` das **10fache** (bei `d = 4, 8, 16, 32` das 5,4-, 2,7-, 7,4- und 7,3fache); bei 10⁻⁶ wächst es von 206 auf 789 (Faktor 2,5 bis 5); das beste ρ wandert mit der Skala nach unten (Stück 9) |
| Zeit je Schritt (ms) | Newton: d = 4 **0,21**; 8: 0,25; 16: 0,53; 32: 4,9; 64: **47,6**: Steigung ab `d = 16` **3,2** (Band 2,5 bis 3,8). ADMM (dünn): 0,13; 0,13; 0,15; 0,18; **0,32**: Steigung ab `d = 16` **0,6** (Band unter 1,2; bei kleinem `d` verdeckt fester Aufwand die Steigung). Gezählt: `n (d + 1)²` gegen `n (d + 1)` Einträge, Verhältnis `d + 1`. Einzelmessung: in einem Nachlauf lagen die Zeiten das 1,5- bis 2fache höher und der Newton-Schritt bei `d = 64` streute zwischen 61 und 123 ms; die Steigungen blieben in den Bändern |
| Gesamtzeit bis 10⁻⁶ (ms, Barriere gegen ADMM; Einzelmessung, auf zwei Ziffern gerundet) | d = 4: **4,8 gegen 26**; 8: 8,5 gegen 37; 16: 18 gegen 69; 32: **160 gegen 68**; 64: **1 800 gegen 260** (Faktor 5,5 bis 7,4 in vier Läufen). Bis 10⁻⁹ (d = 64): 2 100 gegen 460 (Faktor 3 bis 5); bis 10⁻³: 1 300 gegen 44 (Faktor 25 bis 29). **Kreuzung zwischen `d = 16` und `32`** bei jedem der drei Ziele (im Test Band `d` ∈ {16, 32}; die Vormessung sah bei 10⁻³ schon bei 16 ADMM vorn: Rauschen der Zeiten) |
| Dichtes gegen dünnes `A` im ADMM (Wertfehler 10⁻⁶) | Dieselben Iterierten (größter Unterschied der Ergebnisse unter 10⁻⁹, gleiche Schrittzahl). `d = 64`: **2,4 ms** je Schritt dicht, **0,39 ms** dünn (Faktor 6; auf dem Linux-Runner der CI 2,6; 16 640 von 1 081 600 Einträgen ungleich null), insgesamt 1 911 gegen 308 ms. Bei `d = 4` ist das dichte `A` noch **billiger** (0,06 gegen 0,13 ms) |
| Clarabel (Innere Punkte) gegen SCS (ADMM mit Einbettung), cvxpy, eingefroren | Clarabel **10 bis 19 Schritte**, SCS **175 bis 875**. Löserzeit `d = 64`: 90 gegen 371 ms (SCS 10⁻⁶); `d = 128`: **655 gegen 1 467 ms** (Clarabel noch vorn); `d = 256`: **Clarabel 103 229 ms, SCS 7 208 ms** (Faktor 14). Die Kreuzung der echten Löser liegt bei `d = 256` der Tabelle (zwischen 128 und 256). Auffällig: Clarabel braucht von `d = 128` auf 256 das 158fache der Zeit (11 auf 19 Schritte), weit über jede `d³`- bis `d⁴`-Skalierung; der Sprung ist nicht erklärt (Speicher, Fill-in oder ein Artefakt der Einzelmessung), und die Kreuzung hängt an diesem einen Punkt |
| Genauigkeit von SCS | eps 10⁻³ bis 10⁻⁸ kosten bei `d = 128` 525 bis 775 Schritte und 1 054 bis 1 730 ms (Vormessung): **nicht grob**, nur das 1,6fache für fünf Größenordnungen |
| Gegen cvxpy | `R*` der Familie für `d = 4, 16, 32` gegen Clarabel auf 10⁻⁶ (Barriere und dünnes ADMM); Schrittzahlen frisch bei `d = 16`: Clarabel unter 25, SCS über 100 und im Band der Tabelle |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Steigung von ADMM ist live 0,6, nicht 1,5 bis 1,8.** Der Plan nannte die Vormessung bis `d = 128` (1,2 bis 1,8 ms je Schritt gegen 0,15 bei `d = 32`); die App rechnet bis `d = 64` (0,32 gegen 0,18 ms): dort verdeckt fester Aufwand (Projektion von 256 Kegeln, Python) die `O(n d)`-Steigung. Der Text sagt es („bei kleinem `d` verdeckt fester Aufwand die Steigung“) und die Vormessung steht in der Tabelle der Grenzen.
- **Das Ziel 10⁻³ kreuzt im Bau bei `d = 32`, nicht bei 16.** Die Vormessung hatte bei `d = 16` 18,5 gegen 13,7 ms (ADMM vorn), der Bau 12,9 gegen 16,8 ms: Rauschen im Bereich von 30 %; der Test lässt `d` ∈ {16, 32} für jedes Ziel zu.
- **Das beste ρ wird einmal bei 10⁻⁶ gewählt** und für 10⁻³ und 10⁻⁹ wiederverwendet (Vormessung: je Ziel); dadurch ist ADMM bei 10⁻³ nicht ganz optimal (d = 8: 174 statt 101 Schritte).
- **Dichtes `A` im ADMM ist bei kleinem `d` besser.** Bei `d = 4` kostet das dünne `A` wegen des Aufwands von `scipy.sparse` mehr je Schritt (0,13 gegen 0,06 ms): der Gewinn der Struktur beginnt erst bei etwa `d = 32` (0,17 gegen 0,19 ms: gleich) und ist bei `d = 64` ein Faktor 6 (Linux-Runner: 2,6).
- **Die eingefrorene Löser-Tabelle ist nicht live.** Clarabel braucht bei `d = 256` über 100 Sekunden; die App rechnet sie nicht. `tools/messung_loeser.py` reproduziert die Schrittzahlen exakt (13 / 175 / 275 bei `d = 8`); die Zeiten schwanken.
- **Die Kreuzung der echten Löser ist eine Eigenschaft dieser Familie:** vermutlich berührt jede Kegelzeile die gemeinsamen Variablen `c`, und die Faktorisierung von Clarabel füllt auf (Fill-in nicht gemessen); bei Programmen mit dünn gekoppelten Kegeln liegt sie anders. „Innere Punkte skalieren schlecht“ steht deshalb nirgends.
- **Quellen:** geprüft per Suche am 2026-11-02: Goulart und Chen (Clarabel, arXiv 2405.12762; als Preprint bestätigt, Zeitschrift nicht geprüft), O'Donoghue, Chu, Parikh, Boyd (JOTA 169 (3), 2016, S. 1042–1068, SCS), Boyd u. a. (Foundations and Trends in Machine Learning 3 (1), 2011, S. 1–122). Die Aussage der Quellen „Innere Punkte bei hoher Genauigkeit besser“ wird **gemessen, nicht übernommen**; Inhalte (Benchmarks, Beweise) nicht nachgelesen.
- **Port:** 9308 (`launch.json` und Netstat gelesen, frei).

## Ehrliche Grenzen

- **Eine Familie, ein Kegel.** Die Kugel (Lorentz, eine Variablengruppe, die alle Kegel koppelt); Lasso, LP, Programme mit einem Kegel je Variable, PDLP (nur LP) und GPUs sind nicht gemessen und können woanders kreuzen.
- **Die eigene Barriere ist dicht.** Die Kreuzung bei `d` ≈ 16 bis 32 gilt für sie gegen ADMM mit dünnem `A`; eine dünn besetzte Barriere mit Ordnungsverfahren würde sie nach rechts schieben (Experiment 5: `d` ≈ 128 bis 256, eine Einzelmessung). Eine solche wird nicht gebaut.
- **Zeiten hängen vom Rechner ab** und werden nur in ihrer Struktur getestet (Steigungen, Reihenfolge, Kreuzung in einem Band); die Tabelle der echten Löser ist eine einzelne Messung.
- **Live nur bis `d = 64`;** `R*` kommt aus der Barriere (bei größerem `d` aus ADMM nur auf etwa 10⁻⁶ genau, deshalb nicht live).
- **Wertfehler ist ein Maßstab, kein Abbruchkriterium der Praxis:** ADMM kennt `R*` im Vergleich (`ref_value`), ein Solver in der Praxis nicht; die Residuen-Toleranz ist weit vom Wertfehler entfernt (SCS: eps 10⁻³ gibt schon Wertfehler 8·10⁻⁷ bis 5·10⁻⁵, je nach `d`).
- **Keine Verschiebung der Kreuzung auf dem groben Raster heißt nicht, dass die Genauigkeit ohne Einfluss ist** (der Vorsprung von ADMM schrumpft), und eine Genauigkeitskreuzung kann es anderswo geben: ein Programm mit weicher Genauigkeitsanforderung (verrauschte Daten) bleibt offen.

## Verifikation

- **Familie** (`tests/test_family.py`): vier Punkte auf der Geraden von Hand (`d = 1`, `R* = 2`, Mittelpunkt 0, 8 Zeilen, `A = [[0, −1], [1, 0]] ×4`), dünnes gleich dichtes `A` Eintrag für Eintrag, Zählung (`n (d + 1)² = 16` gegen `n (d + 1) = 8` für `n = 4, d = 1`; Verhältnis `d + 1`), ADMM auf der Handinstanz, gleiche Iterierte für dichtes und dünnes `A`, Form, Mittelwert, Determinismus und Streuung der Wolke, `R*` wächst mit `d`.
- **Verfahren** (`tests/test_admm.py`): das Lorentz-Handprogramm von Stück 9 (`x = 1,5` nach einem Schritt, Fixpunkt `x = 5`, `u = (1; −0,6; −0,8)`), das Orthant-Handprogramm, Übereinstimmung mit der Barriere für dichtes und dünnes `A`, Abbruch mit Referenzwert, Ablehnung von `A` ohne vollen Spaltenrang (dicht und dünn) und von ρ ≤ 0, Residuenausgleich auf dünnem `A`.
- **Orakel** (`tests/test_oracle_kegel_kreuzung.py`): `R*` für `d = 4, 16, 32` gegen cvxpy mit Clarabel (`cp.SOC`), die Schrittzahlen von Clarabel und SCS frisch bei `d = 16` gegen die Tabelle.
- **Randfälle** (`tests/test_edge_cases.py`): `d = 1` und `2`, `n = d + 1` Punkte, doppelte Punkte, Daten der Skala 10⁻³ und 10³ mit Ausgleich, ein Ziel unter der Rundung, ein anderer Seed (Kreuzung im selben Band), der kleinste Bereich.
- **Verdikte** (`tests/test_verdicts.py`, Werte von Hand gesetzt), **Claims** (`tests/test_claims.py`: die meisten Zahlen dieser README, Zeiten nur als Struktur, auch der Aufbau der Tabelle der echten Löser), **Oberfläche** (AppTest: jeder Button, jeder Regler an Minimum und Maximum, jedes Ziel, jede Größe, PDF), Presets (die Aussagen der Hilfetexte), Permalink, Layout, Abbildungen, Footer, Laufzeit-Importe, Kern (Prüfsumme gegen das Schwester-Repo).
- **Umfang:** 148 Tests (4 davon, die cvxpy-Orakel gegen Clarabel und SCS, überspringen sich ohne cvxpy; mit cvxpy laufen alle 4 grün), rund 2,5 Minuten (gemessen 2026-11-03, Windows); kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy ist Gegenprobe im dev-Satz; ohne es überspringen sich die Orakel-Tests. Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2 bis 9 |
| `kkr_family.py` | Die Kugel von 4d Punkten in d Dimensionen, dicht und dünn, Zählung der Einträge je Schritt |
| `kkr_admm.py` | ADMM für Kegelprogramme (aus Stück 9, mit Unterstützung für `scipy.sparse`) |
| `kkr_projection.py` | Projektionen auf fünf Kegel (aus Stück 8) |
| `kkr_constants.py` | Regler-Grenzen, feste Reihen, Farben, Presets, eingefrorene Tabelle der echten Löser |
| `kkr_evaluation.py` | Messung je Dimension, Steigungen, Kreuzung, Verdikte, Formatierung |
| `kkr_layout.py` | Fester Kopf (Stylesheet) |
| `kkr_pdf_export.py` | PDF-Export (fpdf2) |
| `kkr_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kkr_visualization.py` | Plotly-Abbildungen (2D-Achsen gesperrt) |
| `tools/messung_loeser.py` | Löser-Tabelle (Experiment 5) neu messen (dev-Abhängigkeit cvxpy) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Familie, Verfahren, Projektionen, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe, Kern |

## Bewusst nicht umgesetzt

- **Eine dünn besetzte Barriere mit Ordnungsverfahren:** Clarabel ist die Gegenprobe; hier nur genannt.
- **Andere Familien** (Lasso, LP, ein Kegel je Variable, PSD): nicht gebaut; Stück 16 (Verfahrenswahl) fasst zusammen.
- **PDLP, GPU, Mehrkern:** nicht gemessen.
- **`d` über 64 live:** zu langsam für die dichte Barriere; nur die eingefrorene Tabelle geht bis 256.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
