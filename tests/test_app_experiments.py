"""Pfade der App, die der Rauchtest nicht erreicht: die fünf Experimente im Seiteninhalt, PDF, jedes Ziel, jede Größe (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kkr_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _widget(at, key):
    if key == "seed_input":
        return at.sidebar.number_input(key=key)
    return at.sidebar.select_slider(key=key)


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    for key, value in state.items():
        _widget(at, key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_all_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["ADMM liegt vorn ab d ="] in ("16", "32") and values["Barriere bei d = 32"].endswith("ms") and values["ADMM bei d = 32"].endswith("ms")
    assert float(values["Faktor Barriere / ADMM"].replace(",", ".")) > 1.0 and "/" in values["Steigung je Schritt (Newton / ADMM)"]
    text = _texts(at)
    for part in ("Auf der Kugel von 4d Punkten in d Dimensionen (d = 4, 8, 16, 32)", "Ein Newton-Schritt der Barriere kostet bei d = 4", "Die Barriere braucht über alle d zwischen", "Dasselbe ADMM braucht bei d = 32 mit dichtem A",
                 "Bei denselben Programmen braucht Clarabel (Innere Punkte) 10 bis 19 Schritte"):
        assert part in text, part


def test_the_five_experiments_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "📐 Mathematische Formulierung"]
    assert len(at.dataframe) >= 5 and not at.error


def test_the_experiment_tables_have_one_row_per_dimension():
    at = _app()
    frames = [d.value for d in at.dataframe]
    assert list(frames[0]["d"]) == [4, 8, 16, 32] and list(frames[1]["d"]) == [4, 8, 16, 32] and len(frames[2]) == 12 and list(frames[3]["d"]) == [4, 8, 16, 32] and list(frames[4]["d"]) == [8, 16, 32, 64, 128, 256]
    assert list(frames[0]["Verhältnis"]) == [5, 9, 17, 33]


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("target", C.TARGETS)
def test_every_target_renders(target):
    at = _app(target_select=target)
    assert not at.error and any("Auf der Kugel von 4d Punkten" in s.value for s in list(at.success) + list(at.info))


def test_the_smallest_range_renders():
    at = _app(dmax_select=16)
    assert not at.error and {m.label: m.value for m in at.metric}["Barriere bei d = 16"].endswith("ms")


def test_the_biggest_range_renders_and_shows_the_crossing():
    at = _app(dmax_select=64)
    assert not at.error and {m.label: m.value for m in at.metric}["ADMM liegt vorn ab d ="] in ("16", "32")
