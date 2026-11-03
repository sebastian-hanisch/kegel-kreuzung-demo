"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte; die Aussagen der Hilfetexte."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import kkr_constants as C
import kkr_evaluation as ev
import kkr_presets as PR

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: Dimensionen bis 32, Ziel 10⁻⁶"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(PR.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_bounds():
    for preset in C.PRESETS.values():
        assert preset["dmax"] in C.DMAX_CHOICES and preset["target"] in C.TARGETS and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings() and next(iter(C.PRESETS)) == DEFAULT


def test_setting_specs_cover_every_preset_key():
    assert set(PR.PRESET_KEYS.values()) == set(PR.SETTING_SPECS)


def test_the_help_texts_name_what_the_presets_produce():
    small = ev.Settings(**C.PRESETS["Kleine Programme (bis d = 16): die Barriere liegt vorn"])
    rows = ev.scaling_rows(small)
    assert [r["d"] for r in rows] == [4, 8, 16] and small.target == 1e-6 and ev.crossing(rows[:2], 1e-6) is None and rows[2]["bar"][1e-6]["seconds"] < rows[2]["admm"][1e-6]["seconds"]
    coarse = ev.Settings(**C.PRESETS["Grobes Ziel 10⁻³: die Kreuzung kommt kaum früher"])
    fine = ev.Settings(**C.PRESETS["Hohes Ziel 10⁻⁹: kaum später"])
    assert coarse.target == 1e-3 and fine.target == 1e-9 and coarse.dmax == fine.dmax == C.DEFAULT_DMAX
    big = ev.Settings(**C.PRESETS["Bis d = 64: ADMM liegt deutlich vorn"])
    assert big.dmax == 64 and "15 bis 20 Sekunden" in C.PRESET_HELP["Bis d = 64: ADMM liegt deutlich vorn"]


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["dmax"] = "16"
    at.query_params["target"] = "0.5"       # kein erlaubter Wert: bleibt Standard
    at.query_params["seed"] = "-5"          # unter der Grenze: wird begrenzt
    at.run()
    assert not at.exception
    assert at.session_state["dmax_select"] == 16 and at.session_state["target_select"] == C.DEFAULT_TARGET and at.sidebar.number_input(key="seed_input").value == 0
    bad = AppTest.from_file(APP, default_timeout=600)
    bad.query_params["dmax"] = "17"
    bad.query_params["seed"] = "abc"
    bad.run()
    assert not bad.exception and bad.session_state["dmax_select"] == C.DEFAULT_DMAX and bad.sidebar.number_input(key="seed_input").value == C.DEFAULT_SEED
