"""Jede Zahl im README (Abschnitt „Befunde“) wird hier über dieselben Experimentfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zähler und Mittel über feste Netze (je Netztyp sechs Netze mit 256 Standorten und je vier Läufe, im Verlauf nach Größe drei Netze), nie ein einzelner Zufalls-Seed. Aufwand in Zügen und besuchten Standorten, nicht in
Sekunden. Die Zahlen der Vormessung im Einzelplan stammen aus Wegwerfskripten mit anderen Netzen und Zeitbudgets; maßgeblich sind diese hier."""

import pytest

import mlp_constants as C
import mlp_evaluation as ev

from conftest import cached

LABELS = ["Kartennetz", "Gitter", "Ring mit Abkürzungen", "Skalenfreies Netz", "Zufallsnetz, Grad 4", "Verbundnetz, Grad 5"]


def test_every_witness_holds_on_every_network():
    rows = cached("exactness_rows")
    assert [r["label"] for r in rows] == LABELS and all(r["n"] == 256 and r["nets"] == 6 and r["checks"] == 216 for r in rows)
    assert sum(r["nets"] for r in rows) == 36 and sum(r["checks"] for r in rows) == 1296 and sum(r["wrong"] for r in rows) == 0                    # je Netz 4 Läufe · 9 Prüfungen


def test_the_multilevel_cut_is_a_third_smaller_on_the_map_network_and_about_equal_elsewhere():
    rows = cached("compare_rows")
    assert all(r["n"] == 256 and r["nets"] == 6 and r["runs"] == 24 for r in rows)
    assert [r["sum_ml"] for r in rows] == [34.75, 99.25, 155.0, 559.5, 507.0, 278.5] and [r["sum_sl"] for r in rows] == [50.25, 97.5, 153.75, 568.0, 521.25, 274.0]
    gains = [1 - r["sum_ml"] / r["sum_sl"] for r in rows]
    assert [round(100 * x) for x in gains] == [31, -2, -1, 1, 3, -2] and round(100 * (1 - sum(r["sum_ml"] for r in rows) / sum(r["sum_sl"] for r in rows))) == 2
    assert round(rows[0]["sum_ml"] / 6, 1) == 5.8 and round(rows[0]["sum_sl"] / 6, 1) == 8.4


def test_how_often_multilevel_wins_ties_or_loses():
    rows = cached("compare_rows")
    assert [r["wins"] for r in rows] == [18, 1, 7, 17, 17, 10] and sum(r["wins"] for r in rows) == 70
    assert [r["ties"] for r in rows] == [3, 19, 3, 0, 2, 2] and sum(r["ties"] for r in rows) == 29 and sum(r["runs"] for r in rows) == 144
    assert [r["sum_ml_best"] for r in rows] == [24, 96, 140, 541, 489, 258] and [r["sum_sl_best"] for r in rows] == [30, 96, 139, 545, 495, 255]
    assert [round(r["sum_starts"] / r["runs"], 1) for r in rows] == [1.5, 1.8, 1.8, 2.8, 2.2, 2.1]                  # Starts des einstufigen Verfahrens im Aufwand eines mehrstufigen Laufs
    assert [round(r["sum_work"] / r["runs"]) for r in rows] == [1440, 1624, 1737, 2400, 1899, 1796]


def test_scale_free_networks_shrink_slowly_and_need_more_levels():
    rows = cached("shrink_rows")
    assert [round(r["mean_shrink"], 2) for r in rows] == [0.58, 0.55, 0.55, 0.73, 0.6, 0.58] and [round(r["mean_levels"], 1) for r in rows] == [6.0, 5.1, 5.2, 8.8, 6.0, 5.9]
    assert [round(r["mean_first"], 2) for r in rows] == [0.56, 0.55, 0.55, 0.65, 0.58, 0.57] and all(0.5 <= r["mean_shrink"] < 1 for r in rows)                  # kein Schritt unter der Hälfte


def test_refining_is_essential_and_refining_every_level_helps_a_little_more():
    rows = cached("refine_rows")
    assert [round(r["sum_projected"] / 6, 1) for r in rows] == [14.6, 58.5, 54.2, 123.7, 119.3, 111.0]
    assert [round(r["sum_finest"] / 6, 1) for r in rows] == [8.1, 16.6, 28.8, 97.6, 88.0, 47.8] and [round(r["sum_each"] / 6, 1) for r in rows] == [5.8, 16.5, 25.8, 93.2, 84.5, 46.4]
    total = lambda key: sum(r[key] for r in rows) / 36   # noqa: E731
    assert (round(total("sum_projected"), 1), round(total("sum_finest"), 1), round(total("sum_each"), 1)) == (80.2, 47.8, 45.4)


def test_the_advantage_appears_only_on_larger_map_networks():
    rows = cached("scale_rows")
    geo = [r for r in rows if r["kind"] == "geometric"]
    rnd = [r for r in rows if r["kind"] == "random"]
    assert [r["n"] for r in geo] == [64, 144, 256] and [r["n"] for r in rnd] == [64, 144, 256]
    assert [(r["sum_ml"], r["sum_sl"]) for r in geo] == [(14.5, 13.25), (16.75, 16.25), (16.0, 20.25)]
    assert [round(100 * (1 - r["sum_ml"] / r["sum_sl"])) for r in geo] == [-9, -3, 21] and [round(100 * (1 - r["sum_ml"] / r["sum_sl"])) for r in rnd] == [-1, -3, 0]


def test_the_preset_descriptions_hold():
    def a(name):
        return ev.analyse(ev.Settings(**C.PRESETS[name]))
    karte = a("Kartennetz: mehrstufig gewinnt")
    assert karte.n == 300 and round(karte.ml_mean, 1) == 8.9 and round(karte.sl_mean, 1) == 12.4 and karte.wins == (6, 1, 1)
    zufall = a("Zufallsnetz: ein kleiner Vorteil")
    assert (round(zufall.ml_mean, 1), round(zufall.sl_mean, 1)) == (66.9, 69.8) and zufall.wins == (5, 1, 2)
    gitter = a("Gitter: kein Vorteil")
    assert gitter.n == 256 and gitter.sl_cuts == (16,) * 8 and gitter.ml_cuts.count(16) == 6 and gitter.wins == (0, 6, 2)
    sf = a("Skalenfreies Netz: einstufig besser")
    assert len(sf.first.levels) == 9 and sf.sl_mean < sf.ml_mean and sf.wins == (2, 0, 6)
    verbund = a("Verbundnetz: einstufig reicht")
    assert verbund.ml_mean > verbund.sl_mean and verbund.wins == (2, 3, 3)
    klein = a("Kleines Kartennetz: noch kein Vorteil")
    assert klein.n == 64 and klein.wins == (2, 5, 1)
    fein = a("Verfeinern nur auf der feinsten Stufe")
    each = ev.analyse(ev.Settings(**{**C.PRESETS["Verfeinern nur auf der feinsten Stufe"], "refine": "each"}))
    assert round(fein.ml_mean, 1) == 6.9 and round(each.ml_mean, 1) == 5.4 and each.ml_mean < fein.ml_mean
    assert pytest.approx(fein.sl_mean) == 6.25
