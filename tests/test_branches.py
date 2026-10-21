"""Zweige der Hauptansicht (Nullspalte-Lehre): Jeder Weg, den die App zeigt, kommt in mindestens einem Preset vor.

Ein Preset, das seine Beschreibung nicht einlöst, fällt in `test_claims.py` auf. Hier: die Zweige und dass jede Abbildung für jeden Zweig gebaut werden kann (kein Fehler, nichtleere Daten)."""

import mlp_constants as C
import mlp_evaluation as ev
import mlp_visualization as V

from conftest import cached


def _all():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


def test_every_preset_validates_clean():
    for name, a in _all().items():
        assert ev.validate(a) == [], name


def test_every_branch_of_the_result_is_reached():
    analyses = list(_all().values())
    assert any(a.wins[0] > a.wins[2] for a in analyses) and any(a.wins[0] < a.wins[2] for a in analyses) and any(a.wins[1] > 0 for a in analyses)       # mehrstufig vorn, hinten, mit Gleichständen
    assert any(a.settings.refine == "finest" for a in analyses) and any(a.settings.refine == "each" for a in analyses)
    assert any(len(a.first.levels) >= 8 for a in analyses) and any(len(a.first.levels) <= 5 for a in analyses)
    assert any(a.n <= 100 for a in analyses) and any(a.n >= 256 for a in analyses)


def test_every_net_kind_but_the_ring_is_used_by_a_preset():
    assert {p["kind"] for p in C.PRESETS.values()} == set(C.KINDS) - {"ring"}                  # der Ring kommt in den Experimenten und in den Orakel-Tests vor


def test_the_figures_build_for_every_preset_and_every_level():
    for name, a in _all().items():
        r = a.first
        assert V.build_runs(a).data and V.build_trace(r).data, name
        for li in range(len(r.levels)):
            assert V.build_level_map(a.inst, r.levels, li, r.sides[li]).data, (name, li)


def test_the_experiment_figures_build_from_the_cached_rows():
    assert V.build_exact(cached("exactness_rows")).data and V.build_compare(cached("compare_rows")).data and V.build_shrink(cached("shrink_rows")).data
    assert V.build_refine(cached("refine_rows")).data and V.build_scale(cached("scale_rows")).data
