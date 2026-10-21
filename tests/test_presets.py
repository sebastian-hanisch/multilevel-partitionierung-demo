"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen; Permalink-Grenzen."""

import mlp_constants as C
import mlp_evaluation as ev
import mlp_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        assert preset["kind"] in C.KINDS and preset["eps"] in C.EPS_CHOICES and preset["refine"] in C.REFINE_MODES, name
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.DEG_MIN <= preset["deg"] <= C.DEG_MAX and C.LINKS_MIN <= preset["links"] <= C.LINKS_MAX, name
        assert 0 <= preset["seed"] <= C.SEED_MAX and C.COARSE_MIN <= preset["coarse"] <= C.COARSE_MAX, name


def test_the_first_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[next(iter(C.PRESETS))]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_every_choice_field_is_listed():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert {spec.url_param for spec in P.SETTING_SPECS.values()} == {"kind", "n", "deg", "links", "seed", "eps", "coarse", "refine"}
    assert set(P.CHOICES) == {"kind_select", "eps_select", "refine_select"}
    assert P.CHOICES["kind_select"] == C.KINDS and P.CHOICES["eps_select"] == C.EPS_CHOICES and P.CHOICES["refine_select"] == C.REFINE_MODES


def test_every_default_is_a_member_of_its_range():
    assert C.DEFAULT_KIND in C.KINDS and C.DEFAULT_EPS in C.EPS_CHOICES and C.DEFAULT_REFINE in C.REFINE_MODES
    assert C.N_MIN <= C.DEFAULT_N <= C.N_MAX and C.COARSE_MIN <= C.DEFAULT_COARSE <= C.COARSE_MAX and C.DEG_MIN <= C.DEFAULT_DEG <= C.DEG_MAX and C.LINKS_MIN <= C.DEFAULT_LINKS <= C.LINKS_MAX
    assert C.N_MIN < C.N_MAX and C.COARSE_MIN < C.COARSE_MAX and C.DEG_MIN < C.DEG_MAX and C.LINKS_MIN < C.LINKS_MAX                        # kein Regler mit min == max


def test_every_kind_has_a_name_a_help_text_and_a_regler_note():
    assert set(C.KIND_NAMES) == set(C.KIND_HELP) == set(C.KIND_REGLER) == set(C.KINDS)
    assert set(C.REFINE_NAMES) == set(C.REFINE_MODES)


def test_the_experiment_grid_uses_only_known_kinds_and_sizes_inside_the_limits():
    assert all(kind in C.KINDS for kind, _deg in C.EX_KINDS) and [k for k, _ in C.EX_KINDS] == list(C.KINDS)
    assert all(kind in C.KINDS for kind, _deg in C.EX_SCALE_KINDS) and all(C.N_MIN <= n <= C.N_MAX for n in C.EX_SCALE_NS) and C.EX_SCALE_NS[-1] == C.EX_N
    assert C.N_MIN <= C.EX_N <= C.N_MAX and int(C.EX_N ** 0.5) ** 2 == C.EX_N and all(C.DEG_MIN <= d <= C.DEG_MAX for _k, d in C.EX_KINDS if d)          # Quadratzahl: Gitter ohne Rundung
    assert all(int(n ** 0.5) ** 2 == n for n in C.EX_SCALE_NS) and len(C.SWEEP_SEEDS) == len(set(C.SWEEP_SEEDS)) == 6 and C.EX_RUNS >= 1 and 0 < C.STALL < 1


def test_the_aliases_in_the_texts_match_the_presets():
    sizes = {name: p["n"] for name, p in C.PRESETS.items()}
    assert sizes["Kartennetz: mehrstufig gewinnt"] == 300 and sizes["Gitter: kein Vorteil"] == 256 and sizes["Kleines Kartennetz: noch kein Vorteil"] == 64
    assert C.PRESETS["Verfeinern nur auf der feinsten Stufe"]["refine"] == "finest" and sum(p["refine"] == "finest" for p in C.PRESETS.values()) == 1
