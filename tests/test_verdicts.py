"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Seeds; so prüft der Test Grenzen und jede genannte Zahl direkt."""

import mlp_constants as C
import mlp_evaluation as ev


def test_formats():
    assert ev._i(1234567) == "1.234.567" and ev._dec(3.0) == "3" and ev._dec(1234.5) == "1.234,5" and ev._dec(0.75) == "0,8" and ev._dec(2.25, 2) == "2,25"
    assert ev._pct(0.256) == "26" and ev._pct(0.256, 1) == "25,6" and ev._of(3, 1200) == "3 von 1.200" and ev._stufen(1) == "1 Stufe" and ev._stufen(6) == "6 Stufen"
    assert ev._signed(0.31) == "31" and ev._signed(-0.018, 1) == "−1,8" and ev._signed(-0.004) == "0" and ev._signed(0.0) == "0" and ev._signed(-0.1) == "−10"


def test_verdict_exact_counts_and_warns_on_any_deviation():
    rows = [{"nets": 3, "checks": 30, "wrong": 0, "n": 64}, {"nets": 2, "checks": 20, "wrong": 0, "n": 64}]
    kind, text = ev.verdict_exact(rows)
    assert kind == "info" and text.startswith("Auf 5 Netzen mit 64 Standorten (2 Netztypen) bestehen 50 Prüfungen:") and text.endswith("Abweichungen: 0.")
    rows[0]["wrong"] = 2
    rows[1]["wrong"] = 1
    assert ev.verdict_exact(rows) == ("warning", "Auf 5 Netzen weichen 3 von 50 Prüfungen ab.")


def test_verdict_compare_sums_and_names_every_type_with_its_sign():
    rows = [{"label": "A", "n": 100, "nets": 2, "runs": 8, "sum_ml": 20, "sum_sl": 40, "wins": 7, "ties": 1}, {"label": "B", "n": 100, "nets": 2, "runs": 8, "sum_ml": 60, "sum_sl": 60, "wins": 0, "ties": 2}]
    kind, text = ev.verdict_compare(rows)
    assert kind == "info"
    assert text == ("Bei 100 Standorten (je Netztyp 2 Netze, je Netz 4 Läufe) ist der Schnitt des mehrstufigen Verfahrens bei gleichem Aufwand in der Summe über alle Netze 20 % kleiner als der des einstufigen "
                    "(mehrstufig besser in 7 von 16 Läufen, gleich in 3 von 16). Je Netztyp (positiv: mehrstufig besser): A 50 %, B 0 %.")
    rows[1]["sum_ml"] = 66
    text = ev.verdict_compare(rows)[1]
    assert "in der Summe über alle Netze 14 % kleiner" in text and text.endswith("A 50 %, B −10 %.")


def test_verdict_shrink_orders_by_speed():
    rows = [{"label": "B", "n": 100, "mean_shrink": 0.73, "mean_levels": 8.5}, {"label": "A", "n": 100, "mean_shrink": 0.55, "mean_levels": 5.0}]
    kind, text = ev.verdict_shrink(rows)
    assert kind == "info"
    assert text == ("Mittleres Verhältnis der Knotenzahlen zweier aufeinanderfolgender Stufen (kleiner heißt schnelleres Vergröbern) bei 100 Standorten: A 0,55 (5 Stufen); B 0,73 (8,5 Stufen).")


def test_verdict_refine_gives_the_three_means():
    rows = [{"label": "A", "n": 100, "nets": 2, "sum_projected": 200, "sum_finest": 100, "sum_each": 60}, {"label": "B", "n": 100, "nets": 2, "sum_projected": 100, "sum_finest": 60, "sum_each": 40}]
    assert ev.verdict_refine(rows) == ("info", "Nur auf die feinste Stufe zu übertragen (ohne jedes Verfeinern) ergibt im Mittel 75 Leitungen, Verfeinern nur auf der feinsten Stufe 40, auf jeder Stufe 25 (4 Netze mit 100 Standorten, 2 Netztypen).")


def test_verdict_scale_per_type_with_signs():
    rows = [{"kind": "geometric", "label": "K", "n": 64, "sum_ml": 9, "sum_sl": 10}, {"kind": "geometric", "label": "K", "n": 256, "sum_ml": 8, "sum_sl": 10}, {"kind": "random", "label": "Z", "n": 64, "sum_ml": 10.5, "sum_sl": 10}]
    assert ev.verdict_scale(rows) == ("info", "Vorteil des mehrstufigen Verfahrens (weniger Leitungen in Prozent) bei wachsender Netzgröße: K: 64 Standorte 10 %, 256 Standorte 20 %; Z: 64 Standorte −5 %.")


def _a(name, **over):
    return ev.analyse(ev.Settings(**{**C.PRESETS[name], **over}))


def test_verdict_run_for_the_default_preset():
    kind, text = ev.verdict_run(_a("Kartennetz: mehrstufig gewinnt"))
    assert kind == "info"
    assert text == ("**Bei 300 Standorten schneidet das mehrstufige Verfahren im Mittel 8,9 Leitungen, das einstufige bei gleichem Aufwand 12,4** (8 Läufe; mehrstufig besser in 6 von 8, gleich in 1 von 8, schlechter in 1 von 8). "
                    "Ein mehrstufiger Lauf braucht 6 Stufen (Knotenzahlen 300 → 169 → 98 → 55 → 30 → 18) und 1.404 Schritte; das einstufige Verfahren schafft dafür im Mittel 1,5 Starts.")


def test_verdict_run_branches():
    fin = ev.verdict_run(_a("Verfeinern nur auf der feinsten Stufe"))[1]
    assert fin.endswith("Verfeinert wird nur auf der feinsten Stufe.") and "mehrstufig besser in 3 von 8, gleich in 1 von 8, schlechter in 4 von 8" in fin
    grid = ev.verdict_run(_a("Gitter: kein Vorteil"))[1]
    assert "das einstufige bei gleichem Aufwand 16**" in grid                                    # ganze Zahl ohne Nachkommastelle
    one = ev.verdict_run(_a("Kleines Kartennetz: noch kein Vorteil", n=24, coarse=60))[1]
    assert "braucht 1 Stufe (Knotenzahlen 24)" in one
