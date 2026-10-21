"""Kennzahlen, Prüfung der Ergebnisse und Verdikte. Verdikte sind reine Funktionen (Art, Text): Zahlen kommen aus den Zeilen, nie aus dem Text.

Standorte sind in der Anzeige ab 1 nummeriert (`inst.name`), im Kern ab 0."""

from dataclasses import dataclass, field

import mlp_algorithm as A
import mlp_constants as C
import mlp_scenario as S


@dataclass(frozen=True)
class Settings:
    kind: str = C.DEFAULT_KIND
    n: int = C.DEFAULT_N
    deg: float = C.DEFAULT_DEG
    links: int = C.DEFAULT_LINKS
    seed: int = C.DEFAULT_SEED
    eps: int = C.DEFAULT_EPS
    coarse: int = C.DEFAULT_COARSE
    refine: str = C.DEFAULT_REFINE


def run_rng(seed, run, salt=0):
    """Zufallsquelle des Laufs `run` (jeder Lauf hat eine eigene Reihenfolge beim Vergröbern und eigene Anordnungen)."""
    return S.make_rng(seed * 53 + C.RUN_SALT + run, salt)


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    inst: S.Instance = field(compare=False, repr=False)
    graph: A.WGraph = field(compare=False, repr=False)
    ml: tuple = field(compare=False, repr=False)             # MLResult je Lauf
    sl: tuple = field(compare=False, repr=False)             # SLResult je Lauf, mit dem Aufwand des mehrstufigen Laufs als Budget

    @property
    def n(self):
        return self.graph.n

    @property
    def m(self):
        return self.inst.m

    @property
    def ml_cuts(self):
        return tuple(r.cut for r in self.ml)

    @property
    def sl_cuts(self):
        return tuple(r.cut for r in self.sl)

    @property
    def wins(self):
        """(mehrstufig besser, gleich, einstufig besser) über die Läufe."""
        pairs = list(zip(self.ml_cuts, self.sl_cuts))
        return sum(1 for a, b in pairs if a < b), sum(1 for a, b in pairs if a == b), sum(1 for a, b in pairs if a > b)

    @property
    def ml_mean(self):
        return sum(self.ml_cuts) / len(self.ml)

    @property
    def sl_mean(self):
        return sum(self.sl_cuts) / len(self.sl)

    @property
    def first(self):
        return self.ml[0]

    @property
    def shrink(self):
        """Verhältnis der Knotenzahlen aufeinanderfolgender Stufen des ersten Laufs."""
        sizes = [lv.graph.n for lv in self.first.levels]
        return tuple(b / a for a, b in zip(sizes, sizes[1:]))


def analyse(settings: Settings) -> Analysis:
    inst = S.make_instance(settings.kind, settings.n, settings.deg, settings.links, settings.seed)
    g = A.from_pairs(inst.n, inst.pairs)
    ml, sl = [], []
    for run in range(C.RUNS):
        m = A.multilevel(g, settings.eps, run_rng(settings.seed, run), settings.coarse, refine=settings.refine)
        ml.append(m)
        sl.append(A.single_level(g, settings.eps, run_rng(settings.seed, run, 1), m.work))
    return Analysis(settings, inst, g, tuple(ml), tuple(sl))


def validate(a: Analysis) -> list:
    """Unabhängige Prüfung vor der Anzeige: gibt die Liste der Probleme zurück (leer = alles in Ordnung)."""
    problems = []
    g, eps = a.graph, a.settings.eps
    for r in a.ml:
        if not A.is_balanced(g, r.side, eps) or any(s not in (0, 1) for s in r.side) or len(set(r.side)) != 2:
            problems.append("Eine mehrstufige Teilung ist nicht ausgewogen oder nicht in zwei nichtleere Hälften geteilt")
            break
        if r.cut != A.cut_value(g, r.side) or r.trace[0].cut_refined != r.cut:
            problems.append("Der gemeldete Schnitt einer mehrstufigen Teilung stimmt nicht mit der Nachzählung überein")
            break
        levels = r.levels
        if any(lv.graph.total != g.total for lv in levels) or any(b.graph.n >= a_.graph.n for a_, b in zip(levels, levels[1:])):
            problems.append("Das Vergröbern erhält das Gesamtgewicht nicht oder verkleinert das Netz nicht")
            break
        if any(r.trace[i].cut_projected != r.trace[i + 1].cut_refined for i in range(len(r.trace) - 1)):
            problems.append("Beim Übertragen auf die feinere Stufe hat sich der Schnitt verändert")
            break
        if any(t.cut_refined > t.cut_projected for t in r.trace):
            problems.append("Das Verfeinern hat den Schnitt vergrößert")
            break
    for r, m in zip(a.sl, a.ml):
        if not A.is_balanced(g, r.side, eps) or r.cut != A.cut_value(g, r.side) or r.cut != min(r.starts):
            problems.append("Eine einstufige Teilung ist nicht ausgewogen oder ihr Schnitt stimmt nicht")
            break
        if r.moves < m.work and len(r.starts) == 1 and r.moves < 1:
            problems.append("Das einstufige Verfahren hat sein Budget nicht benutzt")
            break
    return problems


# --- Formate ---------------------------------------------------------------------------------------------------------------------------------------


def _i(x) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{x:,.0f}".replace(",", ".")


def _dec(x, digits=1) -> str:
    """Zahl mit Komma; ganze Werte ohne Nachkommastelle."""
    x = float(x)
    if x == int(x):
        return _i(x)
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(share, digits=0) -> str:
    return _dec(round(100.0 * share, digits), digits) if digits else _i(round(100.0 * share))


def _of(k, n) -> str:
    return f"{_i(k)} von {_i(n)}"


def _stufen(c) -> str:
    return "1 Stufe" if c == 1 else f"{_i(c)} Stufen"


# --- Verdikte -------------------------------------------------------------------------------------------------------------------------------------


def verdict_run(a: Analysis):
    """Kernfrage: Schnitt des mehrstufigen Verfahrens gegen den des einstufigen bei gleichem Aufwand, über die Läufe."""
    s = a.settings
    ml, sl = a.ml_cuts, a.sl_cuts
    better, equal, worse = a.wins
    r0 = a.first
    head = (f"**Bei {_i(a.n)} Standorten schneidet das mehrstufige Verfahren im Mittel {_dec(a.ml_mean)} Leitungen, das einstufige bei gleichem Aufwand {_dec(a.sl_mean)}** ({_i(len(ml))} Läufe; "
            f"mehrstufig besser in {_of(better, len(ml))}, gleich in {_of(equal, len(ml))}, schlechter in {_of(worse, len(ml))}).")
    steps = (f" Ein mehrstufiger Lauf braucht {_stufen(len(r0.levels))} (Knotenzahlen {' → '.join(_i(lv.graph.n) for lv in r0.levels)}) und {_i(r0.work)} Schritte; "
             f"das einstufige Verfahren schafft dafür im Mittel {_dec(sum(len(r.starts) for r in a.sl) / len(a.sl))} Starts.")
    if s.refine == "finest":
        steps += " Verfeinert wird nur auf der feinsten Stufe."
    return "info", head + steps


def verdict_exact(rows):
    nets, checks, wrong = (sum(r[k] for r in rows) for k in ("nets", "checks", "wrong"))
    if wrong:
        return "warning", f"Auf {_i(nets)} Netzen weichen {_i(wrong)} von {_i(checks)} Prüfungen ab."
    return "info", (f"Auf {_i(nets)} Netzen mit {_i(rows[0]['n'])} Standorten ({len(rows)} Netztypen) bestehen {_i(checks)} Prüfungen: jede Teilung ausgewogen und in zwei nichtleere Hälften geteilt, jeder Schnitt gleich der "
                    f"Nachzählung, das Gesamtgewicht auf jeder Stufe erhalten, jede Stufe kleiner als die feinere, der Schnitt beim Übertragen unverändert und nie größer nach dem Verfeinern. Abweichungen: 0.")


def _signed(share, digits=0) -> str:
    """Prozentzahl mit Vorzeichen (negativ mit Minuszeichen), z. B. 31 oder −1,8."""
    v = round(100.0 * share, digits)
    text = _dec(abs(v), digits) if digits else _i(abs(v))
    return ("−" if v < 0 else "") + text


def verdict_compare(rows):
    runs = sum(r["runs"] for r in rows)
    ml, sl = sum(r["sum_ml"] for r in rows), sum(r["sum_sl"] for r in rows)
    wins, ties = sum(r["wins"] for r in rows), sum(r["ties"] for r in rows)
    gain = lambda r: 1 - r["sum_ml"] / r["sum_sl"]   # noqa: E731
    per = ", ".join(f"{r['label']} {_signed(gain(r))} %" for r in sorted(rows, key=gain, reverse=True))
    text = (f"Bei {_i(rows[0]['n'])} Standorten (je Netztyp {_i(rows[0]['nets'])} Netze, je Netz {_i(rows[0]['runs'] // rows[0]['nets'])} Läufe) ist der Schnitt des mehrstufigen Verfahrens bei gleichem Aufwand in der Summe über alle Netze "
            f"{_pct(1 - ml / sl)} % kleiner als der des einstufigen (mehrstufig besser in {_of(wins, runs)} Läufen, gleich in {_of(ties, runs)}). Je Netztyp (positiv: mehrstufig besser): {per}.")
    return "info", text


def verdict_shrink(rows):
    parts = []
    for r in sorted(rows, key=lambda r: r["mean_shrink"]):
        parts.append(f"{r['label']} {_dec(r['mean_shrink'], 2)} ({_dec(r['mean_levels'])} Stufen)")
    return "info", f"Mittleres Verhältnis der Knotenzahlen zweier aufeinanderfolgender Stufen (kleiner heißt schnelleres Vergröbern) bei {_i(rows[0]['n'])} Standorten: " + "; ".join(parts) + "."


def verdict_refine(rows):
    nets = sum(r["nets"] for r in rows)
    proj, fin, each = (sum(r[k] for r in rows) for k in ("sum_projected", "sum_finest", "sum_each"))
    return "info", (f"Nur auf die feinste Stufe zu übertragen (ohne jedes Verfeinern) ergibt im Mittel {_dec(proj / nets)} Leitungen, Verfeinern nur auf der feinsten Stufe {_dec(fin / nets)}, auf jeder Stufe "
                    f"{_dec(each / nets)} ({_i(nets)} Netze mit {_i(rows[0]['n'])} Standorten, {_i(len(rows))} Netztypen).")


def verdict_scale(rows):
    out = []
    for kind in dict.fromkeys(r["kind"] for r in rows):
        sel = [r for r in rows if r["kind"] == kind]
        out.append(f"{sel[0]['label']}: " + ", ".join(f"{_i(r['n'])} Standorte {_signed(1 - r['sum_ml'] / r['sum_sl'])} %" for r in sel))
    return "info", "Vorteil des mehrstufigen Verfahrens (weniger Leitungen in Prozent) bei wachsender Netzgröße: " + "; ".join(out) + "."
