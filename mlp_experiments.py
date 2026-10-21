"""Experimente über feste Netze (nie ein einzelner Zufalls-Seed als Befund). Jede Funktion gibt eine Liste von Zeilen (dict) zurück; Verdikte und Abbildungen lesen nur diese Zeilen.

1. `exactness_rows`: stimmen Teilungen, Schnitte, Gewichte und Übertragen überein (Zeugnisse)?
2. `compare_rows`: mehrstufig gegen einstufig bei gleichem Aufwand (gleiche Zahl von Zügen und Knotenbesuchen).
3. `shrink_rows`: wie schnell schrumpft das Netz beim Vergröbern?
4. `refine_rows`: was bringt das Verfeinern (ohne, nur feinste Stufe, jede Stufe)?
5. `scale_rows`: wächst der Vorteil mit der Netzgröße?"""

from functools import lru_cache

import mlp_algorithm as A
import mlp_constants as C
import mlp_evaluation as ev
import mlp_scenario as S


def kind_label(kind, deg):
    name = C.KIND_NAMES[kind]
    return f"{name}, Grad {deg:g}".replace(".", ",") if kind in ("random", "verbund") else name


def _graph(kind, deg, n, seed):
    inst = S.make_instance(kind, n, deg, 1, seed)
    return inst, A.from_pairs(inst.n, inst.pairs)


@lru_cache(maxsize=None)
def _runs(kind, deg, n, seed):
    """Die Läufe (mehrstufig und einstufig mit dem Aufwand des mehrstufigen als Budget) für ein festes Netz; mehrere Experimente teilen sie."""
    _inst, g = _graph(kind, deg, n, seed)
    out = []
    for run in range(C.EX_RUNS):
        m = A.multilevel(g, C.DEFAULT_EPS, ev.run_rng(seed, run), C.DEFAULT_COARSE)
        out.append((m, A.single_level(g, C.DEFAULT_EPS, ev.run_rng(seed, run, 1), m.work)))
    return g, tuple(out)


def _checks(g, eps, ml, sl):
    res = []
    for r in ml:
        res += [A.is_balanced(g, r.side, eps) and len(set(r.side)) == 2, r.cut == A.cut_value(g, r.side), r.trace[0].cut_refined == r.cut]
        res.append(all(lv.graph.total == g.total for lv in r.levels))
        res.append(all(b.graph.n < a.graph.n for a, b in zip(r.levels, r.levels[1:])))
        res.append(all(r.trace[i].cut_projected == r.trace[i + 1].cut_refined for i in range(len(r.trace) - 1)))
        res.append(all(t.cut_refined <= t.cut_projected for t in r.trace))
    for r in sl:
        res += [A.is_balanced(g, r.side, eps), r.cut == A.cut_value(g, r.side) == min(r.starts)]
    return res


def exactness_rows():
    rows = []
    for kind, deg in C.EX_KINDS:
        checks = wrong = 0
        for seed in C.SWEEP_SEEDS:
            g, runs = _runs(kind, deg, C.EX_N, seed)
            ml, sl = [m for m, _s in runs], [s_ for _m, s_ in runs]
            res = _checks(g, C.DEFAULT_EPS, ml, sl)
            checks += len(res)
            wrong += res.count(False)
        rows.append({"kind": kind, "deg": deg, "label": kind_label(kind, deg), "nets": len(C.SWEEP_SEEDS), "checks": checks, "wrong": wrong, "n": C.EX_N})
    return rows


def compare_rows():
    rows = []
    for kind, deg in C.EX_KINDS:
        row = {"kind": kind, "deg": deg, "label": kind_label(kind, deg), "nets": 0, "n": 0, "runs": 0, "sum_ml": 0, "sum_sl": 0, "wins": 0, "ties": 0, "sum_work": 0, "sum_starts": 0, "sum_ml_best": 0, "sum_sl_best": 0}
        for seed in C.SWEEP_SEEDS:
            g, runs = _runs(kind, deg, C.EX_N, seed)
            row["nets"] += 1
            row["n"] = g.n
            mls, sls = [], []
            for m, s_ in runs:
                mls.append(m.cut)
                sls.append(s_.cut)
                row["runs"] += 1
                row["wins"] += m.cut < s_.cut
                row["ties"] += m.cut == s_.cut
                row["sum_work"] += m.work
                row["sum_starts"] += len(s_.starts)
            row["sum_ml"] += sum(mls) / len(mls)
            row["sum_sl"] += sum(sls) / len(sls)
            row["sum_ml_best"] += min(mls)
            row["sum_sl_best"] += min(sls)
        rows.append(row)
    return rows


def shrink_rows():
    rows = []
    for kind, deg in C.EX_KINDS:
        row = {"kind": kind, "deg": deg, "label": kind_label(kind, deg), "nets": 0, "n": 0, "sum_shrink": 0.0, "count": 0, "sum_levels": 0, "sum_first": 0.0}
        for seed in C.SWEEP_SEEDS:
            _inst, g = _graph(kind, deg, C.EX_N, seed)
            row["nets"] += 1
            row["n"] = g.n
            for run in range(C.EX_RUNS):
                levels = A.coarsen(g, ev.run_rng(seed, run), C.DEFAULT_COARSE, C.DEFAULT_EPS)
                sizes = [lv.graph.n for lv in levels]
                ratios = [b / a for a, b in zip(sizes, sizes[1:])]
                row["sum_shrink"] += sum(ratios)
                row["count"] += len(ratios)
                row["sum_levels"] += len(levels)
                row["sum_first"] += ratios[0] if ratios else 1.0
        row["mean_shrink"] = row["sum_shrink"] / row["count"]
        row["mean_levels"] = row["sum_levels"] / (row["nets"] * C.EX_RUNS)
        row["mean_first"] = row["sum_first"] / (row["nets"] * C.EX_RUNS)
        rows.append(row)
    return rows


def refine_rows():
    rows = []
    for kind, deg in C.EX_KINDS:
        row = {"kind": kind, "deg": deg, "label": kind_label(kind, deg), "nets": 0, "n": 0, "sum_projected": 0.0, "sum_finest": 0.0, "sum_each": 0.0}
        for seed in C.SWEEP_SEEDS:
            g, runs = _runs(kind, deg, C.EX_N, seed)
            row["nets"] += 1
            row["n"] = g.n
            proj = fin = each = 0
            for run, (m_each, _s) in enumerate(runs):
                m_fin = A.multilevel(g, C.DEFAULT_EPS, ev.run_rng(seed, run), C.DEFAULT_COARSE, refine="finest")
                each += m_each.cut
                fin += m_fin.cut
                proj += m_fin.trace[0].cut_projected                               # beim feinsten Netz übertragen, ohne dort zu verfeinern
            row["sum_projected"] += proj / C.EX_RUNS
            row["sum_finest"] += fin / C.EX_RUNS
            row["sum_each"] += each / C.EX_RUNS
        rows.append(row)
    return rows


def scale_rows():
    rows = []
    for kind, deg in C.EX_SCALE_KINDS:
        for n in C.EX_SCALE_NS:
            row = {"kind": kind, "deg": deg, "label": kind_label(kind, deg), "n": n, "nets": 0, "sum_ml": 0.0, "sum_sl": 0.0}
            for seed in C.SWEEP_SEEDS[:3]:
                g, runs = _runs(kind, deg, n, seed)
                row["n"] = g.n
                row["nets"] += 1
                row["sum_ml"] += sum(m.cut for m, _s in runs) / len(runs)
                row["sum_sl"] += sum(s_.cut for _m, s_ in runs) / len(runs)
            rows.append(row)
    return rows
