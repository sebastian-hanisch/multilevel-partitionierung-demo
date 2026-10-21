"""Randfälle der Kernroutinen: kleinste Netze, Netz ohne Leitungen, schwere Knoten, Gleichstände, Grenzen inklusive (DEMO-PLAYBOOK §12)."""

import dataclasses
import random

import pytest

import mlp_algorithm as A
import mlp_constants as C
import mlp_evaluation as ev

from conftest import FixedRandom, graph, wgraph


def test_two_and_three_nodes():
    g = graph(2, [(0, 1)])
    assert A.limit(2, 3) == 1 and A.limit(2, 100) == 1 and A.limit(3, 100) == 2 and A.is_balanced(g, [0, 1], 3) and not A.is_balanced(g, [0, 0], 3) and A.cut_value(g, [0, 1]) == 1
    r = A.multilevel(g, 3, random.Random(1), 8)
    assert len(r.levels) == 1 and r.cut == 1 and set(r.side) == {0, 1} and r.trace[0].n == 2
    tri = graph(3, [(0, 1), (1, 2), (0, 2)])
    r = A.multilevel(tri, 10, random.Random(1), 8)
    assert r.cut == 2 and A.is_balanced(tri, r.side, 10)


def test_a_network_without_edges_has_nothing_to_gain():
    g = graph(6, [])
    assert A.gains(g, [0, 0, 0, 1, 1, 1]) == [0] * 6 and A.matching(g, range(6), 100) == (-1,) * 6
    r = A.fm(g, A.random_bisection(g, random.Random(2)), 3)
    assert r.cut == 0 and len(r.passes) == 1 and r.passes[0].gain == 0
    ml = A.multilevel(g, 3, random.Random(2), 2)
    assert len(ml.levels) == 1 and ml.cut == 0


def test_isolated_nodes_do_not_break_matching_or_projection():
    g = graph(5, [(0, 1), (1, 2)])
    mate = A.matching(g, range(5), 100)
    assert mate == (1, 0, -1, -1, -1)
    c, cmap = A.contract(g, mate)
    assert cmap == (0, 0, 1, 2, 3) and c.vw == (2, 1, 1, 1) and c.adj[0] == ((1, 1),) and c.adj[2] == ()


def test_heavy_nodes_block_moves_but_never_unbalance_a_balanced_start():
    g = wgraph(4, [(0, 1, 5), (1, 2, 1), (2, 3, 5)], vw=(4, 1, 1, 4))
    side, p = A.fm_pass(g, [0, 0, 1, 1], 10)
    assert A.is_balanced(g, side, 10) and A.cut_value(g, side) <= 1
    r = A.fm(g, [0, 1, 0, 1], 10)
    assert A.is_balanced(g, r.side, 10) and r.cut <= A.cut_value(g, [0, 1, 0, 1])


def test_equal_gains_are_resolved_by_the_smaller_node_number():
    g = graph(4, [(0, 1), (2, 3)])
    side, p = A.fm_pass(g, [0, 1, 0, 1], 100)
    assert p.best >= 1 and A.cut_value(g, side) == 0 or A.cut_value(g, side) < A.cut_value(g, [0, 1, 0, 1])


def test_the_weight_cap_allows_exactly_the_cap():
    g = graph(4, [(0, 1), (2, 3), (1, 2)])
    assert A.matching(g, range(4), 2) == (1, 0, 3, 2) and A.matching(g, range(4), 1) == (-1,) * 4
    w = wgraph(3, [(0, 1, 3), (1, 2, 1)], vw=(3, 3, 1))
    assert A.matching(w, range(3), 6) == (1, 0, -1) and A.matching(w, range(3), 5) == (-1, 2, 1)                  # 3 + 3 = 6 passt gerade, bei 5 nicht


def test_single_level_with_zero_budget_makes_exactly_one_start():
    g = graph(6, [(v, v + 1) for v in range(5)])
    r = A.single_level(g, 3, FixedRandom(), 0)
    assert len(r.starts) == 1 and r.cut == 1


def test_validate_reports_every_kind_of_problem_on_tampered_analyses():
    a = ev.analyse(ev.Settings(kind="geometric", n=64, seed=3))
    assert ev.validate(a) == []
    m = a.ml[0]
    unbalanced = dataclasses.replace(m, side=(0,) * a.n)
    assert any("nicht ausgewogen" in p for p in ev.validate(dataclasses.replace(a, ml=(unbalanced,) + a.ml[1:])))
    wrong_cut = dataclasses.replace(m, cut=m.cut + 1)
    assert any("Nachzählung" in p for p in ev.validate(dataclasses.replace(a, ml=(wrong_cut,) + a.ml[1:])))
    lv = m.levels[1]
    heavy = dataclasses.replace(lv, graph=dataclasses.replace(lv.graph, vw=(lv.graph.vw[0] + 1,) + lv.graph.vw[1:]))
    assert any("Gesamtgewicht" in p for p in ev.validate(dataclasses.replace(a, ml=(dataclasses.replace(m, levels=(m.levels[0], heavy) + m.levels[2:]),) + a.ml[1:])))
    tr = list(m.trace)
    tr[0] = dataclasses.replace(tr[0], cut_projected=tr[1].cut_refined + 1, cut_refined=m.cut)
    assert any("Übertragen" in p for p in ev.validate(dataclasses.replace(a, ml=(dataclasses.replace(m, trace=tuple(tr)),) + a.ml[1:])))
    tr = list(m.trace)
    tr[0] = dataclasses.replace(tr[0], cut_projected=tr[1].cut_refined, cut_refined=m.cut + 1)
    assert any("vergrößert" in p or "Nachzählung" in p for p in ev.validate(dataclasses.replace(a, ml=(dataclasses.replace(m, trace=tuple(tr)),) + a.ml[1:])))
    s = a.sl[0]
    assert any("einstufige Teilung" in p for p in ev.validate(dataclasses.replace(a, sl=(dataclasses.replace(s, cut=s.cut + 1),) + a.sl[1:])))
    assert any("einstufige Teilung" in p for p in ev.validate(dataclasses.replace(a, sl=(dataclasses.replace(s, side=(0,) * a.n),) + a.sl[1:])))


def test_the_analysis_properties():
    a = ev.analyse(ev.Settings(**C.PRESETS["Zufallsnetz: ein kleiner Vorteil"]))
    assert len(a.ml) == len(a.sl) == C.RUNS and a.n == 200 and a.m == a.inst.m and a.ml_cuts == tuple(r.cut for r in a.ml) and a.sl_cuts == tuple(r.cut for r in a.sl)
    assert a.ml_mean == pytest.approx(sum(a.ml_cuts) / C.RUNS) and a.sl_mean == pytest.approx(sum(a.sl_cuts) / C.RUNS) and sum(a.wins) == C.RUNS and a.first is a.ml[0]
    assert len(a.shrink) == len(a.first.levels) - 1 and all(0.5 <= x < 1 for x in a.shrink)
    assert all(m.work <= s.moves or len(s.starts) == 1 for m, s in zip(a.ml, a.sl))                                  # das einstufige Verfahren nutzt das Budget (oder macht einen Start)


def test_the_analysis_is_deterministic_and_the_runs_differ():
    s = ev.Settings(kind="random", n=80, seed=5)
    a1, a2 = ev.analyse(s), ev.analyse(s)
    assert a1.ml == a2.ml and a1.sl == a2.sl and len(set(a1.ml_cuts + a1.sl_cuts)) > 1
    assert ev.run_rng(5, 0).random() != ev.run_rng(5, 1).random() and ev.run_rng(5, 0).random() != ev.run_rng(5, 0, 1).random()


def test_boundaries_of_the_regler_range():
    for kind in C.KINDS:
        for n, eps, coarse, refine in ((C.N_MIN, 1, C.COARSE_MAX, "each"), (C.N_MAX, 10, C.COARSE_MIN, "finest")):
            a = ev.analyse(ev.Settings(kind=kind, n=n, eps=eps, coarse=coarse, refine=refine))
            assert ev.validate(a) == [] and a.n >= n - 20, (kind, n)
    a = ev.analyse(ev.Settings(kind="random", n=C.N_MIN, coarse=C.COARSE_MAX))
    assert len(a.first.levels) == 1 and a.shrink == ()
