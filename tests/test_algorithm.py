"""Kern: jede Einheit an von Hand gerechneten Mini-Netzen (DEMO-PLAYBOOK §1b)."""

import random

import pytest

import mlp_algorithm as A

from conftest import DREIECKE, PFAD6, FixedRandom, graph, wgraph


def test_graph_basics_by_hand():
    g = graph(*PFAD6)
    assert g.n == 6 and g.total == 6 and g.edge_total == 5 and g.adj[0] == ((1, 1),) and g.adj[2] == ((1, 1), (3, 1)) and g.vw == (1,) * 6
    w = wgraph(3, [(0, 1, 4), (1, 2, 2)], vw=(2, 1, 3))
    assert w.total == 6 and w.edge_total == 6 and w.adj[1] == ((0, 4), (2, 2))


def test_cut_value_and_weights_by_hand():
    g = graph(*DREIECKE)
    assert A.cut_value(g, [0, 0, 0, 1, 1, 1]) == 1 and A.cut_value(g, [0, 0, 1, 0, 1, 1]) == 5 and A.cut_value(g, [0] * 6) == 0
    w = wgraph(3, [(0, 1, 4), (1, 2, 2)], vw=(2, 1, 3))
    assert A.cut_value(w, [0, 1, 1]) == 4 and A.cut_value(w, [0, 0, 1]) == 2 and A.side_weights(w, [0, 1, 1]) == (2, 4)


def test_limit_and_balance_by_hand():
    assert A.limit(100, 3) == 52 and A.limit(100, 0) == 50 and A.limit(100, 10) == 55 and A.limit(6, 34) == 5 and A.limit(6, 0) == 3 and A.limit(7, 0) == 4 and A.limit(7, 10) == 4
    g = graph(*PFAD6)
    assert A.is_balanced(g, [0, 0, 0, 1, 1, 1], 0) and not A.is_balanced(g, [0, 0, 0, 0, 1, 1], 0) and A.is_balanced(g, [0, 0, 0, 0, 1, 1], 34) and A.is_balanced(g, [0, 0, 0, 0, 0, 1], 34) and not A.is_balanced(g, [0] * 6, 34)


def test_gains_by_hand():
    g = graph(*DREIECKE)
    assert A.gains(g, [0, 0, 1, 0, 1, 1]) == [0, 0, 3, 3, 0, 0]
    assert A.gains(g, [0, 0, 0, 1, 1, 1]) == [-2, -2, -1, -1, -2, -2]


def test_fm_pass_by_hand():
    g = graph(*DREIECKE)
    side, p = A.fm_pass(g, [0, 0, 1, 0, 1, 1], 34)
    assert side == [0, 0, 0, 1, 1, 1] and p.best == 2 and p.gain == 4 and (p.cut_before, p.cut_after) == (5, 1) and p.moves >= 2


def test_fm_pass_is_blocked_without_slack_and_leaves_the_start():
    g = graph(*DREIECKE)
    side, p = A.fm_pass(g, [0, 0, 1, 0, 1, 1], 0)
    assert side == [0, 0, 1, 0, 1, 1] and p.moves == 0 and p.best == 0 and p.gain == 0 and p.cut_after == 5


def test_fm_pass_respects_the_node_weights_of_the_destination():
    g = wgraph(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)], vw=(3, 1, 1, 3))
    side, p = A.fm_pass(g, [0, 0, 1, 1], 0)                                                   # Hälften 4 und 4, kein Spielraum: nichts darf sich bewegen
    assert side == [0, 0, 1, 1] and p.moves == 0
    side, p = A.fm_pass(g, [0, 1, 0, 1], 25)                                                   # 4 und 4, bis 5 erlaubt: Knoten 1 und 2 dürfen wechseln
    assert A.cut_value(g, side) <= A.cut_value(g, [0, 1, 0, 1]) and A.is_balanced(g, side, 25)


def test_fm_runs_until_no_gain():
    g = graph(*DREIECKE)
    r = A.fm(g, [0, 0, 1, 0, 1, 1], 34)
    assert r.cut == 1 and r.side == (0, 0, 0, 1, 1, 1) and r.passes[-1].gain == 0 and r.passes[0].gain == 4 and len(r.passes) == 2 and r.moves == sum(p.moves for p in r.passes)
    assert A.fm(g, [0, 0, 0, 1, 1, 1], 34).passes[0].gain == 0 and len(A.fm(g, [0, 0, 0, 1, 1, 1], 34).passes) == 1
    assert len(A.fm(g, [0, 0, 1, 0, 1, 1], 34, max_passes=1).passes) == 1


def test_random_bisection_with_a_fixed_source_and_with_heavy_nodes():
    g = graph(*PFAD6)
    assert A.random_bisection(g, FixedRandom()) == [0, 0, 0, 1, 1, 1]
    heavy = wgraph(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)], vw=(5, 1, 1, 1))
    side = A.random_bisection(heavy, FixedRandom())
    assert side == [0, 1, 1, 1] and A.is_balanced(heavy, side, 3)
    skewed = wgraph(3, [(0, 1, 1), (1, 2, 1)], vw=(1, 1, 6))
    side = A.random_bisection(skewed, FixedRandom())
    assert set(side) == {0, 1}                                                                 # keine erlaubte Teilung möglich: beide Hälften bleiben nichtleer
    rng = random.Random(3)
    for _ in range(20):
        s = A.random_bisection(g, rng)
        assert A.is_balanced(g, s, 3) and set(s) == {0, 1}


def test_heavy_edge_matching_by_hand():
    g = graph(*PFAD6)
    assert A.matching(g, range(6), 2) == (1, 0, 3, 2, 5, 4)
    assert A.matching(g, range(6), 1) == (-1,) * 6                                              # das gemeinsame Gewicht 2 überschreitet die Grenze 1
    assert A.matching(g, [5, 4, 3, 2, 1, 0], 2) == (1, 0, 3, 2, 5, 4)
    w = wgraph(3, [(0, 1, 1), (0, 2, 5), (1, 2, 2)])
    assert A.matching(w, range(3), 2) == (2, -1, 0)                                            # Knoten 0 wählt die schwerste Leitung (5) zu Knoten 2
    assert A.matching(w, [1, 0, 2], 2) == (-1, 2, 1)                                           # Knoten 1 wählt zuerst: die schwerste ist (1,2) mit 2
    tie = wgraph(3, [(0, 1, 2), (0, 2, 2)])
    assert A.matching(tie, range(3), 2) == (1, 0, -1)                                          # Gleichstand: der kleinste Nachbar


def test_contract_by_hand():
    w = wgraph(3, [(0, 1, 1), (0, 2, 5), (1, 2, 2)])
    c, cmap = A.contract(w, (2, -1, 0))
    assert cmap == (0, 1, 0) and c.vw == (2, 1) and c.adj == (((1, 3),), ((0, 3),))            # Leitungen 0-1 (1) und 2-1 (2) addieren sich, 0-2 (5) fällt weg
    g = graph(*PFAD6)
    c, cmap = A.contract(g, (1, 0, 3, 2, 5, 4))
    assert cmap == (0, 0, 1, 1, 2, 2) and c.vw == (2, 2, 2) and c.adj == (((1, 1),), ((0, 1), (2, 1)), ((1, 1),)) and c.total == g.total and c.edge_total == 2
    solo, cm = A.contract(g, (-1,) * 6)
    assert cm == tuple(range(6)) and solo == g


def test_coarsen_levels_by_hand():
    g = graph(*PFAD6)
    levels = A.coarsen(g, FixedRandom(), 3, 3)
    assert [lv.graph.n for lv in levels] == [6, 3] and levels[0].cmap is None and levels[1].cmap == (0, 0, 1, 1, 2, 2) and (levels[1].pairs, levels[1].visits) == (3, 6)
    path24 = graph(24, [(v, v + 1) for v in range(23)])
    levels = A.coarsen(path24, FixedRandom(), 2, 3)                                            # Grenze für das Gewicht eines Knotens: ⌊13 / 3⌋ = 4
    assert [lv.graph.n for lv in levels] == [24, 12, 6] and [lv.graph.vw for lv in levels] == [(1,) * 24, (2,) * 12, (4,) * 6] and sum(lv.visits for lv in levels) == 36
    assert len(A.coarsen(path24, FixedRandom(), 24, 3)) == 1 and len(A.coarsen(path24, FixedRandom(), 100, 3)) == 1


def test_coarsening_stalls_on_a_network_without_edges_and_respects_the_weight_cap():
    levels = A.coarsen(graph(3, []), FixedRandom(), 1, 3)
    assert len(levels) == 1
    star = graph(7, [(0, v) for v in range(1, 7)])
    sizes = [lv.graph.n for lv in A.coarsen(star, FixedRandom(), 3, 3)]
    assert sizes == [7, 6]                                                                     # die Grenze ⌊4 / 3⌋ → 2 lässt nur ein Paar zu: weiter paart nichts, das Vergröbern bricht ab
    capped = A.coarsen(graph(*PFAD6), FixedRandom(), 2, 0)                                       # Grenze max(2, limit // 3) = 2: Paare ja, Vierer nein
    assert [lv.graph.vw for lv in capped] == [(1,) * 6, (2, 2, 2)]


def test_project_members_and_reversibility():
    g = graph(*PFAD6)
    levels = A.coarsen(g, FixedRandom(), 3, 3)
    assert A.project([0, 1, 1], levels[1].cmap) == [0, 0, 1, 1, 1, 1]
    assert A.members(levels) == (tuple((v,) for v in range(6)), ((0, 1), (2, 3), (4, 5)))
    side = A.project([0, 1, 0], levels[1].cmap)
    assert A.cut_value(levels[1].graph, [0, 1, 0]) == A.cut_value(g, side)                      # der Schnitt bleibt beim Übertragen gleich


def test_multilevel_on_the_path_by_hand():
    path24 = graph(24, [(v, v + 1) for v in range(23)])
    r = A.multilevel(path24, 3, FixedRandom(), 2)
    assert [lv.graph.n for lv in r.levels] == [24, 12, 6] and r.side == (0,) * 12 + (1,) * 12 and r.cut == 1 and r.trials == (1,) * 5
    assert [(t.index, t.n, t.cut_projected, t.cut_refined) for t in r.trace] == [(0, 24, 1, 1), (1, 12, 1, 1), (2, 6, 1, 1)] and r.visits == 36 and r.work == r.moves + 36
    assert r.sides[0] == r.side and r.sides[1] == (0,) * 6 + (1,) * 6 and r.sides[2] == (0, 0, 0, 1, 1, 1)


def test_multilevel_without_refining_the_middle_levels_only_projects_there():
    path24 = graph(24, [(v, v + 1) for v in range(23)])
    r = A.multilevel(path24, 3, FixedRandom(), 2, refine="finest")
    assert r.trace[1].moves == 0 and r.trace[1].passes == 0 and r.trace[1].cut_projected == r.trace[1].cut_refined and r.trace[0].passes >= 1 and r.cut == 1
    each = A.multilevel(path24, 3, FixedRandom(), 2, refine="each")
    assert each.trace[1].passes >= 1 and each.trace[2].passes >= 1                              # die Anfangsteilung des gröbsten Netzes zählt ihre Durchläufe immer


def test_multilevel_is_balanced_deterministic_and_keeps_the_cut_through_projection():
    g = graph(*DREIECKE)
    r1 = A.multilevel(g, 10, random.Random(5), 2)
    r2 = A.multilevel(g, 10, random.Random(5), 2)
    assert r1 == r2 and A.is_balanced(g, r1.side, 10) and r1.cut == A.cut_value(g, r1.side)
    for i in range(len(r1.trace) - 1):
        assert r1.trace[i].cut_projected == r1.trace[i + 1].cut_refined


def test_single_level_uses_the_budget_and_returns_the_best_start():
    g = graph(*DREIECKE)
    r = A.single_level(g, 34, random.Random(2), 0)
    assert len(r.starts) == 1 and r.cut == min(r.starts) and A.is_balanced(g, r.side, 34)
    many = A.single_level(g, 34, random.Random(2), 40)
    assert len(many.starts) > 1 and many.moves >= 40 and many.cut == min(many.starts) == A.cut_value(g, many.side)
    few = A.single_level(g, 34, random.Random(2), 1)
    assert few.moves >= 1 and len(few.starts) >= 1


def test_coarsening_continues_when_a_step_saves_exactly_five_percent():
    g = graph(20, [(0, 1)])                                                                    # ein einziges Paar: 20 → 19 Knoten, genau 5 % gespart
    levels = A.coarsen(g, FixedRandom(), 8, 3)
    assert [lv.graph.n for lv in levels] == [20, 19] and levels[1].pairs == 1 and levels[1].visits == 20          # der nächste Schritt (19 → 19) bricht ab


def test_the_trace_runs_from_the_finest_level_to_the_coarsest_in_both_refine_modes():
    path24 = graph(24, [(v, v + 1) for v in range(23)])
    for mode in ("each", "finest"):
        r = A.multilevel(path24, 3, FixedRandom(), 2, refine=mode)
        assert [t.index for t in r.trace] == list(range(len(r.levels))) and [t.n for t in r.trace] == [lv.graph.n for lv in r.levels], mode


def test_single_level_keeps_the_first_of_several_equally_good_starts():
    g = graph(6, [])                                                                           # jeder Start hat Schnitt 0, aber eine andere Teilung
    rng = random.Random(3)
    first = A.fm(g, A.random_bisection(g, rng, 3), 3).side
    r = A.single_level(g, 3, random.Random(3), 50)
    assert len(r.starts) > 1 and set(r.starts) == {0} and r.side == first

