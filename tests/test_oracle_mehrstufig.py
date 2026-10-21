"""Orakel-Test multilevel-partitionierung-demo: Schnitt, Vergröbern (Gewichte der gröberen Netze), Matching, Fiduccia–Mattheyses und mehrstufiger Lauf gegen unabhängige Rechenwege (DEMO-PLAYBOOK §12).

Unabhängig: `networkx.cut_size` mit Gewichten, `networkx.quotient_graph` (gröbere Netze aus den Mitgliedergruppen des feinsten Netzes), die Aufzählung aller erlaubten Zweiteilungen bei höchstens 16 Knoten und
Eigenschaften des Matchings (gültig, gleiche Gewichte, maximal). Feste Netze (Seeds)."""

import itertools
import random

import pytest

import mlp_algorithm as A
import mlp_scenario as S

nx = pytest.importorskip("networkx")

from conftest import wgraph  # noqa: E402


def _nets(count=24, base=7000, n=40):
    kinds = ("geometric", "grid", "ring", "scalefree", "random", "verbund")
    for k in range(count):
        inst = S.make_instance(kinds[k % len(kinds)], n + 5 * (k % 3), 4.0, 2, base + k)
        yield inst, A.from_pairs(inst.n, inst.pairs)


def _nx(g):
    h = nx.Graph()
    h.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in g.adj[u]:
            if u < v:
                h.add_edge(u, v, weight=w)
    return h


def _brute_best(g, eps):
    limit = A.limit(g.total, eps)
    best = None
    for mask in range(1, 1 << (g.n - 1)):
        side = [(mask >> v) & 1 for v in range(g.n - 1)] + [0]
        w1 = sum(g.vw[v] for v in range(g.n) if side[v])
        if w1 > limit or g.total - w1 > limit:
            continue
        c = A.cut_value(g, side)
        best = c if best is None else min(best, c)
    return best


def test_cut_value_equals_networkx_on_random_partitions_and_weights():
    rng = random.Random(4)
    for inst, g in _nets(12):
        h = _nx(g)
        for _ in range(8):
            side = [1 if rng.random() < 0.4 else 0 for _ in range(g.n)]
            if len(set(side)) < 2:
                continue
            assert A.cut_value(g, side) == nx.cut_size(h, {v for v in range(g.n) if side[v]}, weight="weight")


def test_every_coarse_level_equals_the_networkx_quotient_of_its_member_groups():
    for inst, g in _nets(12, 7100, 36):
        levels = A.coarsen(g, random.Random(inst.n), 8, 3)
        mem = A.members(levels)
        h = _nx(g)
        for lvl in range(1, len(levels)):
            q = nx.quotient_graph(h, [set(m) for m in mem[lvl]], relabel=False)
            groups = {frozenset(m): i for i, m in enumerate(mem[lvl])}
            cg = levels[lvl].graph
            assert cg.vw == tuple(len(m) for m in mem[lvl]) and cg.total == g.total
            for a_, b_, d in q.edges(data=True):
                u, v = groups[frozenset(a_)], groups[frozenset(b_)]
                assert dict(cg.adj[u])[v] == d["weight"]
            assert cg.edge_total == sum(d["weight"] for _a, _b, d in q.edges(data=True))


def test_the_matching_is_valid_symmetric_maximal_and_respects_the_cap():
    for inst, g in _nets(12, 7200, 36):
        rng = random.Random(inst.n + 3)
        order = S._shuffle(range(g.n), rng)
        for cap in (2, 3, 100):
            mate = A.matching(g, order, cap)
            for v, u in enumerate(mate):
                if u >= 0:
                    assert mate[u] == v and u != v and dict(g.adj[v]).get(u) and g.vw[v] + g.vw[u] <= cap
            free = [v for v in range(g.n) if mate[v] < 0]
            for v in free:
                assert not any(u in free and g.vw[v] + g.vw[u] <= cap for u, _w in g.adj[v])          # kein freies Paar ließe sich noch verbinden


def test_the_matching_takes_the_heaviest_free_neighbour_against_a_reference():
    rng = random.Random(9)
    for _ in range(30):
        n = rng.randint(4, 12)
        edges = [(u, v, rng.randint(1, 6)) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.5]
        g = wgraph(n, edges)
        order = S._shuffle(range(n), rng)
        mate = [-1] * n
        for v in order:                                                                          # Bezugsrechnung: sortierte Nachbarn
            if mate[v] >= 0:
                continue
            cand = sorted(((-w, u) for u, w in g.adj[v] if mate[u] < 0))
            if cand:
                mate[v] = cand[0][1]
                mate[cand[0][1]] = v
        assert A.matching(g, order, 100) == tuple(mate)


def test_projection_keeps_the_cut_and_the_weights_on_random_partitions():
    rng = random.Random(6)
    for inst, g in _nets(12, 7300, 40):
        levels = A.coarsen(g, random.Random(inst.n), 10, 3)
        for lvl in range(1, len(levels)):
            coarse = levels[lvl].graph
            side = [rng.randint(0, 1) for _ in range(coarse.n)]
            fine = A.project(side, levels[lvl].cmap)
            assert A.cut_value(levels[lvl - 1].graph, fine) == A.cut_value(coarse, side) and A.side_weights(levels[lvl - 1].graph, fine) == A.side_weights(coarse, side)


def test_no_method_beats_the_optimum_and_every_result_is_balanced_on_small_networks():
    for k in range(10):
        inst = S.make_instance(("random", "geometric", "ring", "scalefree", "verbund")[k % 5], 14 + k % 3, 3.0, 2, 7400 + k)
        g = A.from_pairs(inst.n, inst.pairs)
        for eps in (3, 10):
            opt = _brute_best(g, eps)
            rng = random.Random(k)
            r = A.fm(g, A.random_bisection(g, rng, eps), eps)
            m = A.multilevel(g, eps, rng, 6)
            s = A.single_level(g, eps, rng, 200)
            for res in (r, m, s):
                assert A.is_balanced(g, res.side, eps) and res.cut >= opt, (inst.kind, inst.n, eps)
            assert A.cut_value(g, m.side) == m.cut


def test_fm_gains_equal_the_difference_of_recounted_cuts():
    for inst, g in _nets(12, 7500, 40):
        side = A.random_bisection(g, random.Random(inst.n))
        new, p = A.fm_pass(g, side, 3)
        assert p.cut_before == A.cut_value(g, side) and p.cut_after == A.cut_value(g, new) and p.cut_before - p.cut_after == p.gain >= 0 and A.is_balanced(g, new, 3)
        r = A.fm(g, side, 3)
        assert r.cut == A.cut_value(g, r.side) <= p.cut_before and A.is_balanced(g, r.side, 3)
