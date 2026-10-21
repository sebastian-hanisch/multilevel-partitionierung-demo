"""Gemeinsame Fixtures: Hilfsfunktionen und Mini-Netze, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Teure Experimente laufen einmal je Testlauf (`lru_cache`) und werden von Claims-, Verdikt- und Mutationstests gemeinsam genutzt."""

import sys
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FixedRandom:
    """Liefert immer fast 1: Fisher–Yates lässt die Reihenfolge unverändert (Identität), `random_bisection` füllt Hälfte 0 mit den ersten Knoten."""

    def random(self):
        return 0.999999


def graph(n, pairs):
    import mlp_algorithm as A
    return A.from_pairs(n, pairs)


def wgraph(n, edges, vw=None):
    """Gewichtetes Netz aus (u, v, Gewicht)-Tripeln und Knotengewichten."""
    import mlp_algorithm as A
    nb = [dict() for _ in range(n)]
    for u, v, w in edges:
        nb[u][v] = w
        nb[v][u] = w
    return A.WGraph(tuple(tuple(sorted(d.items())) for d in nb), tuple(vw or (1,) * n))


PFAD6 = (6, [(v, v + 1) for v in range(5)])
DREIECKE = (6, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)])


@pytest.fixture
def mini():
    return {"pfad6": PFAD6, "dreiecke": DREIECKE}


@lru_cache(maxsize=None)
def cached(name):
    """Einmal je Testlauf berechnete Experimente (Name der Funktion in mlp_experiments)."""
    import mlp_experiments as ex
    return getattr(ex, name)()
