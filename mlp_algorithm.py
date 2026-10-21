"""Kern: gewichtete Netze, Fiduccia–Mattheyses mit Gewichten, Vergröbern (Heavy-Edge-Matching), mehrstufiges Teilen und das einstufige Verfahren als Vergleich.

**Aufgabe:** Die Standorte in zwei Hälften teilen, die höchstens um einen Spielraum ε von der halben Gesamtgröße abweichen, mit möglichst wenigen Leitungen (Gewichten) zwischen den Hälften.

**Einstufig:** Von einer zufälligen Anordnung aus verbessert Fiduccia–Mattheyses (FM): Man verschiebt wiederholt den freien Standort mit dem größten Gewinn (bei Gleichstand der kleinste), sperrt ihn, und führt den besten Anfang
der Zugfolge aus; so lange, bis ein Durchlauf nichts mehr gewinnt. Standorte haben Gewichte, die Hälften dürfen höchstens ⌈(1 + ε) W / 2⌉ wiegen.

**Mehrstufig** (Karypis, Kumar; Hendrickson, Leland): 1. **Vergröbern:** Standorte in zufälliger Reihenfolge besuchen und jeden mit dem Nachbarn verbinden, zu dem die schwerste Leitung führt (Heavy-Edge-Matching);
verbundene Paare werden ein Standort (Gewichte addieren sich), parallele Leitungen addieren sich. Das wiederholt man, bis das Netz klein ist. 2. **Teilen** des gröbsten Netzes: beste von einigen zufälligen Anordnungen
nach FM. 3. **Verfeinern:** Die Teilung stufenweise auf das feinere Netz übertragen (jeder Standort erbt die Hälfte seines Paares; der Schnitt bleibt gleich) und mit FM verbessern."""

import math
from dataclasses import dataclass

import mlp_constants as C
import mlp_scenario as S


@dataclass(frozen=True)
class WGraph:
    adj: tuple              # je Knoten ((Nachbar, Gewicht), ...) aufsteigend nach Nachbar
    vw: tuple               # Knotengewichte

    @property
    def n(self):
        return len(self.vw)

    @property
    def total(self):
        return sum(self.vw)

    @property
    def edge_total(self):
        return sum(w for a in self.adj for _u, w in a) // 2


def from_pairs(n, pairs):
    """Netz mit Knotengewicht 1 und Leitungsgewicht 1."""
    nb = [dict() for _ in range(n)]
    for u, v in pairs:
        nb[u][v] = 1
        nb[v][u] = 1
    return WGraph(tuple(tuple(sorted(d.items())) for d in nb), (1,) * n)


def cut_value(g, side):
    return sum(w for u in range(g.n) for v, w in g.adj[u] if u < v and side[u] != side[v])


def limit(total, eps):
    """Größtes erlaubtes Gewicht einer Hälfte: ⌈(1 + ε) · W / 2⌉ bei ε in Prozent, höchstens W − 1 (beide Hälften bleiben nichtleer)."""
    return min(int(math.ceil((1 + eps / 100.0) * total / 2 - 1e-9)), max(total - 1, 1))


def side_weights(g, side):
    w = [0, 0]
    for v in range(g.n):
        w[side[v]] += g.vw[v]
    return tuple(w)


def is_balanced(g, side, eps):
    return max(side_weights(g, side)) <= limit(g.total, eps)


# --- Fiduccia–Mattheyses mit Gewichten --------------------------------------------------------------------------------------------------------------


def gains(g, side):
    """Gewinn je Knoten: Leitungsgewicht nach außen minus nach innen (so viel sinkt der Schnitt, wenn der Knoten die Hälfte wechselt)."""
    return [sum(w if side[u] != side[v] else -w for u, w in g.adj[v]) for v in range(g.n)]


@dataclass(frozen=True)
class FMPass:
    moves: int              # Zahl der Züge (jeder Knoten höchstens einmal)
    best: int               # Länge des besten Anfangs der Zugfolge (0: nichts ausgeführt)
    gain: int
    cut_before: int
    cut_after: int


def fm_pass(g, side, eps):
    """Ein Durchlauf von FM auf einer Kopie: gibt (neue Hälften, Durchlauf) zurück."""
    n = g.n
    side = list(side)
    maxw = limit(g.total, eps)
    wt = list(side_weights(g, side))
    gain = gains(g, side)
    locked = [False] * n
    start_cut = cut = cut_value(g, side)
    best_cut, best_i = cut, 0
    moved = []
    for _ in range(n):
        pick, pg = -1, None
        for v in range(n):
            if locked[v] or wt[1 - side[v]] + g.vw[v] > maxw:
                continue
            if pg is None or gain[v] > pg:
                pick, pg = v, gain[v]
        if pick < 0:
            break
        v = pick
        cut -= gain[v]
        s = side[v]
        wt[s] -= g.vw[v]
        wt[1 - s] += g.vw[v]
        side[v] = 1 - s
        locked[v] = True
        gain[v] = -gain[v]
        for u, w in g.adj[v]:
            if not locked[u]:
                gain[u] += -2 * w if side[u] == side[v] else 2 * w
        moved.append(v)
        if cut < best_cut:
            best_cut, best_i = cut, len(moved)
    for v in reversed(moved[best_i:]):
        side[v] = 1 - side[v]
    return side, FMPass(len(moved), best_i, start_cut - best_cut, start_cut, best_cut)


@dataclass(frozen=True)
class Refined:
    side: tuple
    cut: int
    passes: tuple           # FMPass je Durchlauf (der letzte ohne Gewinn)
    moves: int


def fm(g, side, eps, max_passes=C.MAX_PASSES):
    """FM bis ein Durchlauf nichts mehr gewinnt (oder `max_passes` Durchläufe)."""
    side = list(side)
    passes = []
    for _ in range(max_passes):
        new, p = fm_pass(g, side, eps)
        passes.append(p)
        if p.gain <= 0:
            break
        side = new
    return Refined(tuple(side), cut_value(g, side), tuple(passes), sum(p.moves for p in passes))


def random_bisection(g, rng, eps=C.DEFAULT_EPS):
    """Zufällige Anordnung: Knoten in zufälliger Reihenfolge der Hälfte 0 zuweisen, solange sie höchstens das erlaubte Gewicht hat und noch nicht die Hälfte des Gesamtgewichts erreicht; reicht das nicht für
    die andere Hälfte, kommen weitere passende Knoten dazu. So sind beide Hälften erlaubt, wenn das überhaupt möglich ist (beide nichtleer)."""
    order = S._shuffle(range(g.n), rng)
    maxw = limit(g.total, eps)
    side = [1] * g.n
    w = 0
    for v in order:
        if w >= g.total / 2 and w > 0:
            break
        if w + g.vw[v] <= maxw or w == 0:
            side[v] = 0
            w += g.vw[v]
    for v in order:
        if g.total - w <= maxw:
            break
        if side[v] == 1 and w + g.vw[v] <= maxw:
            side[v] = 0
            w += g.vw[v]
    return side


# --- Vergröbern --------------------------------------------------------------------------------------------------------------------------------------


def matching(g, order, cap):
    """Heavy-Edge-Matching: Knoten in der Reihenfolge `order` besuchen; ein noch freier Knoten wird mit dem freien Nachbarn verbunden, zu dem die schwerste Leitung führt (bei Gleichstand der kleinste), sofern das
    gemeinsame Gewicht höchstens `cap` ist. Gibt für jeden Knoten seinen Partner zurück (-1: allein)."""
    mate = [-1] * g.n
    for v in order:
        if mate[v] >= 0:
            continue
        best, bu = None, -1
        for u, w in g.adj[v]:
            if mate[u] < 0 and g.vw[v] + g.vw[u] <= cap and (best is None or w > best):
                best, bu = w, u
        if bu >= 0:
            mate[v] = bu
            mate[bu] = v
    return tuple(mate)


def contract(g, mate):
    """Verbundene Paare zu einem Knoten zusammenziehen: gibt (gröberes Netz, Abbildung der Knoten auf die gröberen) zurück. Die Nummern der gröberen Knoten folgen dem kleinsten Mitglied."""
    cmap = [-1] * g.n
    c = 0
    for v in range(g.n):
        if cmap[v] >= 0:
            continue
        cmap[v] = c
        if mate[v] >= 0:
            cmap[mate[v]] = c
        c += 1
    nb = [dict() for _ in range(c)]
    vw = [0] * c
    for v in range(g.n):
        vw[cmap[v]] += g.vw[v]
        for u, w in g.adj[v]:
            a, b = cmap[v], cmap[u]
            if a != b:
                nb[a][b] = nb[a].get(b, 0) + w
    return WGraph(tuple(tuple(sorted(d.items())) for d in nb), tuple(vw)), tuple(cmap)


@dataclass(frozen=True)
class Level:
    graph: WGraph
    cmap: tuple | None      # Abbildung der Knoten der feineren Stufe auf diese Stufe (None bei der feinsten)
    pairs: int              # verbundene Paare im Schritt, der diese Stufe erzeugte
    visits: int             # besuchte Knoten dieses Schritts (Aufwand)


def coarsen(g, rng, coarse_n, eps):
    """Stufen vom feinsten Netz an: so lange vergröbern, bis höchstens `coarse_n` Knoten übrig sind oder ein Schritt weniger als 5 % der Knoten spart."""
    levels = [Level(g, None, 0, 0)]
    cap = max(2, limit(g.total, eps) // 3)
    while levels[-1].graph.n > coarse_n:
        cur = levels[-1].graph
        order = S._shuffle(range(cur.n), rng)
        mate = matching(cur, order, cap)
        coarse, cmap = contract(cur, mate)
        if coarse.n > C.STALL * cur.n:
            break
        levels.append(Level(coarse, cmap, sum(1 for x in mate if x >= 0) // 2, cur.n))
    return tuple(levels)


def project(side, cmap):
    """Teilung der gröberen Stufe auf die feinere übertragen: jeder Knoten erbt die Hälfte seines gröberen Knotens."""
    return [side[c] for c in cmap]


def members(levels):
    """Für jede Stufe die Mitglieder (Knoten des feinsten Netzes) jedes Knotens."""
    out = [tuple((v,) for v in range(levels[0].graph.n))]
    for lv in levels[1:]:
        groups = [[] for _ in range(lv.graph.n)]
        for v, c in enumerate(lv.cmap):
            groups[c].extend(out[-1][v])
        out.append(tuple(tuple(sorted(x)) for x in groups))
    return tuple(out)


# --- Mehrstufig und einstufig -----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelTrace:
    index: int              # 0 = feinste Stufe
    n: int
    edge_total: int
    cut_projected: int      # Schnitt nach dem Übertragen von der gröberen Stufe (beim gröbsten: der beste Anfangswert)
    cut_refined: int        # Schnitt nach FM
    moves: int
    passes: int


@dataclass(frozen=True)
class MLResult:
    side: tuple
    cut: int
    levels: tuple           # Level, feinste zuerst
    trace: tuple            # LevelTrace, feinste zuerst
    sides: tuple            # die Teilung jeder Stufe nach dem Verfeinern, feinste zuerst
    trials: tuple           # Schnitte der Anfangsteilungen auf dem gröbsten Netz
    moves: int              # Züge von FM (alle Stufen und Anfangsteilungen)
    visits: int             # besuchte Knoten beim Vergröbern

    @property
    def work(self):
        return self.moves + self.visits


def multilevel(g, eps, rng, coarse_n=C.DEFAULT_COARSE, trials=C.INITIAL_TRIALS, refine="each"):
    """Ein mehrstufiger Lauf. `refine`: "each" verfeinert auf jeder Stufe, "finest" nur auf der feinsten (dazwischen wird nur übertragen)."""
    levels = coarsen(g, rng, coarse_n, eps)
    coarsest = levels[-1].graph
    best, cuts, moves = None, [], 0
    for _ in range(trials):
        r = fm(coarsest, random_bisection(coarsest, rng, eps), eps)
        moves += r.moves
        cuts.append(r.cut)
        if best is None or (not is_balanced(coarsest, r.side, eps), r.cut) < (not is_balanced(coarsest, best.side, eps), best.cut):
            best = r
    side = list(best.side)
    sides = [tuple(side)]
    trace = [LevelTrace(len(levels) - 1, coarsest.n, coarsest.edge_total, best.cut, best.cut, best.moves, len(best.passes))]
    for li in range(len(levels) - 1, 0, -1):
        fine = levels[li - 1].graph
        side = project(side, levels[li].cmap)
        projected = cut_value(fine, side)
        if refine == "each" or li == 1:
            r = fm(fine, side, eps, C.LEVEL_PASSES if li > 1 else C.MAX_PASSES)
            side, moves = list(r.side), moves + r.moves
            trace.append(LevelTrace(li - 1, fine.n, fine.edge_total, projected, r.cut, r.moves, len(r.passes)))
            sides.append(tuple(side))
        else:
            trace.append(LevelTrace(li - 1, fine.n, fine.edge_total, projected, projected, 0, 0))
            sides.append(tuple(side))
    trace.reverse()
    sides.reverse()
    return MLResult(tuple(side), cut_value(g, side), levels, tuple(trace), tuple(sides), tuple(cuts), moves, sum(lv.visits for lv in levels))


@dataclass(frozen=True)
class SLResult:
    side: tuple
    cut: int
    starts: tuple           # Schnitte der einzelnen Starts
    moves: int


def single_level(g, eps, rng, budget):
    """Einstufig: zufällige Anordnungen mit FM verbessern, bis die Züge das Budget erreichen (mindestens ein Start); gibt den besten Start zurück."""
    best, starts, moves = None, [], 0
    while moves < budget or not starts:
        r = fm(g, random_bisection(g, rng, eps), eps)
        moves += r.moves
        starts.append(r.cut)
        if best is None or r.cut < best.cut:
            best = r
    return SLResult(best.side, best.cut, tuple(starts), moves)
