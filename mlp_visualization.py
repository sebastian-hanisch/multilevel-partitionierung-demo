"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import mlp_algorithm as A
import mlp_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, xy, height=470):
    xs, ys = xy[:, 0], xy[:, 1]
    pad = 0.25 * max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    fig.update_xaxes(visible=False, range=[float(xs.min()) - pad, float(xs.max()) + pad])
    fig.update_yaxes(visible=False, range=[float(ys.min()) - pad, float(ys.max()) + pad])
    return _base(fig, height)


def level_xy(inst, levels, level):
    """Lage der Knoten einer Stufe: gewichteter Mittelpunkt ihrer Mitglieder auf der Karte des feinsten Netzes."""
    import numpy as np
    mem = A.members(levels)[level]
    return np.array([inst.xy[list(m)].mean(axis=0) for m in mem])


def build_level_map(inst, levels, level, side, height=470):
    """Karte einer Stufe: Knotengröße ~ Gewicht, Leitungsbreite ~ Gewicht, Farbe = Hälfte, rot gestrichelt die Leitungen zwischen den Hälften."""
    g = levels[level].graph
    xy = level_xy(inst, levels, level)
    fig = go.Figure()
    groups = {}
    for u in range(g.n):
        for v, w in g.adj[u]:
            if u < v:
                groups.setdefault((side[u] != side[v], min(w, 8)), []).append((u, v))
    cut_total = sum(w for u in range(g.n) for v, w in g.adj[u] if u < v and side[u] != side[v])
    for (is_cut, w), pairs in sorted(groups.items()):
        xs, ys = [], []
        for u, v in pairs:
            xs += [float(xy[u, 0]), float(xy[v, 0]), None]
            ys += [float(xy[u, 1]), float(xy[v, 1]), None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_CUT if is_cut else C.COLOR_EDGE, width=1.0 + 0.7 * w, dash="dot" if is_cut else None), hoverinfo="skip", showlegend=False))
    if cut_total:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=C.COLOR_CUT, width=3, dash="dot"), name=f"Schnitt ({cut_total})"))
    show_text = g.n <= 80 and g.total > g.n
    for half, color, name in ((0, C.COLOR_SIDE_A, "Hälfte 1"), (1, C.COLOR_SIDE_B, "Hälfte 2")):
        nodes = [v for v in range(g.n) if side[v] == half]
        sizes = [min(8 + 3.0 * g.vw[v] ** 0.5, 38) if g.n <= 80 else 5 + g.vw[v] ** 0.5 for v in nodes]
        fig.add_trace(go.Scatter(
            x=[float(xy[v, 0]) for v in nodes], y=[float(xy[v, 1]) for v in nodes], mode="markers+text" if show_text else "markers", marker=dict(size=sizes, color=color, line=dict(color="white", width=1)),
            text=[str(g.vw[v]) for v in nodes] if show_text else None, textfont=dict(size=9, color="#ffffff"), textposition="middle center",
            hovertext=[f"Gewicht {g.vw[v]}" for v in nodes], hoverinfo="text", name=name))
    return _map_layout(fig, xy, height)


def build_trace(r):
    """Schnitt je Stufe: nach dem Übertragen (grau) und nach dem Verfeinern (grün); links die gröbste, rechts die feinste Stufe."""
    t = list(reversed(r.trace))
    xs = [f"Stufe {x.index + 1}: {x.n}" for x in t]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[x.cut_projected for x in t], name="nach dem Übertragen", marker_color="#8a9a9c", hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(x=xs, y=[x.cut_refined for x in t], name="nach dem Verfeinern", marker_color=C.COLOR_BAR3, hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Schnitt (Gewicht der Leitungen zwischen den Hälften)", rangemode="tozero")
    return _base(fig, 340)


def build_runs(a):
    """Schnitt jedes Laufs: mehrstufig gegen einstufig bei gleichem Aufwand."""
    xs = list(range(1, len(a.ml) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=list(a.ml_cuts), name="mehrstufig", marker_color=C.COLOR_BAR3, hovertemplate="Lauf %{x}: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(x=xs, y=list(a.sl_cuts), name="einstufig (gleicher Aufwand)", marker_color=C.COLOR_BAR2, hovertemplate="Lauf %{x}: %{y}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Lauf", dtick=1)
    fig.update_yaxes(title_text="Schnitt", rangemode="tozero")
    return _base(fig, 340)


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------------


def build_exact(rows):
    ys = [round(100 * (r["checks"] - r["wrong"]) / r["checks"]) if r["checks"] else 0 for r in rows]
    fig = go.Figure(go.Bar(x=[r["label"] for r in rows], y=ys, marker_color=C.COLOR_BAR, showlegend=False, hovertemplate="%{x}: %{y} %<extra></extra>"))
    fig.update_yaxes(title_text="Prüfungen ohne Abweichung (%)", range=[0, 105])
    return _base(fig)


def build_compare(rows):
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[round(r["sum_ml"] / r["nets"], 1) for r in rows], name="mehrstufig", marker_color=C.COLOR_BAR3, hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=[round(r["sum_sl"] / r["nets"], 1) for r in rows], name="einstufig (gleicher Aufwand)", marker_color=C.COLOR_BAR2, hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Schnitt (Mittel über Netze und Läufe)", rangemode="tozero")
    return _base(fig, 400)


def build_shrink(rows):
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[round(r["mean_shrink"], 3) for r in rows], name="Mittel über alle Stufen", marker_color=C.COLOR_BAR, hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=[round(r["mean_first"], 3) for r in rows], name="erster Schritt", marker_color=C.COLOR_BAR2, hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Knotenzahl der gröberen / der feineren Stufe", range=[0, 1])
    return _base(fig)


def build_refine(rows):
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    for key, name, color in (("sum_projected", "nur übertragen", "#8a9a9c"), ("sum_finest", "Verfeinern nur auf der feinsten Stufe", C.COLOR_BAR2), ("sum_each", "Verfeinern auf jeder Stufe", C.COLOR_BAR3)):
        fig.add_trace(go.Bar(x=labels, y=[round(r[key] / r["nets"], 1) for r in rows], name=name, marker_color=color, hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Schnitt (Mittel über die Netze)", rangemode="tozero")
    return _base(fig, 400)


def build_scale(rows):
    fig = go.Figure()
    for kind, color in zip(dict.fromkeys(r["kind"] for r in rows), (C.COLOR_BAR3, C.COLOR_BAR)):
        sel = [r for r in rows if r["kind"] == kind]
        fig.add_trace(go.Scatter(x=[r["n"] for r in sel], y=[round(100 * (1 - r["sum_ml"] / r["sum_sl"]), 1) for r in sel], mode="lines+markers", name=sel[0]["label"], line=dict(color=color, width=3),
                                 hovertemplate="%{x} Standorte: %{y} %<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#8a9a9c", dash="dash", width=1.5))
    fig.update_xaxes(title_text="Standorte", dtick=None)
    fig.update_yaxes(title_text="Weniger Leitungen als einstufig (%)")
    return _base(fig)
