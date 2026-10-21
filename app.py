"""Mehrstufig partitionieren: Vergröbern, Teilen, Verfeinern - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Reihe „Schnitte und Zusammenhang“: Warum vergröbert man ein Netz erst, teilt das kleine und verfeinert dann? Heavy-Edge-Matching, Teilen des gröbsten Netzes, Verfeinern mit Fiduccia–Mattheyses
auf jeder Stufe, verglichen mit dem einstufigen Verfahren bei gleichem Aufwand.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import mlp_algorithm as A
import mlp_constants as C
import mlp_evaluation as ev
import mlp_experiments as ex
from mlp_layout import dunkles_design, kopf_css
from mlp_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mlp_visualization import (
    build_compare,
    build_exact,
    build_level_map,
    build_refine,
    build_runs,
    build_scale,
    build_shrink,
    build_trace,
)

st.set_page_config(page_title="Mehrstufig partitionieren: Vergröbern, Teilen, Verfeinern – Sebastian Hanisch", layout="wide")

_i, _dec = ev._i, ev._dec


def _show(verdict):
    kind, text = verdict
    {"success": st.success, "info": st.info, "warning": st.warning}[kind](text)


@st.cache_data(show_spinner=False, max_entries=6)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _exactness():
    return ex.exactness_rows()


@st.cache_data(show_spinner=False)
def _compare():
    return ex.compare_rows()


@st.cache_data(show_spinner=False)
def _shrink():
    return ex.shrink_rows()


@st.cache_data(show_spinner=False)
def _refine():
    return ex.refine_rows()


@st.cache_data(show_spinner=False)
def _scale():
    return ex.scale_rows()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mlp_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Mehrstufig partitionieren: Vergröbern, Teilen, Verfeinern")
st.markdown(
    """
**Warum vergröbert man ein großes Netz erst, teilt das kleine und verfeinert dann?** Das einstufige Verfahren (Fiduccia–Mattheyses aus **bisektion-demo**) startet von einer Zufallsanordnung und verbessert sie Zug um Zug; in einem großen Netz
bleibt es leicht in einem **lokalen Optimum** hängen, weil es nur die unmittelbare Umgebung sieht. Das **mehrstufige** Verfahren zieht erst Paare schwer verbundener Standorte zusammen (**Vergröbern**, mehrmals), teilt das
gröbste Netz, überträgt die Teilung stufenweise auf das feinere Netz (der Schnitt bleibt dabei gleich) und **verfeinert** auf jeder Stufe. Die Demo vergleicht beide **bei gleichem Aufwand**.

Auffällig: Der Vorteil hängt stark vom Netz ab. Auf **Kartennetzen** (räumliche Struktur) ist der Schnitt des mehrstufigen Verfahrens bei 256 Standorten um knapp ein Drittel kleiner, in vielen anderen Netzen kaum oder gar nicht; auf **skalenfreien Netzen**
mit Hubs schrumpft das Netz nur langsam. Ehrlich: Die Demo baut die einfachen Bausteine und erhebt **keinen Anspruch, Systeme wie METIS zu erreichen**.

Das ist das sechste Stück der Reihe **Schnitte und Zusammenhang**; die Verfahren auf einer Stufe zeigt **bisektion-demo**, den Eigenvektor als Alternative **spektrale-schnitte-demo**, Paare zusammenzuziehen (Matching) **hopcroft-karp-demo** und
**blossom-demo**. Weiter unten: die Experimente **🔬**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So arbeitet das mehrstufige Verfahren", expanded=True):
    st.markdown(
        """
1. **Vergröbern:** Man besucht die Standorte in zufälliger Reihenfolge; ein noch freier Standort wird mit dem freien Nachbarn verbunden, zu dem die **schwerste** Leitung führt (**Heavy-Edge-Matching**). Jedes Paar wird ein Standort mit der Summe der Gewichte, parallele Leitungen addieren sich. Das wiederholt man, bis das Netz klein ist.
2. **Teilen:** Das gröbste Netz wird in zwei Hälften geteilt, deren Gewicht höchstens um den **Spielraum ε** vom Ideal abweicht: die beste von einigen zufälligen Anordnungen nach FM.
3. **Übertragen:** Jeder Standort erbt die Hälfte seines gröberen Standorts. Der Schnitt ändert sich dabei **nicht**.
4. **Verfeinern:** Auf der feineren Stufe verbessert FM die Teilung: Man verschiebt den freien Standort mit dem größten Gewinn, sperrt ihn und führt den besten Anfang der Zugfolge aus. Dann geht es eine Stufe feiner.
5. **Gleicher Aufwand:** Das einstufige Verfahren darf so viele Starts machen, bis seine Züge den **Aufwand des mehrstufigen Laufs** (Züge plus besuchte Standorte beim Vergröbern) erreichen.

| Begriff | Bedeutung |
|---|---|
| **Heavy-Edge-Matching** | verbindet jeden Standort mit dem Nachbarn, zu dem die schwerste Leitung führt |
| **Stufe** | ein Netz in der Folge vom feinsten zum gröbsten; die Knotenzahl sinkt von Stufe zu Stufe |
| **Gewicht** | Zahl der ursprünglichen Standorte in einem gröberen Standort bzw. Zahl der ursprünglichen Leitungen zwischen zwei gröberen |
| **Spielraum ε** | wie weit eine Hälfte vom halben Gesamtgewicht abweichen darf |
| **Aufwand** | Züge von FM plus besuchte Standorte beim Vergröbern (verschiedene Einheiten, gleich gezählt) |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    kind = st.selectbox("Art des Netzes", list(C.KINDS), format_func=C.KIND_NAMES.get, key="kind_select", help=C.KIND_HELP.get(st.session_state.get("kind_select", C.DEFAULT_KIND), ""))
    n = st.slider("Standorte n", *bounds("n_slider"), key="n_slider", help="Wie viele Standorte das Netz hat (das Gitter rundet auf ein Rechteck). Ab etwa 150 Standorten zeigt sich der Vorteil des Vergröberns.")
    deg = st.slider("Mittlerer Grad", *bounds("deg_slider"), key="deg_slider", step=0.5, help="Leitungen je Standort im Mittel; wirkt beim Zufallsnetz und innerhalb der Regionen des Verbundnetzes.")
    links = st.slider("Leitungen zwischen je zwei Regionen", *bounds("links_slider"), key="links_slider", help="Wirkt nur beim Verbundnetz.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt das Netz und die Reihenfolgen beim Vergröbern und die Anordnungen fest.")
    st.button("🎲 Neues Netz erzeugen", width="stretch", on_click=randomize_seed)
    st.caption(C.KIND_REGLER[kind])
    st.markdown("**Verfahren**")
    eps = st.selectbox("Spielraum der Balance ε (%)", list(C.EPS_CHOICES), key="eps_select", help="Wie weit eine Hälfte vom halben Gesamtgewicht abweichen darf. Ohne Spielraum wäre bei gleich schweren Standorten jeder Zug blockiert.")
    coarse = st.slider("Größe des gröbsten Netzes", *bounds("coarse_slider"), key="coarse_slider", help="Das Vergröbern hört auf, wenn höchstens so viele Standorte übrig sind (oder ein Schritt weniger als 5 % spart).")
    refine = st.selectbox("Verfeinern", list(C.REFINE_MODES), format_func=C.REFINE_NAMES.get, key="refine_select", help="Auf jeder Stufe oder nur auf der feinsten (dazwischen wird nur übertragen).")

sync_query_params({"kind_select": kind, "n_slider": int(n), "deg_slider": float(deg), "links_slider": int(links), "seed_input": int(seed), "eps_select": int(eps), "coarse_slider": int(coarse), "refine_select": refine})

settings = ev.Settings(kind, int(n), float(deg), int(links), int(seed), int(eps), int(coarse), refine)
with st.spinner("Rechne die Läufe beider Verfahren..."):
    a = _analysis(settings)
problems = ev.validate(a)
for p in problems:
    st.error(f"Prüfung fehlgeschlagen: {p}")
signature = f"{settings.kind}_{settings.n}_{settings.deg}_{settings.links}_{settings.seed}_{settings.eps}_{settings.coarse}_{settings.refine}"
inst = a.inst
g = a.graph

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"{C.KIND_NAMES[kind]} mit {_i(a.n)} Standorten und {_i(a.m)} Leitungen; {_i(len(a.ml))} Läufe je Verfahren, jeder mit eigener Zufallsreihenfolge. Spielraum {_i(eps)} %.")
better, equal, worse = a.wins
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mehrstufig: Schnitt", _dec(a.ml_mean), delta=f"bester Lauf {_i(min(a.ml_cuts))}", delta_color="off")
m2.metric("Einstufig: Schnitt", _dec(a.sl_mean), delta=f"bester Lauf {_i(min(a.sl_cuts))}", delta_color="off")
m3.metric("Unterschied", f"{ev._signed(1 - a.ml_mean / a.sl_mean, 1)} %" if a.sl_mean else "0 %", delta="weniger Leitungen mit Vergröbern" if a.ml_mean < a.sl_mean else "kein Vorteil", delta_color="off")
m4.metric("Läufe mit Vorteil", f"{better} von {len(a.ml)}", delta=f"gleich {equal}, schlechter {worse}", delta_color="off")
_show(ev.verdict_run(a))

tab_run, tab_cmp = st.tabs(["🔍 Ein Lauf", "⚖️ Mehrstufig gegen einstufig"])

# --- Tab 1: Ein Lauf ---------------------------------------------------------------------------------------------------------------------

with tab_run:
    run_no = st.slider("Lauf", 1, len(a.ml), 1, key=f"run_{signature}", help="Welcher der Läufe gezeigt wird (jeder hat eine eigene Reihenfolge beim Vergröbern).")
    r = a.ml[int(run_no) - 1]
    nl = len(r.levels)
    level_no = st.slider("Stufe (1 = feinste)", 1, nl, 1, key=f"level_{signature}_{int(run_no)}", help="Welche Stufe auf der Karte gezeigt wird; 1 ist das ursprüngliche Netz, die letzte das gröbste.") if nl > 1 else 1
    li = int(level_no) - 1
    side_levels = r.sides
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Stufe {_i(li + 1)} von {_i(nl)}: {_i(r.levels[li].graph.n)} Standorte**")
        st.plotly_chart(build_level_map(inst, r.levels, li, side_levels[li], height=440), width="stretch", key=f"level_map_{signature}_{int(run_no)}_{li}")
        st.caption("Knotengröße und Zahl im Knoten: sein Gewicht (Zahl der ursprünglichen Standorte); Leitungsbreite: ihr Gewicht. Gezeigt ist die Teilung der Stufe nach dem Verfeinern.")
    with c2:
        rows = []
        for lv, t in zip(r.levels, r.trace):
            rows.append({"Stufe": _i(t.index + 1), "Standorte": _i(t.n), "Leitungsgewicht": _i(t.edge_total), "Schnitt nach dem Übertragen": _i(t.cut_projected), "Schnitt nach FM": _i(t.cut_refined),
                         "Züge": _i(t.moves), "Durchläufe": _i(t.passes), "Verhältnis zur feineren": ("-" if t.index == 0 else "")})
        for k_, row in enumerate(rows):
            if k_ > 0:
                row["Verhältnis zur feineren"] = _dec(r.levels[k_].graph.n / r.levels[k_ - 1].graph.n, 2)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Der gröbste Schnitt ist der beste von {_i(len(r.trials))} Anfangsteilungen ({', '.join(_i(x) for x in r.trials)}). Aufwand dieses Laufs: {_i(r.moves)} Züge und {_i(r.visits)} besuchte Standorte beim Vergröbern.")
    st.markdown("**Schnitt auf jeder Stufe**")
    st.plotly_chart(build_trace(r), width="stretch", key=f"trace_{signature}_{int(run_no)}")
    st.caption("Grau: der Schnitt nach dem Übertragen auf die feinere Stufe (gleich dem Schnitt der gröberen Stufe nach dem Verfeinern); grün: nach dem Verfeinern.")

# --- Tab 2: Mehrstufig gegen einstufig ----------------------------------------------------------------------------------------------------

with tab_cmp:
    st.markdown("**Schnitt jedes Laufs**")
    st.plotly_chart(build_runs(a), width="stretch", key=f"runs_{signature}")
    st.caption("Jeder Lauf hat eine eigene Zufallsreihenfolge. Das einstufige Verfahren bekommt als Budget den Aufwand des mehrstufigen Laufs.")
    rows = [{"Lauf": _i(k_ + 1), "Mehrstufig": _i(m_.cut), "Aufwand": _i(m_.work), "Stufen": _i(len(m_.levels)), "Einstufig": _i(s_.cut), "Starts": _i(len(s_.starts)), "Züge": _i(s_.moves)} for k_, (m_, s_) in enumerate(zip(a.ml, a.sl))]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Aufwand = Züge von FM plus besuchte Standorte beim Vergröbern; „Züge“ beim einstufigen Verfahren zählen nur Züge (mindestens ein Start).")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Experimente über feste Netze")
st.caption(f"Je Netztyp {len(C.SWEEP_SEEDS)} feste Netze mit {C.EX_N} Standorten und je {C.EX_RUNS} Läufe, Spielraum {C.DEFAULT_EPS} %, gröbstes Netz höchstens {C.DEFAULT_COARSE} Standorte; nie ein einzelner Zufalls-Seed als Befund.")

st.subheader("🔬 Experiment 1: Stimmen Teilungen, Gewichte und Übertragen überein?")
rows_x = _exactness()
st.caption(f"{len(rows_x)} Netztypen mit je {len(C.SWEEP_SEEDS)} festen Netzen mit {C.EX_N} Standorten; geprüft werden Ausgewogenheit, Nachzählung des Schnitts, Gesamtgewicht, Größe der Stufen, Schnitt beim Übertragen und beim Verfeinern.")
st.plotly_chart(build_exact(rows_x), width="stretch", key="ex_exact")
_show(ev.verdict_exact(rows_x))

st.subheader("🔬 Experiment 2: Mehrstufig gegen einstufig bei gleichem Aufwand")
rows_c = _compare()
st.caption(f"Schnitt (Mittel über {len(C.SWEEP_SEEDS)} Netze und {C.EX_RUNS} Läufe) beider Verfahren; das einstufige bekommt als Budget den Aufwand des mehrstufigen Laufs.")
st.plotly_chart(build_compare(rows_c), width="stretch", key="ex_compare")
_show(ev.verdict_compare(rows_c))

st.subheader("🔬 Experiment 3: Wie schnell schrumpft das Netz?")
rows_s = _shrink()
st.caption("Verhältnis der Knotenzahl einer Stufe zu der der feineren (Heavy-Edge-Matching mit zufälliger Reihenfolge): 0,5 hieße, jeder Standort findet einen Partner.")
st.plotly_chart(build_shrink(rows_s), width="stretch", key="ex_shrink")
_show(ev.verdict_shrink(rows_s))

st.subheader("🔬 Experiment 4: Was bringt das Verfeinern?")
rows_r = _refine()
st.caption("Schnitt bei vier Läufen je Netz: nur auf die feinste Stufe übertragen, nur dort verfeinern oder auf jeder Stufe verfeinern.")
st.plotly_chart(build_refine(rows_r), width="stretch", key="ex_refine")
_show(ev.verdict_refine(rows_r))

st.subheader("🔬 Experiment 5: Wächst der Vorteil mit der Netzgröße?")
rows_n = _scale()
st.caption(f"Weniger Leitungen des mehrstufigen Verfahrens in Prozent gegenüber dem einstufigen bei gleichem Aufwand, für Kartennetz und Zufallsnetz bei {', '.join(str(x) for x in C.EX_SCALE_NS)} Standorten (je drei Netze).")
st.plotly_chart(build_scale(rows_n), width="stretch", key="ex_scale")
_show(ev.verdict_scale(rows_n))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Einfache Bausteine** | Zufälliges Heavy-Edge-Matching, FM mit Gewichten und eine Anfangsteilung aus Zufallsanordnungen: Systeme wie METIS, KaHIP oder KaMinPar sind deutlich ausgefeilter und schneller; die Demo erreicht sie nicht. | - |
| **Gleicher Aufwand heißt gleiche Züge** | Der Aufwand zählt Züge und besuchte Standorte; die echte Rechenzeit hängt von der Umsetzung ab (hier ein Python-FM mit Suche des besten Zuges in jedem Zug). | bisektion-demo |
| **Zwei Hälften** | Mehr als zwei Teile brauchen rekursives Teilen oder direktes k-Teilen; Hypergraphen (Hyperkanten statt Leitungen) sind nicht gebaut. | multiway-cut-demo |
| **Gleiche Leitungen am Anfang** | Alle Leitungen und Standorte zählen gleich; echte Lasten und Kapazitäten machen die Balance zu einer Gewichtsbedingung (die Demo kennt sie von den gröberen Stufen). | - |
| **Synthetische Netze, bis 300 Standorte** | Reale Netze haben andere Strukturen; der Vorteil wächst mit der Größe (Experiment 5), über 300 Standorte wird die Python-Umsetzung langsam. | - |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben** ein ungerichteter Graph $G = (V, E)$ mit Knotengewichten $c(v) > 0$ und Kantengewichten $w(e) > 0$ (am Anfang alle 1), $W = \sum_v c(v)$ und ein Spielraum $\varepsilon \ge 0$. Gesucht ist eine Zweiteilung $V = S \cup \bar S$ mit

$$c(S),\; c(\bar S) \;\le\; \Bigl\lceil (1 + \varepsilon)\,\frac{W}{2} \Bigr\rceil \quad \text{und kleinstem Schnitt} \quad w(S, \bar S) = \sum_{\{u,v\} \in E,\; u \in S,\; v \in \bar S} w(\{u,v\}).$$

**Vergröbern.** Ein Matching $M$ (Kanten ohne gemeinsamen Endpunkt) liefert das gröbere Netz $G'$: Jedes Paar $\{u, v\} \in M$ und jeder nicht gepaarte Knoten wird ein Knoten $\pi(\cdot)$; es gilt

$$c'(x) = \sum_{v \in \pi^{-1}(x)} c(v), \qquad w'(\{x, y\}) = \sum_{u \in \pi^{-1}(x),\; v \in \pi^{-1}(y)} w(\{u, v\}) \quad (x \ne y).$$

Beim **Heavy-Edge-Matching** besucht man die Knoten in zufälliger Reihenfolge und paart einen freien Knoten $v$ mit einem freien Nachbarn $u$, für den $w(\{v, u\})$ am größten ist.

**Übertragen.** Ist $S'$ eine Teilung von $G'$, so ist $S = \pi^{-1}(S')$ eine Teilung von $G$ mit

$$c(S) = c'(S'), \qquad w(S, \bar S) = w'(S', \bar S'),$$

denn Kanten innerhalb eines Paares liegen auf einer Seite und fallen weg, alle anderen Kanten werden zu ihrer Summe. Das Übertragen ändert also weder die Gewichte der Hälften noch den Schnitt.

**Fiduccia–Mattheyses.** Der Gewinn von $v$ ist $g(v) = \sum_{u \text{ jenseits}} w(\{v, u\}) - \sum_{u \text{ diesseits}} w(\{v, u\})$; der Schnitt sinkt um $g(v)$, wenn $v$ allein die Hälfte wechselt. Ein Durchlauf verschiebt immer den freien Knoten mit dem größten Gewinn, sofern die Zielhälfte danach höchstens das erlaubte Gewicht hat, sperrt ihn und führt den Anfang der Zugfolge mit dem kleinsten Schnitt aus.

**Aufwand.** Die Demo zählt die Züge von FM und die besuchten Knoten beim Vergröbern; das einstufige Verfahren darf Starts wiederholen, bis seine Züge den Aufwand des mehrstufigen Laufs erreichen.

**Literatur.** G. Karypis, V. Kumar: *A fast and high quality multilevel scheme for partitioning irregular graphs*, SIAM Journal on Scientific Computing 20 (1), 1998, S. 359–392. B. Hendrickson, R. Leland: *A multilevel algorithm for partitioning graphs*, Proceedings of the 1995 ACM/IEEE Supercomputing Conference. C. M. Fiduccia, R. M. Mattheyses: *A linear-time heuristic for improving network partitions*, 19th Design Automation Conference, 1982, S. 175–181.
Nachfolgende Systeme (METIS, KaHIP, KaMinPar, KaHyPar für Hypergraphen) bauen dieses Gerüst mit besseren Zuordnungen und Verfeinerungen aus; sie sind nicht gebaut, und die Demo erhebt keinen Anspruch, sie zu erreichen. Die Demo rechnet FM in der einfachsten Form (ohne Gewinn-Fächer).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
