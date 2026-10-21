"""Konstanten: Regler-Grenzen, feste Annahmen, Experiment-Raster, Farben und Presets."""

KINDS = ("geometric", "grid", "ring", "scalefree", "random", "verbund")
KIND_NAMES = {"geometric": "Kartennetz", "grid": "Gitter", "ring": "Ring mit Abkürzungen", "scalefree": "Skalenfreies Netz", "random": "Zufallsnetz", "verbund": "Verbundnetz"}
KIND_HELP = {
    "geometric": "Zufällige Punkte in einem Quadrat, jeder mit seinen drei nächsten Nachbarn verbunden: ein Straßennetz im Kleinen.",
    "grid": "Standorte auf einem Raster, Verbindungen zu den vier Nachbarn.",
    "ring": "Vier nächste Nachbarn auf einem Ring, ein Zehntel der Leitungen führt zu einem zufälligen Standort.",
    "scalefree": "Neue Standorte hängen sich bevorzugt an gut vernetzte an (je zwei Leitungen).",
    "random": "Zufallsbaum plus zufällige Leitungen; der mittlere Grad ist einstellbar.",
    "verbund": "Drei Regionen mit dichten inneren Verbindungen und wenigen Leitungen dazwischen (einstellbar).",
}
KIND_REGLER = {
    "geometric": "Wirksam: Standorte, Zufalls-Seed.",
    "grid": "Wirksam: Standorte (gerundet auf ein Rechteck). Zufalls-Seed, Grad und Leitungen wirken nicht auf das Netz (der Seed wirkt auf die Reihenfolge beim Vergröbern).",
    "ring": "Wirksam: Standorte, Zufalls-Seed.",
    "scalefree": "Wirksam: Standorte, Zufalls-Seed.",
    "random": "Wirksam: Standorte, mittlerer Grad, Zufalls-Seed.",
    "verbund": "Wirksam: Standorte, mittlerer Grad (in den Regionen), Leitungen zwischen den Regionen, Zufalls-Seed.",
}

N_MIN, N_MAX, DEFAULT_N = 24, 300, 300
DEG_MIN, DEG_MAX, DEFAULT_DEG = 2.0, 10.0, 4.0
LINKS_MIN, LINKS_MAX, DEFAULT_LINKS = 1, 8, 2
SEED_MAX, DEFAULT_SEED = 999999, 7
DEFAULT_KIND = "geometric"
REGIONS = 3
RING_NEIGHBOURS, RING_REWIRE = 4, 0.1
SCALEFREE_M = 2

EPS_CHOICES = (1, 3, 10)                 # erlaubte Abweichung der Hälftengröße vom Ideal in Prozent
DEFAULT_EPS = 3
COARSE_MIN, COARSE_MAX, DEFAULT_COARSE = 8, 60, 24      # Zielgröße des gröbsten Netzes
REFINE_MODES = ("each", "finest")
REFINE_NAMES = {"each": "auf jeder Stufe", "finest": "nur auf der feinsten Stufe"}
DEFAULT_REFINE = "each"
INITIAL_TRIALS = 5                       # zufällige Anfangsteilungen auf dem gröbsten Netz (die beste wird genommen)
MAX_PASSES = 8                           # Durchläufe des Verfeinerns je Stufe (höchstens)
LEVEL_PASSES = 4                         # Durchläufe auf den Zwischenstufen
STALL = 0.95                             # Vergröbern bricht ab, wenn ein Schritt weniger als 5 % Standorte spart
RUNS = 8                                 # Läufe im Vergleich (jeder mit eigener Zufallsreihenfolge)
RUN_SALT = 4242
LEVEL_SALT = 17

SWEEP_SEEDS = tuple(range(100000, 100006))   # feste Netze für alle Experimente: nie ein einzelner Zufalls-Seed als Befund
EX_N = 256                               # Quadratzahl: das Gitter hat dieselbe Größe wie die anderen Netztypen
EX_KINDS = (("geometric", 0.0), ("grid", 0.0), ("ring", 0.0), ("scalefree", 0.0), ("random", 4.0), ("verbund", 5.0))
EX_RUNS = 4
EX_SCALE_NS = (64, 144, 256)             # Größen für den Verlauf des Vorteils
EX_SCALE_KINDS = (("geometric", 0.0), ("random", 4.0))

COLOR_SIDE_A = "#2F6B65"
COLOR_SIDE_B = "#4c78a8"
COLOR_CUT = "#d1495b"
COLOR_NODE = "#c9d1d3"
COLOR_EDGE = "rgba(150,150,150,0.45)"
COLOR_BAR = "#4c78a8"
COLOR_BAR2 = "#e8a33d"
COLOR_BAR3 = "#2F6B65"
COLOR_LOCK = "#8d6cab"

PRESETS = {
    "Kartennetz: mehrstufig gewinnt": {"kind": "geometric", "n": 300, "deg": DEFAULT_DEG, "links": DEFAULT_LINKS, "seed": 7, "eps": 3, "coarse": 24, "refine": "each"},
    "Zufallsnetz: ein kleiner Vorteil": {"kind": "random", "n": 200, "deg": 4.0, "links": DEFAULT_LINKS, "seed": 7, "eps": 3, "coarse": 24, "refine": "each"},
    "Gitter: kein Vorteil": {"kind": "grid", "n": 256, "deg": DEFAULT_DEG, "links": DEFAULT_LINKS, "seed": 7, "eps": 3, "coarse": 24, "refine": "each"},
    "Skalenfreies Netz: kein Vorteil": {"kind": "scalefree", "n": 200, "deg": DEFAULT_DEG, "links": DEFAULT_LINKS, "seed": 7, "eps": 3, "coarse": 24, "refine": "each"},
    "Verbundnetz: einstufig reicht": {"kind": "verbund", "n": 200, "deg": 5.0, "links": 2, "seed": 7, "eps": 3, "coarse": 24, "refine": "each"},
    "Kleines Kartennetz: noch kein Vorteil": {"kind": "geometric", "n": 64, "deg": DEFAULT_DEG, "links": DEFAULT_LINKS, "seed": 7, "eps": 3, "coarse": 24, "refine": "each"},
    "Verfeinern nur auf der feinsten Stufe": {"kind": "geometric", "n": 200, "deg": DEFAULT_DEG, "links": DEFAULT_LINKS, "seed": 7, "eps": 3, "coarse": 24, "refine": "finest"},
}
PRESET_HELP = {
    "Kartennetz: mehrstufig gewinnt": "Ein Straßennetz im Kleinen mit 300 Standorten: Bei gleichem Aufwand schneidet das mehrstufige Verfahren im Mittel 8,9 Leitungen, das einstufige 12,4.",
    "Zufallsnetz: ein kleiner Vorteil": "Zufallsnetz mit 200 Standorten: mehrstufig im Mittel 66,9, einstufig 69,8 Leitungen (der Vorteil liegt bei wenigen Prozent).",
    "Gitter: kein Vorteil": "Ein Gitter mit 256 Standorten: Das Optimum ist 16, das einstufige Verfahren trifft es in allen acht Läufen, das mehrstufige in sechs von acht.",
    "Skalenfreies Netz: kein Vorteil": "Das Netz schrumpft beim Vergröbern langsam (neun Stufen für 200 Standorte, vermutlich wegen der Hubs). In diesem einen Netz ist das einstufige Verfahren bei gleichem Aufwand besser (2 zu 6 Läufe); im Mittel über die gemessenen Netze ist der Unterschied klein, und einzelne Läufe wechseln.",
    "Verbundnetz: einstufig reicht": "Drei Regionen: Auch das einstufige Verfahren findet die Gemeinschaften, ein Vorteil des Vergröberns bleibt aus.",
    "Kleines Kartennetz: noch kein Vorteil": "Bei 64 Standorten genügen wenige Starts des einstufigen Verfahrens: Der Vorteil des Vergröberns zeigt sich erst bei größeren Netzen.",
    "Verfeinern nur auf der feinsten Stufe": "Wird auf den Zwischenstufen nicht verfeinert, verliert das mehrstufige Verfahren seinen Vorteil: Der Schnitt steigt von 5,4 auf 6,9 (Mittel über acht Läufe).",
}
