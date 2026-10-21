# Mehrstufig partitionieren: Vergröbern, Teilen, Verfeinern – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-multilevel-partitionierung-demo.streamlit.app/)**

Sechstes Stück der Reihe **Schnitte und Zusammenhang** (Arbeitstitel der Linie `schnitte-zusammenhang`) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Bereich Graphentheorie. Vehikel: ein **Verbindungsnetz zwischen Standorten** (Depots, Umschlagpunkte, Werke), synthetisch und mit festem Zufalls-Seed, bis 300 Standorte. Die Demo fragt: **Warum vergröbert man ein Netz erst, teilt das kleine und verfeinert dann, und wann lohnt das?** Aufgabe: das Netz in **zwei fast gleich schwere Hälften** teilen (Abweichung einstellbar: 1, 3 oder 10 %) und dabei möglichst wenige Leitungen zwischen den Hälften lassen. Das **mehrstufige Verfahren** paart Standorte nach dem schwersten Leitungsgewicht (Heavy-Edge-Matching), fasst jedes Paar zu einem Standort zusammen, wiederholt das bis zu einem kleinen Netz, teilt dieses (beste von fünf zufälligen Anfangsteilungen, dann verbessert) und überträgt die Teilung stufenweise zurück, nach jeder Stufe mit **Fiduccia–Mattheyses (FM)** verbessert. Es wird bei **gleichem Aufwand** (Züge und besuchte Standorte; das einstufige Verfahren darf seinen letzten Start zu Ende führen) gegen das **einstufige Verfahren** (FM mit zufälligen Neustarts auf dem feinsten Netz) gestellt. Es ist **kein METIS**: die Bausteine sind einfach (siehe Grenzen).

Nachbarn: das ausgewogene Teilen mit Kernighan–Lin und FM im Kleinen zeigt `bisektion-demo` (Stück 5), die spektrale Alternative `spektrale-schnitte-demo`, mehr als zwei Teile `multiway-cut-demo`, die Paarung selbst (Matching) `hopcroft-karp-demo` und `blossom-demo`.

## Warum dieses Problem

Eine lokale Verbesserung (FM) sieht nur die Umgebung eines Standorts: Ein Zug, der den Schnitt erst größer macht, damit er später kleiner wird, gelingt nur in engen Grenzen. Auf einem **vergröberten** Netz ist ein Standort ein ganzer Verbund, und ein Zug dort verschiebt viele Standorte auf einmal; das gibt dem Verfahren einen weiten Blick, den die Verfeinerung danach zurechtrückt. Die Demo misst, **wann das gewinnt**: deutlich bei **großen Kartennetzen** (256 Standorte: 31 % weniger Leitungen im Schnitt über sechs Netze, von Netz zu Netz stark streuend), **kaum** bei Zufallsnetzen, **nicht** bei Gittern, Ringen und Verbundnetzen, **kein Vorteil** bei skalenfreien Netzen (+1 %, im Rahmen der Streuung; dort schrumpft das Netz beim Vergröbern langsamer, vermutlich wegen der Hubs); und dass das Verfeinern der Kern ist, nicht das Vergröbern allein: Auf jeder Stufe zu verfeinern bringt im Mittel wenig (5 %), beim Kartennetz aber den ganzen Vorsprung.

## Befunde und Korrekturen gegenüber dem Plan

- **Die Zahlen der Vormessung gelten nicht:** Sie stammen aus Wegwerfskripten mit anderen Netzen (um 1.000 Standorte), anderem Zeitbudget und einer anderen Zählung des Aufwands (Plan: „kein klarer Sieger“, mehrstufig besser auf Ring und skalenfrei). Die Demo misst neu (Tabelle unten): **Ring** und **skalenfrei** sind bei gleichem Aufwand **nicht** besser mit Vergröbern (−1 und +1 %, bei 256 Standorten); der klare Befund ist das **Kartennetz**.
- **Der Vorteil wächst mit der Größe:** Auf Kartennetzen mit 64 Standorten ist das einstufige Verfahren besser (−9 %), mit 144 gleichauf (−3 %), mit 256 liegt mehrstufig vorn (21 % weniger Leitungen; Summe über drei Netze). Bei Zufallsnetzen bleibt es bei −1, −3 und 0 %.
- **Aufwand:** Der Aufwand eines mehrstufigen Laufs sind die Züge des Verfeinerns plus die besuchten Standorte beim Vergröbern; das einstufige Verfahren bekommt diese Zahl als Budget (Neustarts, solange Züge übrig sind) und darf den letzten Start zu Ende führen; es verbraucht dadurch je nach Netztyp das 1,2- bis 1,5-Fache des Budgets, was das mehrstufige Verfahren eher benachteiligt. Ein mehrstufiger Lauf kostet bei 256 Standorten im Mittel 1.440 (Kartennetz) bis 2.400 Schritte (skalenfrei), das einstufige Verfahren schafft dafür 1,5 bis 2,8 Starts.
- **Kein Hypergraph:** Der Plan sah eine Messzahl zu Hyperkanten vor; sie ist nicht gebaut (siehe unten).
- **Keine Kopplung an eine Rechenzeit:** Verglichen wird in Zügen, nicht in Sekunden.

## Modell

Ein einfaches, **zusammenhängendes** Netz mit n Standorten (Zufallsnetz: Zufallsbaum plus Leitungen bis zum mittleren Grad; Gitter als Rechteck; Ring mit Abkürzungen; skalenfreies Netz nach bevorzugter Anbindung; Verbundnetz aus drei Regionen mit einstellbar vielen Leitungen dazwischen; Kartennetz aus zufälligen Punkten, jeder mit seinen drei nächsten Nachbarn verbunden). Jeder Standort hat Gewicht 1, jede Leitung Gewicht 1; beim Vergröbern addieren sich Standortgewichte, und parallele Leitungen werden zu einer mit der Summe der Gewichte. Eine Teilung in zwei Hälften ist **zulässig**, wenn keine Hälfte schwerer ist als ⌈(1 + ε/100) · W / 2⌉ (W: Gesamtgewicht; höchstens W − 1, damit beide Hälften nichtleer bleiben). Der **Schnitt** ist die Summe der Gewichte der Leitungen zwischen den Hälften. Weil das Übertragen auf die feinere Stufe den Schnitt und die Hälftengewichte unverändert lässt, kann man auf jeder Stufe mit dem Wert der gröberen weiterrechnen. Der Zufall kommt nur aus `random.Random` (plattformstabil). Formeln: Expander „📐 Mathematische Formulierung“ in der App.

## Methodik

- **Heavy-Edge-Matching:** Standorte in zufälliger Reihenfolge; jeder noch freie Standort wird mit dem freien Nachbarn der schwersten Leitung gepaart (bei Gleichstand der kleineren Nummer), wenn das Gesamtgewicht des Paars höchstens `max(2, ⌊Grenze/3⌋)` bleibt. Das **Vergröbern** bricht ab, wenn das Netz höchstens die gewählte Zielgröße (8 bis 60) hat oder ein Schritt weniger als 5 % der Standorte spart.
- **Anfangsteilung:** fünf zufällige ausgewogene Teilungen des gröbsten Netzes, jede mit FM verbessert; genommen wird die beste (bevorzugt ausgewogen, dann kleinster Schnitt).
- **FM mit Gewichten:** Gewinn eines Standorts = Leitungsgewicht nach außen minus nach innen; in jedem Durchlauf wird jeder Standort höchstens einmal bewegt, immer der Standort mit dem größten Gewinn, dessen Zug die Balance einhält (bei Gleichstand der kleinere), auch wenn der Gewinn negativ ist; zurückgesetzt wird auf den besten Stand des Durchlaufs. Durchläufe, bis einer nichts mehr gewinnt: höchstens 8 auf der gröbsten und der feinsten Stufe, 4 auf Zwischenstufen.
- **Verfeinern:** „auf jeder Stufe“ oder „nur auf der feinsten Stufe“ (Ablation).
- **Einstufig bei gleichem Aufwand:** zufällige ausgewogene Teilung auf dem feinsten Netz, FM, und so fort mit neuem Start, solange das Zugbudget des mehrstufigen Laufs reicht; der letzte Start wird zu Ende geführt, die beste Teilung gezählt.
- **Läufe:** acht Läufe je Einstellung, jeder mit eigener Zufallsreihenfolge.
- **Unabhängige Prüfer:** Die App prüft jedes Ergebnis vor der Anzeige (Teilung ausgewogen und in zwei nichtleere Hälften; Schnitt gleich der Nachzählung; Gesamtgewicht auf jeder Stufe erhalten; jede Stufe kleiner als die feinere; Schnitt beim Übertragen unverändert; nie größer nach dem Verfeinern; einstufiges Ergebnis ausgewogen und gleich dem besten Start).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`; Zähler über feste Netze (je Netztyp sechs Netze mit 256 Standorten und je vier Läufe, beim Gitter sechsmal dasselbe Netz, weil der Zufalls-Seed dort nur die Reihenfolge ändert; im Verlauf nach Größe drei Netze), nie ein einzelner Zufalls-Seed. Netztypen: Kartennetz, Gitter, Ring mit Abkürzungen, skalenfreies Netz, Zufallsnetz (Grad 4), Verbundnetz (Grad 5, eine Leitung zwischen je zwei Regionen).

| Frage | Befund |
|---|---|
| Stimmt die Rechnung auf allen Netzen? | Auf **36** Netzen (das Gitter sechsmal dasselbe) bestehen **1.296** Prüfungen (je Netz vier Läufe mit je neun Prüfungen) ohne Abweichung. |
| Gewinnt mehrstufig bei gleichem Aufwand? | **Über alle Netztypen kaum:** in der Summe 2 % weniger Leitungen. Je Netztyp (positiv: mehrstufig besser): Kartennetz **31 %** (von Netz zu Netz stark streuend), skalenfreies Netz 1 %, Zufallsnetz 3 %, Ring −1 %, Gitter −2 %, Verbundnetz −2 %. Kartennetz im Mittel **5,8** Leitungen gegen **8,4**. |
| Wie oft gewinnt es einzelne Läufe? | In **70 von 144** Läufen ist mehrstufig besser, in **29** gleich, in 45 schlechter. Kartennetz 18 von 24, Gitter 1 (19 gleich), Ring 7, skalenfrei 17, Zufallsnetz 17, Verbundnetz 10. |
| Wie schnell schrumpft das Netz? | Mittleres Verhältnis der Knotenzahlen aufeinanderfolgender Stufen: Gitter und Ring **0,55**, Kartennetz und Verbundnetz 0,58, Zufallsnetz 0,60, **skalenfrei 0,73** (8,8 Stufen gegen 5,1 bis 6,0). Das Verhältnis liegt bei jedem Netztyp über 0,5 (eine Paarung kann höchstens halbieren). |
| Wie wichtig ist das Verfeinern? | Nur auf die feinste Stufe zu übertragen (ohne Verfeinern) ergibt im Mittel **80,2** Leitungen, Verfeinern nur auf der feinsten Stufe **47,8**, auf jeder Stufe **45,4** (36 Netze, sechs Netztypen). |
| Wächst der Vorteil mit der Größe? | Kartennetz mit 64 / 144 / 256 Standorten: **−9 / −3 / 21 %**; Zufallsnetz **−1 / −3 / 0 %** (Summe über drei Netze, vier Läufe). Bei Kartennetzen kippt das Bild mit der Größe; die Einzelzahlen streuen von Netz zu Netz stark (bei 256 Standorten 21 % in den ersten drei Netzen, 31 % in sechs), und „wächst mit der Größe“ beruht auf nur drei Größen. |
| Das Kartennetz-Preset? | Ein einzelnes Netz (kein Typenmittel), 300 Standorte: mehrstufig im Mittel **8,9** Leitungen, einstufig **12,4**; in 6 von 8 Läufen besser, in 1 gleich, in 1 schlechter; sechs Stufen (300 → 169 → 98 → 55 → 30 → 18), 1.404 Schritte, im Mittel 1,5 Starts des einstufigen Verfahrens. |
| Die anderen Presets? | Je ein Netz mit acht Läufen. Zufallsnetz (200): 66,9 gegen 69,8, Läufe 5 / 1 / 2. Gitter (256): Optimum **16**, das einstufige Verfahren trifft es in allen acht Läufen, das mehrstufige in sechs. Skalenfrei (200): neun Stufen, in diesem Netz einstufig besser (2 / 0 / 6), im Mittel über die sechs Netze der Tabelle bei 256 Standorten +1 %. Verbundnetz (200): in diesem Netz einstufig besser (2 / 3 / 3). Kleines Kartennetz (64): 2 / 5 / 1. Verfeinern nur auf der feinsten Stufe (200): **6,9** gegen **5,4** bei Verfeinern auf jeder Stufe. |
| Ist die Rechnung korrekt? | Schnitt gleich `networkx.cut_size`, gröbere Netze gleich `networkx.quotient_graph` der Mitgliedergruppen des feinsten Netzes, Paarung gültig, symmetrisch, maximal und gleich einer Bezugsrechnung mit sortierten Nachbarn, Übertragen erhält Schnitt und Gewichte, kein Verfahren unter dem Optimum (Aufzählung aller erlaubten Teilungen bei höchstens 16 Standorten) und jedes Ergebnis ausgewogen, FM-Gewinn gleich der Differenz nachgezählter Schnitte (`tests/test_oracle_mehrstufig.py`). |

## Ehrliche Grenzen

- **Kein METIS.** Zufälliges Heavy-Edge-Matching, einfaches FM und eine Anfangsteilung aus Zufallsanordnungen; METIS, KaHIP und KaMinPar sind ausgefeilter und viel schneller. Ob ein starker Partitionierer auf den Netzen anders aussähe, ist nicht gemessen.
- **Der Befund hängt am Netztyp und an der Größe.** Klar vorn ist mehrstufig in den gemessenen Netzen nur auf größeren Kartennetzen; bei den übrigen Typen ist der Unterschied klein oder negativ. Größere Netze anderer Typen und größere Kartennetze sind nicht systematisch gemessen. Die Vormessung des Plans (um 1.000 Standorte) ist ausdrücklich nicht übertragen.
- **Gleicher Aufwand heißt gleiche Züge.** Der Aufwand zählt Züge und besuchte Standorte (das einstufige Verfahren führt seinen letzten Start zu Ende und bekommt dadurch das 1,2- bis 1,5-Fache); Sekunden hängen von der Umsetzung ab (hier ein Python-FM mit Suche des besten Zuges in jedem Zug).
- **Zwei Hälften, einfache Leitungen.** Mehr als zwei Teile brauchen rekursives Teilen oder direktes k-Teilen; Hypergraphen sind nicht gebaut.
- **Synthetische Netze bis 300 Standorte.** Reale Netze haben andere Strukturen; über 300 Standorte wird die Python-Umsetzung langsam.
- **Wenige Läufe.** Acht Läufe je Einstellung, in den Experimenten vier Läufe auf sechs Netzen; die Anzahl der Gewinnläufe ist ein Zähler, kein Test.

## Tests

Das Verzeichnis `tests/` enthält zurzeit 114 Tests (Laufzeit etwa eine Minute auf dem Entwicklungsrechner):

- **Kern** (`test_algorithm.py`): jede Einheit an von Hand gerechneten Mini-Netzen: Schnitt, Grenze, Gewinne, ein FM-Durchlauf, Paarung, Zusammenfassen, Übertragen, Mitgliedergruppen, mehrstufiger Lauf auf einem Pfad mit 24 Standorten.
- **Szenarien** (`test_scenario.py`): einfach, zusammenhängend, deterministisch für jeden Netztyp; Größen.
- **Orakel** (`test_oracle_mehrstufig.py`): networkx (Schnitt, Quotientennetz), Aufzählung aller erlaubten Teilungen, Bezugsrechnung der Paarung.
- **Randfälle** (`test_edge_cases.py`): zwei und drei Standorte, Netz ohne Leitungen, isolierte Standorte, schwere Standorte, Gleichstände, Gewichtsgrenze, jede Art von Prüffehler an verfälschten Analysen, Grenzen der Regler.
- **Zweige, Verdikte, Befunde** (`test_branches.py`, `test_verdicts.py`, `test_claims.py`): jedes Preset prüft sauber und jede Abbildung rendert (auch jede Stufe); jedes Verdikt an handgesetzten Zeilen; jede Zahl dieses README.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), Layout (Schnellstart oben, aktives Preset hervorgehoben), Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: ein Lauf Stufe für Stufe, Vergleich mehrstufig gegen einstufig, fünf Experimente |
| `mlp_algorithm.py` | Kern: gewichtetes Netz, Schnitt, FM, Heavy-Edge-Matching, Zusammenfassen, mehrstufiger Lauf, einstufiges Verfahren |
| `mlp_constants.py` | Regler-Grenzen, Netztypen, Raster der Experimente, Farben, Presets |
| `mlp_evaluation.py` | Analyse, Prüfung vor der Anzeige, Verdikte |
| `mlp_experiments.py` | Experimente über feste Netze |
| `mlp_layout.py` | Fester Kopf (Stylesheet) |
| `mlp_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `mlp_scenario.py` | Netze: Zufallsnetz, Gitter, Ring, skalenfreies Netz, Verbundnetz, Kartennetz |
| `mlp_visualization.py` | Plotly-Abbildungen (Karte je Stufe, Schnitt je Stufe, Läufe, Experimente) |
| `requirements-dev.txt` | Laufzeit plus pytest und networkx (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Mini-Netze, Orakel, Randfälle, Zweige, Verdikte, Befunde, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Hypergraph-Partitionierung** (Hyperkanten statt Leitungen, KaHyPar): ein eigenes Gerüst; hier nur als Einordnung in der Formulierung.
- **Mehr als zwei Teile, Flussverfeinerung, gelernte Partitionierer:** nicht gebaut.
- **Echte Netze oder Dateiimport:** die Netze sind synthetisch.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, pandas und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
