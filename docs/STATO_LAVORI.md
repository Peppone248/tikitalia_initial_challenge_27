# Stato dei lavori — MTAH27 Phase 1

Cosa c'è, cosa manca e cosa blocca cosa. Si aggiorna a ogni passo che cambia lo stato di una
riga, con una voce nel registro in fondo.

**Legenda:** ✅ fatto · 🟡 parziale / da rivedere · ⬜ da fare · ⏸ rimandato (v1 o se avanza tempo)

**Scadenza:** 17 novembre 2026. Consegna: CSV `<team>.csv` + repo pubblico + documento tecnico
`<team>.pdf` (SUBMISSION.md).

**Obiettivo immediato:** un primo CSV valutabile (v0), con le sole regole D già pronte + una regola
H/A + il motore v0. Vedi §8.

Riferimenti: [REGOLE_PSEUDO_LABEL.md](REGOLE_PSEUDO_LABEL.md) (schede delle regole) ·
[NOMENCLATURA.md](NOMENCLATURA.md) (termini; motore al §6) ·
[IMPROVEMENTS_DA_DISCUTERE.md](IMPROVEMENTS_DA_DISCUTERE.md) (proposte A1–B3).

---

## 1. Dati e preprocessing — `preprocessing.py`, `geometry.py`

| Pezzo | Stato | Owner | Note |
|---|---|---|---|
| Parsing ATD → array in metri, `Match` | ✅ | Beppe | cache in `outputs/cache/atd_arrays.npz` |
| Smart Tagging caricato in `Match` | ✅ | Beppe | intervalli in frame, `team` 0/1/None |
| Pulizia salti della palla (`V_JUMP`), smoothing, velocità | ✅ | Beppe | |
| `ball_still_for` | ✅ | Beppe | |
| Geometria del campo (`Pitch`) | ✅ | Beppe | `geometry.yaml` |
| Unità della velocità ATD (`s`: km/h o m/s?) | ⬜ | Gio | finché non è chiarito, velocità solo dalle posizioni |
| Slot vuoti / duplicati / scambi di slot | 🟡 | Gio | `scripts/check_atd.py`; da riportare nel preprocessing |
| Join ATD – Smart Tagging normalizzato (Team H/A/N/A, fasi a coppie) | ⬜ | Francesco | serve solo alle regole S-*; **non** blocca la v0 |

## 2. Feature per frame — `features.py` (non esiste)

| Pezzo | Stato | Owner | Note |
|---|---|---|---|
| Giocatore più vicino per squadra: `nearest[team]` (dist, pos, vel) | ⬜ | | **blocca la v0** |
| `nearest_team` | ⬜ | | **blocca la v0** |
| `n_within(team, r)` | ⬜ | | |
| Centroide, spread, giocatori in metà/area avversaria | ⬜ | | per P-SHAPE, R-SET |
| `collective_slowdown` (velocità mediana vs riferimento mobile) | ⬜ | | per D-WHISTLE, D-GAP |

## 3. Eventi — `events.py`

| Pezzo | Stato | Owner | Note |
|---|---|---|---|
| Calcio d'inizio dallo schieramento (`kickoffs`) | ✅ | Beppe | 5 accettati, tutti con un tag ST; recupero con `rescued_by_st` |
| Direzione d'attacco per tempo (`attack_directions`) | ✅ | Beppe | coerente con il metodo del portiere |
| `control_episode` | ⬜ | | **blocca la v0** (P-DRIB, P-FREE, conferma dei TURNOVER) |
| `pass_event` | ⬜ | | per P-PASS, T-INTERCEPT |
| `restart_kick` (palla ferma → calcio) | ⬜ | | **blocca la v0** (chiusura dei D) |

## 4. Regole — `rules/`

### 4.1 Infrastruttura

| Pezzo | Stato | Note |
|---|---|---|
| `Rule`, `RuleOutput`, registro da `rules.yaml` | ✅ | parametri dichiarati per regola, letture non dichiarate = errore |
| `params.yaml` validato (valore, unità, origine, motivazione) | ✅ | |
| Forza: z-score, saturazione, fattore di durata (`strength.py`) | ✅ | |
| `emit()`: sovrapposizioni con etichette diverse | ⬜ | oggi la label si sovrascrive e la forza prende il max → `assert` prima delle regole H/A |
| Scala dello z per condizione (σ = max(sensore, tolleranza semantica)) | ⬜ | REGOLE §13: oggi lo z satura sempre |

### 4.2 Regole D (palla morta) — `rules/dead.py`

| Regola | Stato | Note |
|---|---|---|
| D-OUT | 🟡 | implementata; manca la plausibilità delle uscite brevissime (0,04–0,08 s); il D finisce al rientro della palla, non alla ripresa |
| D-GOAL (dal calcio d'inizio all'indietro) | 🟡 | 3 goal nel 1° tempo (1 senza candidato), 2 `void` nel 2°, 2 tag GOALS ST senza goal |
| D-KO come regola che emette D | ⬜ | il rilevamento c'è (`events.py`); manca il D dall'inizio del blocco valutato al calcio — **v0** |
| D-HOLD (D fino alla ripresa) | ⬜ | **v0**, dipende da `restart_kick` |
| D-SHOT | ⬜ | |
| D-WHISTLE (varianti A e B) | ⬜ | importante per i D brevi (IMPROVEMENTS A4) |
| D-GAP (solo con rallentamento) | ⬜ | |
| D-SP | ⬜ | dipende dal join ST |
| D-EDGE | ⬜ | |

### 4.3 Regole H/A (possesso) — `rules/possession.py` (non esiste)

| Regola | Stato | Note |
|---|---|---|
| P-DRIB | ⬜ | **v0**, dipende da `control_episode` |
| P-FREE | ⬜ | **v0** (in alternativa a P-DRIB, o insieme) |
| P-PASS | ⬜ | dipende da `pass_event` |
| P-PROG | ⬜ | |
| P-GOAL | ⬜ | dipende da D-GOAL |
| P-SHAPE-OPEN | ⬜ | SOFT, utile per lo stato iniziale |

### 4.4 TURNOVER — `rules/turnover.py` (non esiste)

| Regola | Stato | Note |
|---|---|---|
| T-REV, T-INTERCEPT, T-TACKLE, T-LOOSE | ⬜ | dopo la v0: nella v0 il cambio lo decidono le ancore P-* |
| T-TEAMFLIP | ⬜ | per i cambi durante i buchi della palla |

### 4.5 RESTART — `rules/restart.py` (non esiste)

| Regola | Stato | Note |
|---|---|---|
| R-TOUCH | ⬜ | **v0**: la più generale, basta da sola per chiudere i D |
| R-KO | ⬜ | |
| R-THROW, R-LAST, R-CORNER, R-GOALKICK, R-WALL, R-SET | ⬜ | |
| Consenso sulle riprese | ⬜ | |

### 4.6 Smart Tagging — `rules/smart_tagging.py` (non esiste)

| Regola | Stato | Note |
|---|---|---|
| S-PHASE, S-TRANS, S-FLIP, S-ZONE, S-SP, S-GOAL | ⬜ | dipendono dal join di Francesco |
| S-QUALITY (coerenza per code nel report) | ⬜ | |

## 5. Motore — `engine.py` (non esiste)

| Pezzo | Stato | Note |
|---|---|---|
| `collect_votes` (matrice H/A/D, `w = q · s`, maschera di scope) | ⬜ | **v0** |
| `find_anchors` (`CONF_ANCHOR`, conflitti nel log) | ⬜ | **v0** |
| `run_inertia` (macchina a stati per tempo, `INERTIA_MIN`, reset a ogni tempo) | ⬜ | **v0** |
| Retrodatazione dei cambi all'inizio dell'episodio (A2) | ⬜ | **v0** |
| Nessuna X sui frame valutati (A1) | ⬜ | **v0** |
| Parametri del motore in `params.yaml` (`CONF_ANCHOR`, `INERTIA_MIN`, `CONF_SOFT_START`) | ⬜ | **v0** |
| Test di inerzia su scenari sintetici | ⬜ | **v0**: es. NOMENCLATURA §6 |
| Label model (Dawid–Skene) al posto del voto pesato | ⏸ | v1 |
| HSMM / Viterbi al posto dell'isteresi | ⏸ | v1 |

## 6. Uscita e valutazione

| Pezzo | Stato | Note |
|---|---|---|
| Writer CSV `frame,label` (199.319 righe) — `submission.py` | ⬜ | **v0** |
| Comando unico dati → CSV (es. `scripts/make_submission.py`) | ⬜ | **v0**; serve anche per la riproducibilità |
| `validate_submission.py` sul file | ⬜ | **v0** |
| `run_evaluation.py` sui 5 minuti pubblici (solo controllo, non tuning) | ⬜ | **v0** |
| Baseline di riferimento | ✅ | `evaluation/scoreboard.csv`: migliore `baseline_smooth125`, spellF1 0,42, macroF1 0,58 |
| Report per regola (copertura, conflitti, accordo ST) | 🟡 | `report.py` copre D-OUT e D-GOAL |
| Set di validazione su altre partite (M1, IMPROVEMENTS B3) | ⬜ | unico modo onesto di confrontare versioni |

## 7. Repo, consegna, documentazione

| Pezzo | Stato | Owner | Note |
|---|---|---|---|
| Codice e docs di Beppe sul repo condiviso | ⬜ | Beppe | al 27/10 `master` remoto ha solo `src/tikitalia/__init__.py` |
| Dati fuori da git, `outputs/` in `.gitignore` | ✅ | | |
| Dipendenze bloccate + istruzioni di riproduzione | ⬜ | | requisito di consegna |
| Documento tecnico | ⬜ | | approccio, asset, dati etichettati, regole, limiti, riproduzione |
| Risposte Q1–Q8 dalla GT pubblica (solo significato) | ⬜ | Gio | REGOLE §11 |
| Decisioni A1–A5, B1–B3 | ⬜ | team | prossima call |

## 8. Strada verso il primo CSV (v0)

In ordine. Si spunta qui e si aggiorna la riga corrispondente sopra.

- [ ] `features.py`: `nearest[team]`, `nearest_team`
- [ ] `events.py`: `control_episode`, `restart_kick`
- [ ] `emit()`: niente sovrapposizioni con etichette diverse
- [ ] Regole: D-KO (emette D), D-HOLD, P-DRIB e/o P-FREE, R-TOUCH
- [ ] `engine.py`: voti → ancore → inerzia → retrodatazione → riempimento X
- [ ] Test unitari dell'inerzia
- [ ] `submission.py` + `scripts/make_submission.py`
- [ ] `validate_submission.py` ok
- [ ] Prima valutazione sui 5 minuti pubblici, risultato nel registro (controllo, non tuning)

## 9. Problemi noti

- D-OUT: uscite di 2 frame considerate D; serve la plausibilità dell'evento, non il margine
  geometrico.
- Lo z-score con σ = rumore della palla satura sempre (D-OUT mediana 32): la forza non discrimina.
- Palla assente nel 41% dei frame valutati, proprio nelle situazioni contese: limite principale
  della v0 (IMPROVEMENTS B1, B2).
- D-GOAL: un goal "no_candidate" (solo il calcio d'inizio) e due tag GOALS dello ST senza goal
  rilevato, da guardare nel video.

---

## Registro

| Data | Chi | Cosa |
|---|---|---|
| 2026-10-28 | Beppe | creato il file; stato iniziale ricostruito dal codice locale al 27/10 |
