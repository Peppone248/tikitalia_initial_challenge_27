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
[MOTORE.md](MOTORE.md) (come funziona il motore) ·
[FEATURES.md](FEATURES.md) (le misure per frame) ·
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

## 2. Feature per frame — `features.py`

Calcolate in `pipeline.build_context` → `context["features"]`. Test: `tests/test_features.py`
(scenari sintetici). Controllo su BAR-ATH: `scripts/inspect_features.py` (solo aggregati).
Spiegazione completa: [FEATURES.md](FEATURES.md). Controllo su BAR-ATH del 28/10: valori
plausibili (registro).

| Pezzo | Stato | Owner | Note |
|---|---|---|---|
✅ = testata su sintetico **e** quantili plausibili su BAR-ATH. 🟡 = il motivo è nella nota.

| Pezzo | Stato | Owner | Note |
|---|---|---|---|
| Velocità dei giocatori (smussate, scambi di slot > `V_PLAYER_MAX` scartati) | ✅ | Beppe | passi scartati 0,11%; limite: non definita ai bordi delle tracce spezzate |
| Giocatore più vicino per squadra: `nearest_*` (dist, slot, pos, vel) | ✅ | Beppe | mediana 1,91 m, coerente con `profile.txt`; rischio slot arbitri (ultima riga) |
| `nearest_team`, `nearest_gap`, `n_within(team, r)` | ✅ | Beppe | gap mediano 3,5 m |
| Palla–giocatore: `rel_speed`, `align_cos` | ✅ | Beppe | `rel_speed` mediana 3,7 m/s; `align_cos` non stampato ma dipende dalle stesse velocità |
| Palla–giocatore: `closing_speed`, `chase_speed` | 🟡 | Beppe | solo sintetico: non ancora in `inspect_features.py` |
| Pressione: `pressure_dist` (portatore → avversario più vicino) | ✅ | Beppe | mediana 5,1 m |
| Palla: `ball_acc`, `ball_missing_for` | 🟡 | Beppe | solo sintetico; `ball_acc` serve per stimare `ACC_KICK` |
| Forma: `n_visible`, `centroid_fwd`, `spread` | ✅ | Beppe | coerenti con `attack_dir` (casa più avanzata) |
| Forma: `n_opp_half`, `n_opp_box`, `n_own_box`, `v_forward` | 🟡 | Beppe | solo sintetico |
| Ritmo collettivo: `speed_med`, `speed_ref`, `slowdown` (H, A, tutti) | ✅ | Beppe | ≤ 0,5 sul 7,8%; limiti noti (§9): soste > `REF_WIN`, NaN sul ~5% |
| Rumore: `SIGMA_PLAYER_POS`, σ di distanza e velocità | 🟡 | Beppe | **problema reale**: σ sul pavimento, non misura niente (REGOLE §13) |
| Slot di arbitri / doppioni assegnati a una squadra | ⬜ | Gio | falsano `nearest_*` e conteggi; dipende dai controlli sugli slot |

### 2.1 Da discutere — call di venerdì 30/10: σ e la forza delle regole

Spiegazione completa con esempi: [FEATURES.md](FEATURES.md) §4.7.

**Cos'è σ.** L'errore tipico di misura di una feature: di quanto "trema" la posizione del tracker
tra un frame e l'altro. Si stima senza etichette: si smussa la traccia (media mobile di 5 frame),
si prende il residuo grezza − smussata, se ne misura l'ampiezza tipica (1,4826 × MAD).

**A cosa serve.** Alla forza di ogni condizione: `z = margine / σ`, forza = `min(1, z / 3)`. Una
condizione soddisfatta di poco rispetto a σ è incerta, una soddisfatta di molto è sicura.

**Il problema.** Su BAR-ATH σ stimato è sotto il pavimento `SIGMA_POS_FLOOR` = 0,10 m, sia per la
palla sia per i giocatori, quindi vale sempre 0,10: non lo misuriamo, lo imponiamo. Il tracking
che ci hanno dato è già corretto e smussato, il tremolio è stato tolto a monte. Conseguenze:
1. **la forza non distingue**: con σ = 0,10 m (0,14 per una distanza palla–giocatore) bastano
   40–50 cm di margine per saturare. D-OUT ha z mediano 32: quasi tutte le attivazioni hanno
   forza 1, giuste o sbagliate;
2. **è il rumore sbagliato**: il tracker dà il baricentro del giocatore, la palla è ai piedi
   (30–80 cm), lo smoothing a monte può spostare tutto di decimetri senza tremolio visibile.
   L'incertezza vera di "controlla la palla?" è dell'ordine del metro, non dei 10 cm del sensore.

**Proposta** (REGOLE §13): per ogni condizione `σ_i = max(σ sensore, tolleranza semantica)`, con
la tolleranza a priori e motivata in `params.yaml`.

| Condizione | Tolleranza proposta | Perché |
|---|---|---|
| distanza giocatore–palla (controllo) | ~0,5 m | baricentro vs piedi, falcata |
| palla oltre la linea | ~0,5–1 m | diametro della palla, smoothing vicino alle linee |
| giocatore dentro l'area | ~2 m | posizioni a ridosso della linea dell'area |
| velocità relativa palla–giocatore | ~1 m/s | tocchi a ogni passo in conduzione |

Effetto, controllo con `R_CONTROL` = 1,5 m:

| Distanza | Oggi (σ = 0,14 m) | Con tolleranza 0,5 m |
|---|---|---|
| 1,45 m | z 0,4 → forza 0,12 | z 0,1 → forza 0,03 |
| 1,20 m | z 2,1 → forza 0,71 | z 0,6 → forza 0,20 |
| 1,00 m | z 3,6 → forza 1 | z 1,0 → forza 0,33 |
| 0,30 m | z 8,6 → forza 1 | z 2,4 → forza 0,80 |

**Decisioni da prendere**
- [ ] Adottiamo `σ_i = max(σ sensore, tolleranza semantica)`? Alternative: forza = 1 per le
      regole dove lo z non informa; oppure stimare `p(z)` dalla calibrazione su altre partite
      (M1, NOMENCLATURA §4) quando ci saranno.
- [ ] Valori delle tolleranze: chi li propone, per quali condizioni (partire da quelle della v0:
      distanza di controllo, velocità relativa, palla oltre la linea).
- [ ] Va deciso **prima** di `control_episode` e P-DRIB, la cui forza dipende da questo.
- [ ] Piccola modifica utile: stampare anche σ stimato **prima** del pavimento, per sapere quanto è
      liscio il tracking (0,08 m o 0,005 m?) e scriverlo nel documento tecnico.

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

- [x] `features.py`: `nearest[team]`, `nearest_team` (+ tutte le altre del §2)
- [x] `scripts/inspect_features.py` su BAR-ATH: valori plausibili (vedi registro 28/10)
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
  Confermato dalle feature: σ stimato di palla **e** giocatori finisce sul pavimento (0,10 m), il
  tracking corretto è già smussato. Serve σ = max(sensore, tolleranza semantica) (REGOLE §13).
- Velocità mediana dei giocatori non definita sul ~5% dei frame valutati, anche con ≥ 9
  giocatori visibili: probabile frammentazione delle tracce (velocità NaN ai bordi o scartate).
- `slowdown` > 2 sul 10% dei frame: dopo una sosta lunga il riferimento è basso e la ripresa
  sembra un'accelerazione. Le regole leggono `slowdown < 1`, ma va tenuto presente.
- Palla assente nel 41% dei frame valutati, proprio nelle situazioni contese: limite principale
  della v0 (IMPROVEMENTS B1, B2).
- D-GOAL: un goal "no_candidate" (solo il calcio d'inizio) e due tag GOALS dello ST senza goal
  rilevato, da guardare nel video.

---

## Registro

| Data | Chi | Cosa |
|---|---|---|
| 2026-10-28 | Beppe | creato il file; stato iniziale ricostruito dal codice locale al 27/10 |
| 2026-10-28 | Beppe | aggiunto `docs/MOTORE.md` (descrizione del motore con esempi); prossimo passo: `features.py` |
| 2026-10-28 | Beppe | `features.py` v1 + 4 parametri in `params.yaml` + `tests/test_features.py` (39/39 test ok, sintetici) + `scripts/inspect_features.py`; feature collegate in `build_context` |
| 2026-10-28 | Beppe | `inspect_features.py` su BAR-ATH: 4,2 s; direzione d'attacco concorde tra calcio d'inizio e portiere in entrambi i tempi; passi > 12 m/s 0,11%; distanza mediana del più vicino 1,91 m; `slowdown` ≤ 0,5 sul 7,8% dei frame; σ sul pavimento (§9) |
| 2026-10-28 | Beppe | docs: nuovo `FEATURES.md`; `REGOLE_PSEUDO_LABEL.md` v0.5 (§2.3 nomi del codice, §3 parametri, §12 struttura, §13 σ); `NOMENCLATURA.md` §7 feature; link a FEATURES in MOTORE, POSSIBLE_PATHS, STATO |
| 2026-10-28 | Beppe | §2: stati delle feature aggiornati dopo il controllo su BAR-ATH (✅ quelle verificate; 🟡 solo σ e le feature non ancora stampate da `inspect_features.py`) |
| 2026-10-28 | Beppe | §2.1: punto su σ preparato per la call di venerdì 30/10 (spiegazione, proposta, decisioni da prendere) |
