# Modello di sequenza bidirezionale — piano di implementazione

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** una TCN bidirezionale che dà per frame P(H/A/D) e P(cambio di possesso), decodificata da un
HMM e misurata in leave-one-match-out su IDSSE contro lo stesso pipeline senza sequenza.

**Architecture:** cartella autonoma `sequence_model/` (package `seqmodel`). Pipeline: partita nel
formato comune → degrado "come BAR-ATH" → feature simmetriche H/A → modello (classificatore per frame
oppure TCN) → HMM con durate minime e cambi guidati da P(cambio) → scorer ufficiale.

**Tech Stack:** Python 3.13 (Anaconda), numpy, pandas, PyYAML, pytest; PyTorch CPU solo per
l'addestramento; inferenza in numpy.

**Spec:** `docs/superpowers/specs/2026-11-12-sequence-model-design.md`

## Global Constraints

- 25 fps; un "frame" è sempre un frame a 25 fps.
- Etichette: codici interi `H=0, A=1, D=2, X=3` (gli stessi di `evaluation/possession_metrics.label_codes`).
- Etichette "controllate": un tratto di squadra più corto di 1 s (25 frame) tra due tratti dell'altra squadra prende l'etichetta dell'altra.
- Durate minime nell'HMM: 0,8 s (20 frame) per H e A, 2 s (50 frame) per D.
- Finestra del ramo dei cambi: k = 12 frame.
- Degrado: palla morta nascosta con probabilità 0,8; due seed (2027, 2028).
- Criterio per tenere il modello: TCN completa − classificatore per frame ≥ 0,01 di spellF1, ≥ 5 partite su 7, con entrambi i seed.
- Nessun numero di frame o di slot di BAR-ATH nel codice; nessuna lettura di `evaluation/gt/` tranne lo scorer.
- Inferenza deterministica: input arrotondati a 2^-16, probabilità arrotondate a 3 decimali.
- La memoria non attraversa mai l'intervallo: ogni tempo (`period`) è una sequenza separata.
- Nessuna GPU sul PC: modello piccolo (32 canali, 6 livelli) addestrato su CPU.
- Ogni costante sta in `sequence_model/config.yaml` con il suo perché.
- Commit su un branch `francesco/sequence-model`, solo con l'ok di Francesco.

## Review Focus

1. **Giocatori mancanti** (espulsione, sostituzione, tracce spezzate, colonne tutte NaN): le feature devono restare finite e il codice non deve andare in errore. Test in Task 3.
2. **Lunghi tratti senza palla** (60 s e oltre, come su BAR-ATH): le feature della palla restano finite, con il flag "non vista" e i secondi di assenza che crescono. Test in Task 3.
3. **Cambio di tempo:** la direzione d'attacco si inverte, e l'HMM non deve portare lo stato da un tempo all'altro. Test in Task 3 e Task 4.
4. **Frame con etichetta X** (possesso mancante in DFL): esclusi dalla loss e dalla misura, senza far fallire nulla. Test in Task 5 e Task 6.
5. **Probabilità degeneri** (0 o 1 esatti, uniformi): l'HMM non produce NaN o −inf su tutto il percorso. Test in Task 4.

---

## Struttura dei file

```text
sequence_model/
  README.md                 come si installa e si lancia
  requirements.txt          numpy, pandas, pyyaml, pytest, torch (cpu)
  config.yaml               costanti con il perché
  seqmodel/__init__.py
  seqmodel/match.py         formato comune `Match`, etichette controllate, caricatore IDSSE
  seqmodel/degrade.py       degrado come BAR-ATH
  seqmodel/features.py      feature per frame
  seqmodel/decode.py        HMM a 3 stati
  seqmodel/frame_model.py   classificatore per frame (logistica numpy, termine di paragone)
  seqmodel/tcn.py           TCN PyTorch + addestramento + export pesi
  seqmodel/infer.py         TCN in numpy, deterministica
  seqmodel/evaluate.py      leave-one-match-out + scorer ufficiale
  scripts/fetch_idsse.py    copiato dal branch approach-b-broadcast
  scripts/run_lomo.py       tabella dei 4 confronti × 2 seed
  scripts/predict.py        probs.npz + CSV su una partita (anche BAR-ATH)
  tests/conftest.py         partita sintetica
  tests/test_*.py
```

Tutti i comandi si lanciano dalla root della repo con
`C:\Users\PC\anaconda3\python.exe -m pytest sequence_model/tests -q`
(nel seguito `PY` = `C:\Users\PC\anaconda3\python.exe`).

---

### Task 1: formato comune, etichette controllate, IDSSE

**Files:**
- Create: `sequence_model/{README.md,requirements.txt,config.yaml}`, `sequence_model/seqmodel/__init__.py`, `sequence_model/seqmodel/match.py`, `sequence_model/scripts/fetch_idsse.py`, `sequence_model/tests/conftest.py`, `sequence_model/tests/test_match.py`
- Modify: `.gitignore` (aggiungi `sequence_model/data/` e `sequence_model/runs/`)

**Interfaces:**
- Produces:
  - `@dataclass Match: name: str; px: np.ndarray (T,P) float32; py: np.ndarray (T,P); team: np.ndarray (P,) int8 (0=H,1=A); ball_x: np.ndarray (T,); ball_y: np.ndarray (T,); period: np.ndarray (T,) int8; label: np.ndarray (T,) int8; length: float; width: float; home_attacks_right: dict[int,bool]`. Coordinate in metri, origine al centro del campo.
  - `controlled_labels(label: np.ndarray, min_frames: int = 25) -> np.ndarray`
  - `attack_direction(m: Match) -> dict[int, bool]` (da dove stanno i portieri: il giocatore più arretrato di ogni squadra, mediana del tempo)
  - `load_idsse(path: str) -> Match` (legge il `.npz` di `fetch_idsse.py`; `label`: D se `ball_alive` è falso, altrimenti H/A da `ball_possession`, X se 0)
  - fixture pytest `synthetic_match(seed=0, n_frames=3000) -> Match`: 22 giocatori, 2 tempi, possessi alternati di 200–400 frame, palla vicina a un giocatore della squadra in possesso, tratti D di 100 frame.

- [ ] **Step 1: test** in `tests/test_match.py`:
  - `test_controlled_labels_removes_short_flip`: `[H]*50+[A]*10+[H]*50` → tutto H; `[H]*50+[A]*30+[H]*50` resta invariato; i D non vengono mai toccati.
  - `test_attack_direction_flips_between_halves`: nella partita sintetica il valore del tempo 1 è l'opposto di quello del tempo 2.
  - `test_load_idsse_roundtrip`: un `.npz` scritto dal test con le chiavi di `fetch_idsse.py` (`period, ball_x, ball_y, ball_alive, ball_possession, px, py, player_team, pitch_x, pitch_y`) → `label` uguale a quella attesa, campo 0 → 105×68.
- [ ] **Step 2:** `PY -m pytest sequence_model/tests/test_match.py -q` → FAIL (modulo mancante).
- [ ] **Step 3:** implementa `match.py`; copia `HMM_first_imp/validation/fetch_idsse.py` dal branch (`git show origin/approach-b-broadcast:HMM_first_imp/validation/fetch_idsse.py`) in `scripts/fetch_idsse.py`, con l'output in `sequence_model/data/idsse/`.
- [ ] **Step 4:** test → PASS.
- [ ] **Step 5:** lancia `PY sequence_model/scripts/fetch_idsse.py` (scarica le 7 partite; alcuni GB) e verifica con `load_idsse` che ogni partita abbia due tempi e circa il 45% di D.
- [ ] **Step 6:** commit `feat(seqmodel): common match format, controlled labels, IDSSE loader`.

### Task 2: degrado come BAR-ATH

**Files:** Create `sequence_model/seqmodel/degrade.py`, `sequence_model/tests/test_degrade.py`

**Interfaces:**
- Consumes: `Match`
- Produces: `degrade(m: Match, seed: int, cfg: dict) -> Match` (copia: `ball_x/ball_y` a NaN nei frame nascosti, `px/py` a NaN nelle tracce spezzate; `label` invariata). `cfg` da `config.yaml → degrade`: `dead_hide_p: 0.8`, `live_gap_s: [1, 3, 8, 20]` (quantili a priori della lunghezza dei buchi a palla viva), `live_seen_target: 0.6`, `player_drop_frac: 0.07`.

- [ ] **Step 1: test:**
  - `test_degrade_is_deterministic`: stesso seed → array identici; seed diverso → diversi.
  - `test_labels_untouched`.
  - `test_dead_hidden_more_than_live`: quota di palla vista a palla morta < quota a palla viva.
  - `test_live_seen_share`: quota di palla vista a palla viva in [0,5; 0,7].
- [ ] **Step 2:** FAIL. **Step 3:** implementa con `np.random.default_rng(seed)`. **Step 4:** PASS. **Step 5:** commit.

### Task 3: feature per frame

**Files:** Create `sequence_model/seqmodel/features.py`, `sequence_model/tests/test_features.py`

**Interfaces:**
- Consumes: `Match`, `attack_direction`
- Produces:
  - `FEATURES: list[str]`
  - `ODD: np.ndarray[bool]`: True per le feature che cambiano segno scambiando le squadre.
  - `frame_features(m: Match) -> np.ndarray (T, F) float32`, senza NaN.
  - `swap_teams(m: Match) -> Match` (team 0↔1, H↔A nelle etichette, direzione d'attacco invertita)

  Feature, nell'ordine dello spec §5.4: `ball_seen, ball_unseen_s` (fino a 60 s), `ball_speed`, `ball_line_dist`, `ball_x_att`; H−A di `nearest_dist`, `n_within_3`, `n_within_8`; H−A di `centroid_x, def_line, front_line, width, length, v_forward, mean_speed`; `mean_speed_all`; `ball_out`, `slow_unseen`. Le coordinate x vanno sempre espresse nella direzione d'attacco di H. Dove la palla non si vede, le feature della palla usano l'ultima posizione vista e i secondi di assenza.

- [ ] **Step 1: test:**
  - `test_swap_antisymmetry`: `frame_features(swap_teams(m))[:, ODD] == -frame_features(m)[:, ODD]` e le colonne pari sono uguali (tolleranza 1e-5).
  - `test_no_nan_with_missing_players`: azzera a NaN 3 colonne intere e 200 frame di una squadra → `np.isfinite` ovunque.
  - `test_long_ball_gap`: 1500 frame senza palla → `ball_seen == 0` e `ball_unseen_s` che cresce fino a 60.
  - `test_attack_dir_used`: invertendo la direzione d'attacco di un tempo cambia segno `ball_x_att` in quel tempo.
- [ ] **Step 2:** FAIL. **Step 3:** implementa (velocità dalle differenze tra posizioni smussate su 5 frame). **Step 4:** PASS. **Step 5:** commit.

### Task 4: HMM a 3 stati con cambi guidati

**Files:** Create `sequence_model/seqmodel/decode.py`, `sequence_model/tests/test_decode.py`

**Interfaces:**
- Produces:
  - `decode(p: np.ndarray (T,3), p_change: np.ndarray (T,) | None, period: np.ndarray (T,), w: float = 1.0, beta: float = 0.0, trans: dict | None = None, prior: np.ndarray | None = None) -> np.ndarray (T,) int8` (codici 0..2)
  - `count_transitions(labels: list[np.ndarray], periods: list[np.ndarray]) -> dict` (tassi per frame H→A, H→D, A→H, A→D, D→H, D→A, con pseudo-conteggio 1)

  Viterbi separato per ogni tempo. Stati espansi: catene di Bakis (20 sotto-stati per H e A, 50 per D); si esce solo dall'ultimo sotto-stato. Emissione = `w*(log(p+1e-6) - log(prior))`. Al frame t, la transizione H↔A vale `rate*(1+beta*p_change[t])`, rinormalizzata.

- [ ] **Step 1: test:**
  - `test_min_durations`: probabilità che alternano H/A ogni 5 frame → nessun tratto H/A < 20, nessun D < 50.
  - `test_constant_probs_no_change`: P(H)=0,9 costante → tutto H.
  - `test_beta_moves_change`: con P ambigua (0,5/0,5) su 200 frame dopo un tratto H sicuro e prima di un tratto A sicuro, un picco di `p_change` al frame 120 con β=20 mette il cambio a 120±12.
  - `test_degenerate_probs`: probabilità esattamente 0/1 e uniformi → nessun NaN, uscita in {0,1,2}.
  - `test_halves_independent`: la fine del tempo 1 in A non impone A all'inizio del tempo 2.
- [ ] **Step 2:** FAIL. **Step 3:** implementa in numpy vettoriale (un passo per frame sul vettore degli stati espansi; 90 stati × 80k frame ≈ qualche secondo). **Step 4:** PASS. **Step 5:** commit.

### Task 5: classificatore per frame + leave-one-match-out

**Files:** Create `sequence_model/seqmodel/frame_model.py`, `sequence_model/seqmodel/evaluate.py`, `sequence_model/tests/test_evaluate.py`

**Interfaces:**
- Consumes: `frame_features`, `swap_teams`, `decode`, `count_transitions`, `degrade`, `controlled_labels`
- Produces:
  - `FrameModel.fit(X: (N,F), y: (N,) in {0,1,2}) -> FrameModel` (logistica multinomiale, L2=1, Newton, deterministica; ogni frame visto anche con le squadre scambiate); `.predict_proba(X) -> (N,3)`
  - `lomo(matches: list[Match], make_model: Callable, seed: int, cfg: dict, use_change: bool) -> pd.DataFrame` (colonne `match, spellF1, macroF1, n_pred_poss, n_true_poss, transF1_25`). Allena su 6, valuta sulla settima. Le etichette X sono escluse dall'addestramento; lo scorer è `possession_metrics.evaluate(gt, pred)`, con `gt` = etichette controllate e X sui frame non valutati.

- [ ] **Step 1: test** (3 partite sintetiche):
  - `test_lomo_runs`: 3 righe, spellF1 in [0,1].
  - `test_x_frames_ignored`: con il 10% delle etichette a X, l'addestramento non va in errore e i frame X non entrano nella loss (le X non sono mai previste).
  - `test_frame_model_symmetric`: `predict_proba` su X scambiato = H e A invertite.
- [ ] **Step 2:** FAIL. **Step 3:** implementa; importa lo scorer con `sys.path.insert(0, <repo>/evaluation)`. **Step 4:** PASS.
- [ ] **Step 5:** su IDSSE: `PY sequence_model/scripts/run_lomo.py --models baseline50 frame --seeds 2027` (`run_lomo.py` creato qui, con baseline50 = giocatore più vicino + voto ±50 frame + D dove la palla non si vede). Atteso: baseline50 ≈ 0,2; frame model sopra. Annota i numeri.
- [ ] **Step 6:** commit.

### Task 6: TCN bidirezionale (addestramento)

**Files:** Create `sequence_model/seqmodel/tcn.py`, `sequence_model/tests/test_tcn.py`; aggiungi `torch` (CPU) a `requirements.txt` e installalo con `PY -m pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Interfaces:**
- Produces:
  - `change_targets(label: np.ndarray, period: np.ndarray, k: int = 12) -> np.ndarray (T,) float32`: 1 entro ±k frame da un cambio H↔A diretto (D escluso), 0 altrove.
  - `class TCN(nn.Module)`: `__init__(n_in, channels=32, levels=6, kernel=5)`; blocchi residui con convoluzioni non causali, dilatazione 2^i, padding "same". `forward(x: (B,F,T)) -> (logits3: (B,3,T), logit_change: (B,1,T))`.
  - `symmetric_forward(model, x, odd_mask) -> (p3, p_change)`: media di f(x) e di swap(f(swap(x))).
  - `train_tcn(matches: list[Match], cfg: dict, seed: int) -> TCN` (finestre di 60 s = 1500 frame, Adam lr 1e-3, 15 epoche al massimo, early stop sull'ultima partita del training, loss = CE pesata (X esclusi) + 0,5·BCE sui cambi con pos_weight = negativi/positivi)
  - `export_weights(model, path: str)` → `.npz` con i pesi di ogni strato, chiamati `l{i}.conv.w` ecc.

- [ ] **Step 1: test:**
  - `test_change_targets`: un cambio H→A al frame 100 → target 1 sui frame 88..112, 0 altrove; un passaggio H→D→A non produce target.
  - `test_noncausal`: perturbare l'input al frame t+20 cambia l'uscita al frame t.
  - `test_symmetry_exact`: le uscite di `symmetric_forward` su x scambiato hanno H↔A invertite (tolleranza 1e-6), con p_change uguale.
  - `test_overfits_tiny`: su una partita sintetica, dopo 200 passi, accuratezza per frame > 0,9.
- [ ] **Step 2:** FAIL. **Step 3:** implementa con `torch.manual_seed(seed)`, `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(4)`. **Step 4:** PASS. **Step 5:** commit.

### Task 7: inferenza numpy deterministica

**Files:** Create `sequence_model/seqmodel/infer.py`, `sequence_model/tests/test_infer.py`

**Interfaces:**
- Consumes: il `.npz` di `export_weights`
- Produces: `NumpyTCN.load(path) -> NumpyTCN`; `.predict(X: (T,F), odd_mask) -> (p3: (T,3), p_change: (T,))`. Input arrotondati a 2^-16, uscite arrotondate a 3 decimali, stessa simmetria di `symmetric_forward`.

- [ ] **Step 1: test:**
  - `test_matches_torch`: su input casuale, |numpy − torch| < 1e-4 prima dell'arrotondamento.
  - `test_deterministic`: due chiamate → output identici byte per byte (`hashlib.sha256`).
  - `test_rounding_stable`: un rumore di 1e-9 sull'input non cambia l'output arrotondato.
- [ ] **Step 2:** FAIL. **Step 3:** implementa la convoluzione dilatata con `np.lib.stride_tricks.sliding_window_view` + `einsum`. **Step 4:** PASS. **Step 5:** commit.

### Task 8: la tabella dei confronti (decisione del 14)

**Files:** Modify `sequence_model/scripts/run_lomo.py`; Create `sequence_model/runs/README.md` (i risultati vanno in `sequence_model/runs/lomo_<data>.md`)

**Interfaces:**
- Consumes: `lomo`, `FrameModel`, `train_tcn`, `NumpyTCN`
- Produces: `run_lomo.py --models baseline50 frame tcn tcn_change --seeds 2027 2028` → una tabella markdown con lo spellF1 medio per riga e seed, il delta appaiato contro `frame`, le partite migliori/peggiori, possessi previsti/veri, transF1@25; in fondo la riga "criterio: SODDISFATTO / NON SODDISFATTO".

`tcn` = TCN con β=0; `tcn_change` = β e w scelti tra `beta ∈ {0, 5, 20, 50}` e `w ∈ {0.5, 1.0}` sulle 6 partite di training di ogni turno.

- [ ] **Step 1: test** `test_criterion`: dati delta inventati per 7 partite × 2 seed, la funzione `criterion(deltas: dict[int, list[float]]) -> bool` restituisce True solo se la media è ≥ 0,01 e ci sono ≥ 5 partite positive in ogni seed.
- [ ] **Step 2:** FAIL. **Step 3:** implementa. **Step 4:** PASS.
- [ ] **Step 5:** lancia la tabella completa su IDSSE. Salva in `runs/lomo_2026-11-14.md`.
- [ ] **Step 6:** commit. **Decisione con Francesco**: se il criterio non è soddisfatto, si scrive il risultato negativo e ci si ferma qui.

### Task 9: previsione su una partita (anche BAR-ATH)

**Files:** Create `sequence_model/scripts/predict.py`, `sequence_model/tests/test_predict.py`; Modify `sequence_model/seqmodel/match.py` (aggiunge `load_atd(path_xml, path_txt) -> Match` tramite `starter/load_atd.py`, con `label` tutta X)

**Interfaces:**
- Consumes: `NumpyTCN`, `frame_features`, `decode`
- Produces: `predict.py --weights <npz> --match <idsse npz | atd> --out <dir>` → `probs.npz` (`p_h, p_a, p_d, p_change`, float32, una riga per frame video) e `labels.csv` (`frame,label`, X fuori dai tempi valutati).

- [ ] **Step 1: test** `test_predict_writes_files`: su una partita sintetica i file esistono, `probs.npz` ha T righe, `labels.csv` ha T righe più l'intestazione e solo etichette in {H,A,D,X}.
- [ ] **Step 2:** FAIL. **Step 3:** implementa. **Step 4:** PASS.
- [ ] **Step 5:** quando ci sono i dati: su BAR-ATH, `PY evaluation/validate_submission.py <out>/labels.csv` → validazione superata; due esecuzioni → stesso SHA-256. **Nessuna lettura della ground truth pubblica qui.**
- [ ] **Step 6:** commit; consegna `probs.npz` a Beppe, con il formato descritto in `sequence_model/README.md`.

### Task 10 (solo se avanza tempo): Metrica

**Files:** Modify `sequence_model/seqmodel/match.py` (`load_metrica(game_id) -> Match` dai dati di esempio pubblici via kloppy; etichette dagli eventi: palla viva tra una ripresa e l'evento successivo di palla fuori o fallo; squadra = squadra dell'ultimo evento con la palla); Test `test_load_metrica_labels` su un piccolo estratto di eventi scritto nel test.
- Rilancia la tabella del Task 8 con 10 partite.
