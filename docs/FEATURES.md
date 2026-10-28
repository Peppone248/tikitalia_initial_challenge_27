# Le feature per frame

Cosa misura `src/tikitalia/features.py`, come lo misura, perché serve e cosa abbiamo visto su
BAR-ATH. Termini: [NOMENCLATURA.md](NOMENCLATURA.md) (§2 rumore, §7 feature). Regole che le usano:
[REGOLE_PSEUDO_LABEL.md](REGOLE_PSEUDO_LABEL.md) §2.3. Posizione nella pipeline:
[MOTORE.md](MOTORE.md) §2.

---

## 1. In una frase

Le feature sono **misure, non decisioni**: dicono quanto distano palla e giocatori, quanto corrono,
come sono disposte le squadre. Non dicono mai chi ha il possesso. Quello lo dicono le regole,
confrontando le feature con le soglie, e alla fine il motore.

```text
Match ─► features ─► events ─► regole ─► motore
misure   misure      fatti      opinioni  decisione
pulite   derivate    nel tempo  pesate    unica
```

Esempio di come si dividono il lavoro sullo stesso frame:

| Strato | Cosa dice | Esempio |
|---|---|---|
| feature | una misura | "il giocatore H più vicino è a 0,8 m, si muove a 4 m/s, la palla a 4,2 m/s nella stessa direzione" |
| evento | un fatto con durata | "H è in controllo della palla da 1,3 s" (`control_episode`) |
| regola | un'opinione pesata | "P-DRIB: possesso H, forza 0,7" |
| motore | la decisione | "frame = H" |

## 2. Principi

1. **Niente soglie, niente etichette.** Se una funzione restituisce vero/falso su un concetto di
   calcio ("in controllo", "palla ferma per una ripresa"), non è una feature: è un evento o una
   regola. Le uniche costanti ammesse sono finestre di calcolo e guardie di validità
   (`SMOOTH_WIN`, `REF_WIN`, `V_PLAYER_MAX`, `TEMPO_MIN_PLAYERS`), tutte in `params.yaml` con origine e
   motivazione.
2. **Calcolate una volta, condivise da tutti.** La distanza dalla palla serve a P-DRIB, P-FREE,
   R-TOUCH, R-LAST, T-*: se ognuna se la calcolasse, prima o poi due regole userebbero due
   definizioni diverse della stessa cosa. Si calcolano in `pipeline.build_context` e le regole le
   leggono da `context["features"]`.
3. **Un valore per frame, vettorizzato.** Array numpy indicizzati per frame video (199.319 righe),
   nessun loop Python per frame. Tempo su BAR-ATH: circa 4 s insieme agli eventi.
4. **NaN = non misurabile**, mai zero. Palla non vista, giocatore non tracciato, troppo pochi
   giocatori per una mediana, frame fuori dai tempi valutati per le misure relative alla
   direzione d'attacco. Le regole decidono cosa fare di un NaN (di solito astenersi).
5. **Con il loro rumore.** La forza di una regola è uno z-score (NOMENCLATURA §2): serve σ di ogni
   feature, stimato senza etichette. `features.py` lo calcola insieme alle feature.
6. **Relative alla squadra, non al campo.** Tutto ciò che dipende dalla direzione ("avanti",
   "metà avversaria", "area propria") si calcola con `attack_dir`, così vale uguale nei due tempi.

## 3. Convenzioni

| Forma | Significato |
|---|---|
| `(n,)` | un valore per frame |
| `(n, 2)` | per squadra: colonna 0 = HOME, 1 = AWAY |
| `(n, 3)` | per squadra + tutti: colonne HOME, AWAY, ALL (= 2) |
| `(n, n_slots)` | per traccia (46 slot nell'ATD, ~22 giocatori reali) |

Unità: metri, secondi, m/s. Coordinate: `x ∈ [0, 105,3]`, `y ∈ [0, 68]`. Le velocità sono
**derivate dalle posizioni**, non dal campo `s` dell'ATD (unità ancora da chiarire).

---

## 4. I gruppi, uno per uno

### 4.1 Cinematica dei giocatori — `pvx`, `pvy`, `pspeed`

**Cosa misura.** La velocità di ogni traccia, frame per frame.

**Come.** Stessa procedura della palla (`preprocessing.smooth_and_differentiate`): media mobile
centrata su `SMOOTH_WIN` = 5 frame (0,2 s) su ogni tratto continuo della traccia, poi differenze
centrali. Un passo più veloce di `V_PLAYER_MAX` = 12 m/s non è uno sprint (i più veloci in partita
arrivano a 10–11 m/s, il record umano è ~12,4) ma uno **scambio di identità** tra due tracce o un
salto del tracker: la velocità lì diventa NaN, la posizione resta.

**Perché.** Le distanze non bastano. "Il giocatore porta la palla" vuol dire che si muovono
**insieme**; il fischio si riconosce perché **tutti rallentano**; un recupero si vede quando una
squadra **inverte la corsa**. Tutto questo è velocità.

**BAR-ATH.** Passi oltre 12 m/s: 0,11% (tracker pulito). Velocità dei giocatori nei frame valutati:
mediana 1,4 m/s, 90° percentile 3,8 m/s.

**Limite.** La velocità non esiste ai bordi di una traccia spezzata: dove le tracce si
frammentano molto, la velocità manca anche se il giocatore è visibile.

### 4.2 Palla — `ball_acc`, `ball_missing_for`

**Cosa misura.**
- `ball_acc`: modulo dell'accelerazione della palla (m/s²), dalle velocità già calcolate in
  `preprocessing.prepare_ball`.
- `ball_missing_for`: secondi dall'ultimo frame in cui la palla era visibile (0 se visibile, NaN
  prima del primo avvistamento).

**Perché.** Un calcio è un picco di accelerazione: serve a `restart_kick` (palla ferma → calcio) e al
release di `pass_event` (`ACC_KICK`, da ricavare dalla distribuzione). Da quanto manca la palla
serve a D-GAP (buchi lunghi) e a D-SHOT (palla persa durante un tiro).

### 4.3 Palla ↔ giocatori

È il gruppo più importante: da qui passa quasi ogni regola di possesso.

**Distanze e giocatore più vicino**

| Feature | Forma | Cosa misura |
|---|---|---|
| `dist_ball` | `(n, slots)` | distanza di ogni traccia dalla palla |
| `nearest_dist` | `(n, 2)` | distanza del più vicino di ogni squadra |
| `nearest_slot` | `(n, 2)` | quale traccia è (−1 se nessuna) |
| `nearest_x/y`, `nearest_vx/vy` | `(n, 2)` | sua posizione e velocità |
| `nearest_team` | `(n,)` | squadra del più vicino in assoluto (−1 se la palla non si vede) |
| `nearest_gap` | `(n,)` | quanto l'altra squadra è più lontana: `|d_H − d_A|` |
| `n_within(team, r)` | metodo | giocatori di `team` entro `r` m dalla palla |

`n_within` è un metodo e non un array perché il raggio `r` è una soglia, e la soglia appartiene
alla regola che la usa (`R_CONTROL`, `WALL_DIST`, …), non alla feature.

`nearest_gap` misura quanto è **conteso** il pallone: con un gap di 0,3 m chi è "più vicino" lo
decide il rumore; con un gap di 5 m no.

**Come si muove la palla rispetto al giocatore più vicino**

| Feature | Formula | Valori tipici |
|---|---|---|
| `rel_speed` | `|v_palla − v_giocatore|` | ~0 se la palla segue il giocatore |
| `align_cos` | `cos(v_palla, v_giocatore)` | ~1 se vanno nella stessa direzione |

Sono la definizione di `control_episode` (REGOLE §2.4: `|ball.vel − nearest.vel| ≤ V_REL_CONTROL`)
e della condizione di P-DRIB (`cos ≥ 0,8`).

**Due velocità "verso la palla", da non confondere.** Con `u` = versore dal giocatore alla palla:

| Feature | Formula | Domanda |
|---|---|---|
| `closing_speed` | `(v_giocatore − v_palla) · u` | la distanza si sta accorciando? |
| `chase_speed` | `max` sulla squadra di `v_giocatore · u` | qualcuno sta correndo verso la palla? |

Esempio numerico (palla 1 m davanti al giocatore, tutto lungo x):

| Situazione | v giocatore | v palla | `closing_speed` | `chase_speed` |
|---|---|---|---|---|
| conduzione | 4 m/s | 4 m/s | 0 | 4 |
| inseguimento di una palla che rotola via | 6 m/s | 8 m/s | −2 (perde terreno) | 6 |
| palla ferma, giocatore che arriva | 3 m/s | 0 | 3 | 3 |
| dopo il fischio: palla calciata via, nessuno la segue | 0,5 m/s | 10 m/s | −9,5 | 0,5 |

La prima versione aveva una sola grandezza (`v_giocatore · u`), e chi conduce risultava "in
avvicinamento" a 4 m/s: l'ha trovato un test. Servono tutte e due: `closing_speed` per sapere se
un controllo sta per iniziare, `chase_speed` per la variante B di D-WHISTLE ("la palla viene
calciata via e nessuno la insegue"), dove l'inseguitore può essere più lento della palla.

**BAR-ATH** (frame valutati con palla visibile):

| Feature | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| più vicino in assoluto (m) | 0,44 | 0,89 | 1,91 | 4,05 | 6,99 |
| più vicino H (m) | 0,61 | 1,40 | 3,53 | 7,12 | 11,98 |
| più vicino A (m) | 0,90 | 2,10 | 4,65 | 8,35 | 12,49 |
| `nearest_gap` (m) | 0,58 | 1,48 | 3,52 | 7,26 | 11,68 |
| `rel_speed` della squadra più vicina (m/s) | 0,64 | 1,42 | 3,65 | 8,17 | 12,68 |

Il più vicino in assoluto è sotto i 2 m in metà dei frame, coerente con `profile.txt` (51,5%). Con
`R_CONTROL` = 1,5 m circa il 40% dei frame con palla ha qualcuno abbastanza vicino da poterla
controllare. In metà dei frame l'altra squadra è più lontana di 3,5 m: chi è vicino alla palla è
quasi sempre chiaro.

### 4.4 Pressione — `pressure_dist`

**Cosa misura.** Per ogni squadra: distanza tra il suo giocatore più vicino alla palla (il
probabile portatore) e l'avversario più vicino a lui.

**Perché è l'unica distanza giocatore–giocatore.** Le distanze tra tutte le coppie di giocatori
(46 × 46 per frame) sono quasi tutte rumore per il possesso: senza la palla non dicono chi ce l'ha.
Quella che conta è la pressione sul portatore: un portatore libero (P-FREE, `R_FREE_OPP`) ha il
possesso più sicuro di uno con un avversario addosso (contrasto, T-TACKLE). Le altre distanze tra
giocatori (barriera a 9,15 m per R-WALL) si aggiungeranno quando servono, come feature dedicate.

**BAR-ATH.** Mediana 5,1 m per la squadra di casa.

### 4.5 Forma delle squadre

Tutte relative ad `attack_dir` (calcolata in `events.attack_directions`): NaN fuori dai tempi
valutati, dove la direzione non è definita.

| Feature | Cosa misura | Per |
|---|---|---|
| `n_visible` | giocatori tracciati | affidabilità dei conteggi |
| `centroid_fwd` | baricentro lungo la direzione d'attacco: 0 = metà campo, > 0 = nella metà avversaria | P-SHAPE |
| `centroid_y` | baricentro trasversale | |
| `spread` | `sd_x + sd_y` della squadra (m) | P-SHAPE: chi attacca si allarga |
| `n_opp_half` | giocatori nella metà avversaria | P-SHAPE |
| `n_opp_box`, `n_own_box` | giocatori nell'area avversaria / propria | R-SET (corner, punizioni) |
| `v_forward` | velocità media della squadra verso la porta attaccata | T-TEAMFLIP, P-SHAPE |

**Perché relative.** "Avanzato" deve significare la stessa cosa nei due tempi, e per entrambe le
squadre. Con `fwd = (x − L/2) · attack_dir` un solo segno basta: > 0 è sempre "verso la porta che
attacco".

**Come si leggono.** La forma reagisce al possesso con secondi di ritardo: dopo un recupero chi
attaccava impiega tempo a rientrare. È quindi un'evidenza debole sulle transizioni (le regole che
la usano sono SOFT) e forte a palla ferma, dove la disposizione **anticipa** chi riprende (R-SET).
`v_forward` è l'eccezione: l'inversione della corsa di entrambe le squadre è uno dei pochi segnali
di cambio di possesso disponibili **quando la palla non si vede** (T-TEAMFLIP).

**BAR-ATH.** Giocatori visibili per squadra: mediana 11, 10° percentile 9. Baricentro di casa più
avanzato di quello ospite (mediane −3,6 m e −9,1 m), spread 29 m contro 26 m: coerente con una
squadra di casa che tiene più palla.

**Limite.** Slot di arbitri o doppioni assegnati a una squadra (POSSIBLE_PATHS §3.2: > 11 giocatori
di una squadra nel 12,9% dei frame) spostano baricentro e conteggi e possono diventare il
"giocatore più vicino". Dipende dai controlli sugli slot (task Gio).

### 4.6 Ritmo collettivo — `speed_med`, `speed_ref`, `slowdown`

**Cosa misura.** Quanto stanno correndo le squadre adesso, rispetto a quanto correvano poco prima.

**Come.**
1. `speed_med[f, g]`: mediana delle velocità dei giocatori del gruppo `g` (H, A, tutti) al frame `f`.
   NaN se i giocatori con velocità valida sono meno di `TEMPO_MIN_PLAYERS` = 6 per squadra (12 per
   tutti): con meno giocatori non è una misura collettiva.
2. `speed_ref[f, g]`: mediana delle mediane al secondo negli ultimi `REF_WIN` = 30 s (finestra
   all'indietro: "quanto correvano appena prima?"). Si calcola al secondo e si espande ai frame:
   costa poco ed è abbastanza preciso per un riferimento.
3. `slowdown = speed_med / speed_ref`: < 1 = più lenti del solito.

La mediana e non la media: un giocatore che scatta durante una sosta (va a prendere il pallone) non
deve far sembrare la squadra in movimento.

**Perché.** I falli, i fuorigioco e gli infortuni non hanno un segnale nella palla (lo Smart Tagging
ha solo 5 `Free Kick`): dopo il fischio **tutti rallentano**. È la base di D-WHISTLE e D-GAP. È
calcolato per squadra perché D-WHISTLE chiede il rallentamento di **entrambe**: una squadra lenta
da sola è un giro palla, non un fischio.

**Esempio** (tempo di gioco in secondi):

```text
t        0 ─────── 60   60 ── 75   75 ─────── 120
in campo gioco a 6 m/s  fischio     sosta lunga, tutti a 2 m/s
speed_med     6            2             2
speed_ref     6            6             scende verso 2
slowdown      1            0,33          risale verso 1
```

**Limite (voluto, da conoscere).** In una sosta più lunga di `REF_WIN` anche il riferimento si
abbassa, e `slowdown` torna verso 1. Al contrario, alla ripresa dopo una sosta lunga il riferimento
è basso e il gioco sembra un'accelerazione (`slowdown` > 2). Conseguenza per le regole: il
rallentamento indica **l'inizio** di un D, non quanto dura. La fine la danno `restart_kick` e D-HOLD.

**BAR-ATH.** Velocità mediana collettiva 1,4 m/s. `slowdown` ≤ 0,5 sul 7,8% dei frame valutati,
> 2 sul 10% (riprese dopo soste lunghe). Velocità mediana non definita sul ~5% dei frame anche con
9–11 giocatori visibili: tracce spezzate (§4.1).

### 4.7 Rumore — `sigma`

**Cosa misura.** L'errore tipico di ogni feature, per gli z-score della forza delle regole
(NOMENCLATURA §2, REGOLE §4.1).

| Chiave | Come si ottiene |
|---|---|
| `player_pos` | MAD robusto del residuo (posizione grezza − media mobile), su tutte le tracce; parametro stimato `SIGMA_PLAYER_POS`, con pavimento `SIGMA_POS_FLOOR` |
| `ball_pos` | stesso metodo sulla palla (`SIGMA_POS`, già in preprocessing) |
| `dist_ball_player` | `√(σ_palla² + σ_giocatore²)` |
| `player_speed`, `ball_speed` | `σ_pos · fps / SMOOTH_WIN`: la velocità è la differenza di due medie mobili su 5 frame, che differiscono per 4 campioni indipendenti |

**Esempio di stima.** Il residuo è il "tremolio" attorno alla traiettoria smussata:

```text
grezza     50.00  50.12  50.19  50.33  50.41
smussata   50.03  50.11  50.21  50.31  50.42
residuo    -0.03  +0.01  -0.02  +0.02  -0.01   → σ ≈ 0,02 m
```

**A cosa serve.** Alla forza: `z = margine / σ`, forza = `min(1, z / Z_SAT)`. Con `R_CONTROL` =
1,5 m e σ della distanza = 0,14 m: un giocatore a 1,45 m dà z = 0,4 (forza 0,12, il rumore potrebbe
ribaltare la condizione); a 1,20 m z = 2,1 (forza 0,71); a 1,00 m z = 3,6 (satura, forza 1).

**Il pavimento.** `SIGMA_POS_FLOOR` = 0,10 m è un valore minimo imposto a σ:
`sigma = max(sigma_stimato, SIGMA_POS_FLOOR)`. Serve a evitare che con un σ stimato quasi zero
ogni z diventi enorme e qualunque condizione soddisfatta di 1 cm risulti "certa".

**BAR-ATH: σ finisce sul pavimento** (0,10 m) sia per la palla sia per i giocatori: la stima è
più bassa e il pavimento prende il suo posto. σ resta fisso perché non lo stiamo misurando, lo
stiamo imponendo. Il motivo: il tracking che ci hanno dato è già **corretto e smussato** a monte
(il campo 105,3 × 68 di `games.py` è ricavato "dal tracking corretto"), quindi il tremolio rispetto
alla nostra media mobile è minuscolo.

Non è un bug, è un problema di concetto, con due facce:
1. **la forza non distingue**: con σ = 0,10 m bastano 40–50 cm di margine per saturare. È lo z
   mediano 32 di D-OUT: quasi tutte le attivazioni hanno forza 1, giuste o sbagliate;
2. **è il rumore sbagliato**: il tracker dà il baricentro del giocatore, la palla è ai piedi
   (30–80 cm a seconda della corsa), lo smoothing a monte può spostare tutto di qualche decimetro
   in modo sistematico, e questo non appare come tremolio. L'incertezza che conta per "controlla la
   palla?" è dell'ordine del metro.

**Proposta** (REGOLE §13), da decidere prima della forza di `control_episode` e P-DRIB:
`σ_i = max(σ sensore, tolleranza semantica a priori della condizione)`. Tolleranze, valori ed
effetto sugli z, decisioni da prendere: [STATO_LAVORI.md](STATO_LAVORI.md) §2.1 (call del 30/10).

---

## 5. Mappa feature → regole

| Feature | Regole ed eventi che la leggono |
|---|---|
| `nearest_*`, `nearest_team`, `nearest_gap` | `control_episode`, P-DRIB, P-FREE, P-PROG, R-TOUCH, R-LAST, T-REV, T-TACKLE, T-LOOSE |
| `n_within` | P-FREE, R-WALL, T-LOOSE |
| `rel_speed`, `align_cos` | `control_episode`, P-DRIB |
| `closing_speed` | `control_episode` (inizio del controllo), T-INTERCEPT |
| `chase_speed` | D-WHISTLE variante B |
| `pressure_dist` | P-FREE, T-TACKLE |
| `ball_acc` | `restart_kick`, `pass_event` |
| `ball_missing_for` | D-GAP, D-SHOT |
| `centroid_fwd`, `spread`, `n_opp_half` | P-SHAPE-OPEN |
| `n_opp_box`, `n_own_box` | R-SET, R-GOALKICK |
| `v_forward` | T-TEAMFLIP, P-SHAPE-OPEN |
| `slowdown`, `speed_med` | D-WHISTLE, D-GAP, D-EDGE |
| `sigma` | forza di tutte le regole (§4.1 di REGOLE) |

## 6. Cosa NON è una feature

| Sembra una feature | In realtà è | Dove va |
|---|---|---|
| "giocatore in controllo" | soglia su distanza + durata | `events.control_episode` |
| "palla ferma per una ripresa" | soglia su `ball_still_for` | `events.restart_kick` |
| "squadra in pressing" | soglia su `pressure_dist` / forma | una regola |
| "giocatori entro 1,5 m" come array | feature + soglia | `n_within(team, R_CONTROL)` nella regola |
| "possesso secondo lo Smart Tagging" | lettura di un'etichetta automatica | regole S-* |

## 7. Come aggiungere una feature

1. Una regola nella bozza ne ha bisogno? Se nessuna la legge, non si aggiunge.
2. È una misura (numero continuo o conteggio), non una decisione? (§6)
3. Ha una forma tra quelle del §3, NaN dove non misurabile, ed è vettorizzata?
4. Se dipende dalla direzione, usa `attack_dir`?
5. Ogni costante nuova è in `params.yaml` con origine e motivazione?
6. Ha il suo σ, o è chiaro da quale σ deriva?
7. Test su uno scenario sintetico in `tests/test_features.py` e controllo dei quantili su BAR-ATH
   con `scripts/inspect_features.py` (solo aggregati, nessun numero di frame).
8. Riga nella tabella del §5 e voce nel registro di `STATO_LAVORI.md`.

## 8. Verifica

```bash
python -m unittest discover -s tests -v       # 39 test, scenari sintetici
python scripts/inspect_features.py            # quantili su BAR-ATH, solo aggregati
```

Il controllo su BAR-ATH serve a vedere se i valori sono **fisicamente plausibili**, non a scegliere
soglie: nessuna soglia si fissa guardando questi numeri contro la GT (RULES.md §2).

## 9. Aperte

- σ sul pavimento: decidere la tolleranza semantica per condizione (§4.7, REGOLE §13).
- Slot di arbitri / doppioni: filtrarli prima di `nearest_*` e dei conteggi (task Gio).
- Velocità non definita sul ~5% dei frame: ricucire le tracce spezzate o tollerare i buchi nelle
  regole che leggono `slowdown`.
- `ACC_KICK`: ricavarlo dalla distribuzione di `ball_acc` (quantile alto, senza etichette).
