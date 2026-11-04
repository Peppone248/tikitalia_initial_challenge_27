# Regole di pseudo-labelling — bozza v0.5

Bozza delle regole del motore di pseudo-etichettatura (H / A / D) per MTAH27 Phase 1.
Owner: Beppe. Stato: **pseudocodice**, da tradurre nel motore in `src/tikitalia/rules/` (§12).
Implementate finora: D-OUT, D-GOAL, rilevamento del calcio d'inizio, feature per frame (§2.3).
Stato aggiornato: [STATO_LAVORI.md](STATO_LAVORI.md).

Contesto e motivazioni: [POSSIBLE_PATHS.md](POSSIBLE_PATHS.md) §3.
Termini (margine, rumore, forza, qualità, errore, …): [NOMENCLATURA.md](NOMENCLATURA.md).
Esempi numerici di valutazione di una regola: [ESEMPI_VALUTAZIONE_REGOLE.md](ESEMPI_VALUTAZIONE_REGOLE.md).
Feature per frame: [FEATURES.md](FEATURES.md). Motore: [MOTORE.md](MOTORE.md).

| Versione | Modifiche |
|---|---|
| v0.1 | prima bozza: primitive, parametri, regole D / P / R / S, motore, edge case |
| v0.2 | §4 modello di confidenza (forza per attivazione + qualità per regola); §6.2 regole di cambio possesso (inversione della palla, intercetto, contrasto, inversione collettiva); §7 riprese più robuste (rimessa, barriera, ultimo tocco, consenso); §8 nuove regole Smart Tagging e affidabilità del tag; D-GOAL con goal annullato |
| v0.3 | §4.1: forza calcolata per condizione (T è della condizione, la regola prende il minimo); qualità = precisione con prior Beta; definizione dell'errore per tipo di regola; calibrazione di `w = q · s`; tolleranza Δt per le TURNOVER. Nuovi `NOMENCLATURA.md` ed `ESEMPI_VALUTAZIONE_REGOLE.md` |
| v0.4 | Dopo i primi test su BAR-ATH (D-OUT, D-GOAL, calcio d'inizio) e le proposte del team del 27/10: D-GOAL riscritta partendo dal calcio d'inizio (la palla in rete il tracking la vede poco; senza Z un tiro sopra la traversa sembra un goal); nuova D-SHOT (tiro che esce, anche con palla persa); D-WHISTLE con palla calciata via; D-GAP solo con rallentamento; D-KO e tempi dei calci d'inizio; R-GOALKICK e R-THROW più precise; §14 mappa delle proposte del team sulle regole |
| v0.5 | §2.3 primitive implementate in `features.py`, con i nomi del codice (vecchio nome → nuovo); nuove `closing_speed` / `chase_speed` (distinte: la prima è ~0 in conduzione), `pressure_dist`, `v_forward`, rallentamento per squadra; §3 nuovi parametri `V_PLAYER_MAX`, `SIGMA_PLAYER_POS`, `TEMPO_MIN_PLAYERS`; §12 struttura del codice aggiornata; §13 σ sul pavimento anche per i giocatori. Nuovi `FEATURES.md`, `MOTORE.md`, `STATO_LAVORI.md` |

---

## 0. Idea in una pagina

Il motore valuta ogni frame e riconosce le **situazioni di gioco in cui il possesso è "semplice"
da determinare** (palla fuori, kick-off, conduzione, sequenza di passaggi, …). Lì emette un'etichetta
con una confidenza. Altrove **si astiene**.

Due livelli:

1. **Ancore** — regole molto precise che fissano `D`, `H` o `A`. Copertura bassa, va bene così.
2. **Inerzia** — tra un'ancora e l'altra il possesso resta a chi lo aveva. Cambia solo con
   un'evidenza di **controllo** da parte dell'altra squadra (CHALLENGE.md: un tocco, una deviazione
   o un duello aereo senza controllo successivo non cambiano il possesso).

Le regole deboli (forma delle squadre, fasi Smart Tagging) non decidono i cambi: votano, e servono
soprattutto a riempire i tratti senza palla.

---

## 1. Checklist di conformità (valida per ogni regola)

Ogni regola che entra nel motore deve rispettare tutti i punti. Va spuntata in code review.

- [ ] La condizione descrive una **situazione di gioco**, non un momento della partita: nessun
      numero di frame, nessun timestamp, nessun riferimento a un'azione specifica di BAR-ATH.
- [ ] Ogni soglia è un **parametro con nome** (§3) con origine dichiarata: `a priori` (significato
      calcistico/fisico) oppure `distribuzione` (quantile o mixture ricavati senza etichette).
- [ ] Nessuna soglia è stata scelta o ritoccata guardando la **GT pubblica**.
- [ ] Nessun **dato umano sulla partita** (eventi, risultato, formazioni, tracking corretto).
- [ ] Deterministica: stessi dati + stesso codice → stesso output (niente random senza seed fisso).
- [ ] Ha una **scheda** in questo file (condizione, output, confidenza, edge case, metrica).

Riferimenti: RULES.md §1 ("What remains fine"), §2, §3.
In caso di dubbio su una regola: issue agli organizzatori prima di costruirci sopra.

---

## 2. Convenzioni e primitive

### 2.1 Indici e unità

```text
frame         = video_frame (0 … 199318)            # ATD: frame_id - 1
t             = frame / 25                           # secondi
graded(f)     = f in half_video_blocks("BAR-ATH")    # pubblicati, consentiti
half(f)       = 1 | 2
pos           = metri: x ∈ [0, 105.3], y ∈ [0, 68.0] (da games.py)
vel, speed    = m/s, derivate dalle posizioni con smoothing (vedi §2.3)
team          = H | A ;  other(H) = A
```

> ⚠️ Unità della velocità ATD: XML dice `speed_kph`, DATA.md dice m/s. Finché non è chiarito
> (task Gio), **tutte le velocità si derivano dalle posizioni** e il campo `s` dell'ATD non si usa.

### 2.2 Geometria del campo

```text
L = 105.3 ; W = 68.0
GOAL_HALF_WIDTH  = 3.66            # porta 7.32 m
BOX_DEPTH        = 16.5 ; BOX_HALF_WIDTH = 20.16
CENTRE           = (L/2, W/2) ; CENTRE_RADIUS = 9.15

# direzione d'attacco, calcolata dai dati (non cablata):
attack_dir[team][half] ∈ {+1, -1}
    = segno di (L/2 - mediana(centroide_x[team]) nei primi K_DIR secondi del tempo)
    # al kick-off ogni squadra è nella propria metà → chi sta a sinistra attacca verso destra
    # controllo di coerenza: attack_dir[H][h] == -attack_dir[A][h]

opp_goal_x(team, half) = L  if attack_dir[team][half] == +1 else 0
own_goal_x(team, half) = L - opp_goal_x(team, half)

in_own_half(p, team, half)  = (p.x - L/2) * attack_dir[team][half] < 0
in_opp_half(p, team, half)  = not in_own_half(...)
in_opp_box(p, team, half)   = |p.x - opp_goal_x| <= BOX_DEPTH and |p.y - W/2| <= BOX_HALF_WIDTH
in_own_box(p, team, half)   = idem con own_goal_x
in_pitch(p, margin)         = -margin <= p.x <= L+margin and -margin <= p.y <= W+margin
near_line(p, d)             = min(p.x, L-p.x, p.y, W-p.y) <= d
near_corner(p, d)           = dist(p, corner più vicino) <= d
in_goal_mouth(p)            = (p.x < 0 or p.x > L) and |p.y - W/2| <= GOAL_HALF_WIDTH + GOAL_TOL
```

### 2.3 Primitive per frame (calcolate nel preprocessing, task Gio + Beppe)

**v0.5 — implementate** in `src/tikitalia/features.py` (descrizione completa, formule e valori su
BAR-ATH in [FEATURES.md](FEATURES.md)). Le regole le leggono da `context["features"]`. Il
pseudocodice qui sotto resta come specifica originale; i nomi nel codice sono:

| Pseudocodice | Codice | Note |
|---|---|---|
| `ball.pos`, `ball.vel`, `ball.speed`, `ball.still_for` | `match.ball_x/y`, `match.ball_vx/vy`, `match.ball_speed`, `match.ball_still_for` | in preprocessing |
| `ball.acc`, `ball.missing_for` | `f.ball_acc`, `f.ball_missing_for` | |
| `ball.last_seen_pos` | — | da aggiungere quando serve (D-SHOT) |
| `players[team]` (vel) | `f.pvx`, `f.pvy`, `f.pspeed` | per slot; velocità > `V_PLAYER_MAX` = scambio di traccia → NaN |
| `nearest[team]` | `f.nearest_dist`, `nearest_slot`, `nearest_x/y`, `nearest_vx/vy` | `(n, 2)` |
| `nearest_team` | `f.nearest_team` (+ `f.nearest_gap`: quanto è conteso) | |
| `n_within(team, r)` | `f.n_within(team, r)` | metodo: il raggio è della regola |
| `|ball.vel − nearest.vel|`, `cos(ball.vel, nearest.vel)` | `f.rel_speed`, `f.align_cos` | per `control_episode`, P-DRIB |
| — | `f.closing_speed`, `f.chase_speed` | nuove: distanza che si accorcia / corsa verso la palla (D-WHISTLE B) |
| — | `f.pressure_dist` | nuova: portatore → avversario più vicino (P-FREE, T-TACKLE) |
| `centroid[team]`, `spread[team]` | `f.centroid_fwd` (relativo ad `attack_dir`), `f.centroid_y`, `f.spread` | |
| `n_in_opp_half`, `n_in_opp_box`, `n_in_own_box` | `f.n_opp_half`, `f.n_opp_box`, `f.n_own_box` | relativi ad `attack_dir` |
| `v_team` (T-TEAMFLIP) | `f.v_forward` | |
| `speed_med_all`, `speed_ref`, `collective_slowdown` | `f.speed_med`, `f.speed_ref`, `f.slowdown` | `(n, 3)`: H, A, tutti — D-WHISTLE chiede entrambe le squadre |
| σ delle feature (§4.1) | `f.sigma` | posizione, distanza, velocità |

```text
ball.visible                  # bool
ball.pos, ball.vel, ball.speed, ball.acc      # NaN se non visibile; dopo pulizia dei salti
ball.still_for                # secondi consecutivi con ball.speed < V_BALL_STILL
ball.missing_for              # secondi dall'ultimo frame visibile
ball.last_seen_pos

players[team]                 # lista di (pos, vel) delle tracce valide di quella squadra
nearest[team]                 # (dist, pos, vel) del giocatore di team più vicino alla palla
nearest_team                  # team con nearest più piccolo
n_within(team, r)             # giocatori di team entro r m dalla palla
centroid[team], spread[team]  # baricentro, sd_x + sd_y
n_in_opp_half[team], n_in_opp_box[team], n_in_own_box[team]

speed_med_all                 # mediana delle velocità di tutti i giocatori nel frame
speed_ref                     # mediana mobile di speed_med_all sugli ultimi REF_WIN s
collective_slowdown           = speed_med_all / speed_ref          # < 1 = rallentamento

# contesto di gioco (calcolato prima di applicare le regole H/A)
context ∈ {OPEN_PLAY, DEAD}   # DEAD = una regola D è attiva sul frame
```

### 2.4 Eventi derivati (task Beppe)

```text
control_episode(team, start, end)
    # tratto continuo in cui un giocatore di team è "attaccato" alla palla:
    #   nearest[team].dist <= R_CONTROL  per >= T_CONTROL
    #   and nearest[other].dist > nearest[team].dist        (non conteso)
    #   and |ball.vel - nearest[team].vel| <= V_REL_CONTROL (palla che segue il giocatore)

pass_event(team, f_release, f_receive)
    # release:  fine di un control_episode di team + ball.acc > ACC_KICK
    # volo:     ball.speed > V_PASS, nessun giocatore entro R_CONTROL
    # receive:  inizio di un control_episode di team (stessa squadra) entro T_PASS_MAX
    # NB: senza ID, "giocatore diverso" = posizione di ricezione a > D_PASS_MIN dal punto di release

restart_kick(f)
    # fine di un tratto DEAD: ball.still_for >= T_STILL_RESTART poi ball.acc > ACC_KICK
```

---

## 3. Parametri

Valori di partenza. **Origine** = `a priori` (fisso, motivato) o `distribuzione` (ricavato dai dati
senza etichette). Nessun valore va regolato sulla GT pubblica.

| Nome | Valore iniziale | Origine | Motivazione |
|---|---|---|---|
| `K_DIR` | 5 s | a priori | al kick-off le squadre sono nelle proprie metà |
| `OUT_MARGIN` | 0,5 m | a priori | rumore di tracking vicino alle linee |
| `GOAL_TOL` | 0,5 m | a priori | idem |
| `V_BALL_STILL` | 0,5 m/s | a priori | palla ferma (3% dei frame con palla in `profile.txt`) |
| `V_JUMP` | 40 m/s | a priori | oltre = salto del tracker, non un tiro |
| `R_CONTROL` | 1,5 m | a priori + distribuzione | `profile.txt`: < 1 m nel 28,5%, < 2 m nel 51,5% dei frame con palla |
| `T_CONTROL` | 1,0 s | a priori | "a player keeps the ball for a few seconds" (CHALLENGE.md), versione prudente |
| `V_REL_CONTROL` | 2,0 m/s | a priori | palla e giocatore si muovono insieme |
| `ACC_KICK` | da definire | distribuzione | picco di accelerazione della palla (quantile alto) |
| `V_PASS` | 5 m/s | distribuzione | mediana della velocità palla in `profile.txt`: 5,1 m/s |
| `T_PASS_MAX` | 3 s | a priori | passaggio lungo ~40–50 m a velocità tipica |
| `D_PASS_MIN` | 5 m | a priori | distingue il passaggio dal controllo dello stesso giocatore |
| `N_PASS_ANCHOR` | 2 ricezioni (≥ 3 tocchi) | a priori | "more than two consecutive passes" (CHALLENGE.md) |
| `R_FREE_OPP` | 8 m | a priori | possesso non contestato: avversario più vicino lontano |
| `T_PROG` | 3 s | a priori | progressione con palla al piede |
| `REF_WIN` | 30 s | a priori | ritmo "normale" del tratto di gioco |
| `V_PLAYER_MAX` | 12 m/s | a priori | oltre = scambio di traccia o salto, non uno sprint (in partita 10–11 m/s) |
| `SIGMA_PLAYER_POS` | stimato | distribuzione | MAD del residuo posizione − media mobile, su tutte le tracce; su BAR-ATH finisce sul pavimento 0,10 m (§13) |
| `TEMPO_MIN_PLAYERS` | 6 per squadra | a priori | sotto, la velocità mediana non è una misura collettiva |
| `SLOWDOWN_RATIO` | 0,5 | a priori | dimezzamento collettivo della velocità = probabile fischio |
| `T_GAP_LONG` | 10 s | distribuzione | `profile.txt`: i buchi > 10 s sono l'85% dei frame senza palla |
| `T_STILL_RESTART` | 1 s | a priori | palla ferma prima di una ripresa |
| `N_PUSH` | da definire | distribuzione | giocatori in area avversaria a palla ferma (§6, R-SET) |
| `INERTIA_MIN` | 1 s | a priori | durata minima di un nuovo possesso prima di accettarlo |
| `ANGLE_REV` | 120° | a priori | inversione netta della direzione della palla |
| `T_REV_WIN` | 0,4 s | a priori | finestra prima/dopo per misurare la direzione (evita il rumore frame-to-frame) |
| `T_TURN_CONFIRM` | 1,5 s | a priori | tempo massimo tra evento di recupero e controllo che lo conferma |
| `T_TEAM_FLIP` | 3 s | a priori | finestra in cui entrambe le squadre invertono il movimento |
| `WALL_DIST` | 9,15 ± 1,5 m | a priori (regolamento) | distanza della barriera / degli avversari a palla ferma |
| `R_THROW_OUT` | 0,5 m | a priori | giocatore fuori dalla linea laterale = chi batte la rimessa |
| `Z_SAT` | 3 | a priori | z oltre il quale la forza di una condizione è satura: oltre 3σ la probabilità di un falso dovuto al rumore è ~0,13%; da verificare con la calibrazione (§4.1) |
| `N0_PRIOR` | 30 episodi | a priori | peso del prior di qualità, in osservazioni fittizie |
| `DT_EVAL` | ±0,5 s | a priori | tolleranza per giudicare corretto un cambio proposto da una TURNOVER (solo valutazione, non entra nel motore) |
| `K_BOUNDARY` | da definire | a priori | frame intorno a un cambio vero esclusi dal calcolo dell'errore (GT ambigua) |

---

## 4. Formato di una regola

```text
RULE <ID>                      # D-*, P-*, R-*, S-*
  kind:        ANCHOR | SOFT | RESTART
  scope:       OPEN_PLAY | DEAD | ANY
  inputs:      ATD | ST | ATD+ST
  when:        <condizione sul frame o sulla finestra>
  emit:        label ∈ {H, A, D} | restart_team ∈ {H, A}
  conf:        0..1  (prior di qualità `conf_prior`, poi stimata come in §4.1)
  span:        quali frame etichetta (il frame, la finestra, retroattivo da … a …)
  edge cases:  …
  metric:      copertura, conflitti, accordo con ST, precisione su altre partite (M1) se solo-ATD
```

Tipi:
- **ANCHOR**: fissa l'etichetta. Il motore temporale non la può ribaltare se `conf` ≥ soglia.
- **SOFT**: voto pesato, usato per riempire e per rompere la parità.
- **RESTART**: non etichetta il tratto D, dice **chi riprende** → etichetta del possesso che parte
  al `restart_kick`.
- **TURNOVER**: non etichetta un tratto, propone **un istante di cambio possesso** verso una
  squadra; il motore lo accetta solo se confermato da un controllo (§6.2).

### 4.1 Confidenza: due numeri distinti

Il `conf` fisso della v0.1 mescola due cose diverse. Si separano:

| | Cosa misura | Varia per | Da dove viene |
|---|---|---|---|
| **forza** `s ∈ [0,1]` | quanto *questa attivazione* è netta | frame | margine delle feature rispetto alla soglia, come z-score |
| **qualità** `q ∈ [0,1]` | quanto *la regola* è affidabile in generale | regola | prior → stima senza etichette (label model) → precisione su altre partite (M1) |

Peso del voto nel motore: `w = q · s`.

**Forza come z-score.** Una regola ha una o più condizioni in AND. Ogni condizione
`feature ≷ soglia` diventa un margine standardizzato, e la regola prende la condizione più debole:

```text
m_i  = margine: di quanto la feature sta dentro la soglia (m > 0 = condizione soddisfatta)
       x ≤ θ → m = θ − x ;  x ≥ θ → m = x − θ
σ_i  = rumore tipico della feature, stimato SENZA etichette:
       - posizioni: deviazione standard del jitter (residuo tra traccia grezza e traccia smussata)
       - distanza tra due oggetti: σ = √(σ_a² + σ_b²)
       - velocità/angoli: idem, sulla grandezza derivata
       - conteggi (n giocatori, n passaggi): σ = 1
z_i  = m_i / σ_i
d_i  = min(1, durata_osservata / (2·T_i))   # solo condizioni "per ≥ T_i"; T è della condizione
       1                                     # condizioni istantanee
s_i  = clip(z_i / Z_SAT, 0, 1) · d_i         # forza della condizione, satura a z ≥ Z_SAT
s    = min_i s_i                             # condizioni in AND: conta la più debole
```

Esempio completo (P-DRIB, s = 0,37 → 0,93 mezzo secondo dopo): ESEMPI_VALUTAZIONE_REGOLE.md §1.

**Qualità = precisione.** `q` risponde a "quando la regola scatta, quanto spesso ha ragione?".
La copertura (quanto spesso scatta) **non** entra in `q`: una regola frequente pesa già di più
perché vota più spesso.

```text
errore:  r scatta su f e ℓ_r(f) ≠ y(f)        # ℓ_r = etichetta emessa, y = etichetta di riferimento
         ANCHOR / SOFT → per frame e per episodio (riportati entrambi)
         RESTART       → una volta per ripresa (squadra che riprende)
         TURNOVER      → per evento: corretto se entro ±DT_EVAL da un cambio vero non ancora
                         abbinato → precisione e richiamo
         frame entro ±K_BOUNDARY da un cambio vero: esclusi (GT ambigua)
q_r   =  (conf_prior · N0_PRIOR + corretti_r) / (N0_PRIOR + attivazioni_r)     # prior Beta
```

Con poche attivazioni `q` resta vicino al prior, con molte è dominata dai dati: la frequenza
decide quanto ci fidiamo della stima, non la stima stessa.

**Da dove vengono `corretti` e `y`, in tre stadi:**

1. **Prior** dichiarato nella scheda (il `conf` di v0.1), `corretti = attivazioni = 0`.
2. **Stima senza etichette su BAR-ATH** (consentita: nessuna persona decide etichette):
   - *consenso leave-one-out*: `y` = uscita del motore **senza** `r`, solo sui frame con
     `conf₋ᵣ(f)` alta. Non vede gli errori comuni a tutte le regole;
   - *accordo pesato*: frazione dei frame coperti in cui la regola concorda con le ANCHOR delle
     **altre** famiglie (ATD vs ST) — una regola che contraddice spesso le altre è sospetta;
   - *label model* (Dawid–Skene / Snorkel): stima l'accuratezza di ogni regola dalla matrice di
     accordi/disaccordi, assumendo che gli errori di regole diverse siano poco correlati.
     Le regole che usano le stesse feature (es. P-DRIB e P-FREE) vanno dichiarate correlate.
3. **Precisione misurata su altre partite (M1)** per le regole solo-ATD: è l'unico numero
   "vero"; quando c'è sostituisce la stima.

**Report per regola** (`report.py`), da guardare a ogni modifica:

```text
rule_id | n_attivazioni | copertura % | s medio | conflitti % (con ANCHOR) | accordo ST % | q_label_model | q_M1
```

> ⚠️ La confidenza non va calibrata sui 5 minuti pubblici: sarebbe tuning sulla GT (RULES.md §2).

---

## 5. Regole D (dead ball)

### D-OUT · palla fuori dal campo — ANCHOR

```text
when:  ball.visible and not in_pitch(ball.pos, OUT_MARGIN)
emit:  D
span:  dal primo frame fuori fino al restart_kick successivo (vedi D-HOLD)
conf:  0.95
edge:  palla oltre la linea ma in rete → D-GOAL ha precedenza
       palla che "esce" per un salto del tracker → scartata in preprocessing (V_JUMP)
stato: primo test su BAR-ATH: 70 episodi sui frame valutati; 26/26 rimesse ST precedute da un
       D-OUT, corner 8/12, rinvii 6/10. La forza basata sul margine geometrico NON discrimina
       (fascia più bassa coerente con lo ST 10/11): σ della palla è piccolo, z satura (mediana 32),
       e le uscite brevi sono quasi sempre vere. Da rifare sulla plausibilità dell'evento:
       continuità della traccia prima dell'uscita, palla che resta fuori o sparisce (non rientra
       entro ~0,2 s), rallentamento e ripresa dopo. Il controllo di coerenza con lo ST va
       confrontato con istanti casuali (i SET PIECES coprono il 27,6% dei frame).
```

### D-KO · calcio d'inizio — ANCHOR

```text
when:  dist(ball.pos, CENTRE) <= 1 m
       and ball.still_for >= T_STILL_RESTART
       and all(in_own_half(p, team, half) for team, p in players)      # tolleranza: ≤ 1 per squadra
emit:  D
span:  finché vale la condizione → fine al restart_kick
plus:  restart_team = squadra con il giocatore più vicino alla palla al restart_kick  (R-KO)
conf:  0.95  (0.99 se c'è uno ST `Kick Off (Start)` o `Kick Off (After Goal)` sovrapposto)
note:  frame non valutati = X. MA (FAQ.md): il 2° tempo è valutato da prima del kick-off
       → quel tratto è valutato ed è tutto D. Regola generale, senza
       costanti: dall'inizio di ogni blocco valutato (half_video_blocks) fino al restart_kick
       del primo D-KO del tempo → D.
tempi: il calcio d'inizio NON coincide in generale con il primo frame del tempo.
       1° tempo: valutato dal calcio d'inizio stesso (FAQ.md) → la preparazione sta nei frame
       NON valutati prima → il rilevatore deve cercare anche fuori dai blocchi valutati.
       2° tempo: preparazione e calcio d'inizio dentro il blocco valutato, ~2 minuti dopo il suo inizio.
       In nessun caso si usa il numero di frame pubblicato come costante: serve solo per il debug
       del rilevatore (scripts/diagnose_kickoff.py).
stato: primo test su BAR-ATH → 0 calci d'inizio rilevati. diagnose_kickoff.py ha mostrato che
       la causa è la palla, non la soglia: nei 10 s prima dei calci d'inizio la palla è vista
       al centro 0 volte (nascosta da chi batte) e a tratti il tracker segue un oggetto a 30–50 m
       dal gioco. I giocatori invece rispettano lo schieramento nel 100% dei frame.
v0.4 (implementata in events.py): rilevamento SOLO dai giocatori, in due tempi come la Law 8
       1. schieramento: squadre nelle proprie metà (i giocatori nel cerchio non contano),
          al massimo KO_MAX_INTRUDERS = 1 giocatore della squadra che NON batte dentro il cerchio
          (arbitro o doppione assegnato a una squadra dal tracker), per ≥ T_KO_SETUP = 2 s
       2. battitore: un giocatore arriva sul dischetto (entro KO_KICKER_R = 3 m), spesso solo
          nell'ultimo secondo → si cerca su tutto il tratto, non frame per frame
       Senza battitore, o con più di un intruso (mediana sul tratto), lo schieramento è scartato.
       La palla al centro è solo un indizio in più, mai una condizione.
       Secondo test: 1° tempo trovato (+0,68 s), direzione d'attacco coerente e confermata dal
       metodo del portiere; 2° tempo non trovato (intrusi nel cerchio, battitore tardivo) →
       motivo della v0.4.
       Terzo test: 5 calci d'inizio accettati, tutti con un tag di calcio d'inizio dello ST
       (5/6 tag agganciati). Il 2° tempo era trovato ma scartato (2 avversari sul bordo del
       cerchio) e il calcio d'inizio dopo un goal al 35' veniva promosso a "primo del tempo",
       nascondendo quel goal a D-GOAL. Correzioni (nessuna soglia ritoccata):
       - primo del tempo solo entro KO_HALF_START_WIN = 5 min dall'inizio del blocco valutato
       - schieramento scartato SOLO per gli intrusi → accettato se un calcio d'inizio dello ST
         lo conferma (rescued_by_st); schieramento e battitore restano del tracking
       - battitore = unica squadra con un giocatore sul dischetto; gli intrusi si controllano
         una volta sola sull'intero schieramento
       - ball_after_m: dove si rivede la palla entro 5 s dalla fine dello schieramento. Lontano
         dal centro = schieramento finito senza calcio d'inizio (es. goal annullato dopo che le
         squadre si erano già schierate): da guardare, non ancora usato come regola
```

### D-GOAL · dopo un goal — ANCHOR

**v0.4 — si parte dal calcio d'inizio, non dalla palla in rete.** Primo test su BAR-ATH: solo 2
candidati "palla in rete" e nessuno vicino ai 4 `GOALS` dello Smart Tagging. La palla dentro la
porta il tracking la vede poco (rete, portiere, mischia). Inoltre **l'ATD non ha la Z**: un tiro
sopra la traversa, visto dall'alto, attraversa la linea tra i pali esattamente come un goal.
La palla tra i pali quindi non basta mai da sola; il calcio d'inizio a centrocampo sì.

```text
ancora: D-KO che NON è il primo del tempo  →  c'è stato un goal prima (Laws: il calcio d'inizio
        a gioco iniziato segue solo un goal)
ricerca all'indietro, nella finestra [ko.start − T_GOAL_KO, ko.start]:
        goal_end = porta della squadra che NON batte il calcio d'inizio (chi ha subìto riparte)
                   (controllo: deve essere la porta difesa dalla squadra che batte, via attack_dir)
        f_goal   = l'ultimo istante in cui, verso goal_end:
                     (a) palla vista oltre la linea tra i pali            (in_goal)
                     (b) palla persa in volo con traiettoria che taglia la linea tra i pali
                         prima di essere ritrovata (estrapolazione, vedi D-SHOT)   (lost_shot)
                     (c) palla vista per l'ultima volta in area, prima di un buco lungo   (lost_box)
                   in ordine di preferenza a > b > c
        se nessun candidato: f_goal = inizio del rallentamento collettivo (D-WHISTLE) che
                   precede il calcio d'inizio; source = "slowdown_only"
emit:   D da f_goal al restart_kick del calcio d'inizio
plus:   scorer_side = other(squadra che batte il calcio d'inizio)  → P-GOAL
conferme (alzano la forza, nessuna crea un goal da sola):
        ST `GOALS` sovrapposto alla finestra; ST `Kick Off (After Goal)` sovrapposto al calcio d'inizio;
        scorer_side coerente con la porta attaccata (attack_dir)
forza:  per condizione (§4.1): z della geometria del candidato · fattore per source
        (in_goal 1 · lost_shot 0.8 · lost_box 0.6 · slowdown_only 0.4) · conferme
edge:   autogol → scorer_side dalla porta, non dal Team del tag
        candidati tra i pali SENZA calcio d'inizio dopo → non sono goal: tiro sopra la traversa
        o goal annullato → D-SHOT (D fino alla ripresa), mai P-GOAL automatico
        ST `GOALS` senza calcio d'inizio rilevato → nessun D da questa regola; finisce nel report
```

**Goal annullato — D-GOAL-VOID.** Palla in porta ma **nessun kick-off** a centrocampo nei secondi
successivi. Non è un caso da trattare a parte con una regola dedicata a "quel" goal: la regola
generale deve già comportarsi bene.

```text
when:  palla in porta (come D-GOAL) and NOT D-KO entro 120 s
emit:  D da f_goal fino al restart_kick effettivo (punizione, rinvio, …)
       # spesso il D è lungo (controllo VAR): D-GAP / D-WHISTLE lo coprono
plus:  P-GOAL vale comunque: l'azione prima del goal era della squadra che ha segnato
       restart_team: dalle regole R-* generali (di solito chi difendeva: fuorigioco o fallo in
       attacco → punizione per la difesa), NON assegnato per default
conf:  D 0.85 ; restart_team dalla regola R-* che lo determina
note:  ST `GOALS` + nessun kick-off = indizio di goal annullato o di falso positivo del tag;
       in entrambi i casi il D si chiude con la ripresa reale, quindi il risultato non cambia
```

> ⚠️ Il risultato reale della partita (o sapere quanti goal sono stati annullati) è un dato
> prodotto da persone sulla partita: non entra nel codice e non si usa per controllare l'output
> (RULES.md §2). La regola deve reggere da sola.

### D-SHOT · tiro (o lancio) che esce dal fondo — ANCHOR se vista uscire, SOFT se estrapolata

Risposta alla domanda del team: **sì, un tiro che esce è palla morta** (Laws 9: la palla è fuori
gioco quando ha superato interamente la linea di porta o laterale) fino al rinvio dal fondo o al
corner. Se la palla si vede fuori, D-OUT basta già. Questa regola serve per il caso frequente in
cui il tracking la perde durante il tiro: nel test, corner e rinvii dello Smart Tagging con un
D-OUT prima erano solo 8/12 e 6/10.

```text
tiro:   palla che si stacca da un giocatore (ball.acc > ACC_KICK) con ball.speed ≥ V_SHOT
        e direzione verso una linea di fondo: |angolo tra velocità e asse x| ≤ ANGLE_SHOT
        ("verticale" rispetto alle porte; un lancio verso la linea laterale è "orizzontale" e
        porta a una rimessa: stessa regola, linea laterale)
uscita: (a) palla vista oltre la linea entro T_SHOT_OUT            → D-OUT, ANCHOR
        (b) palla persa: estrapolazione lineare dall'ultimo punto visto con la sua velocità;
            se taglia la linea prima di T_SHOT_OUT e prima di qualunque giocatore
            (nessun giocatore entro R_CONTROL lungo il tratto estrapolato)  → SOFT
conferma (b): dopo il buco la ripresa è coerente con la linea tagliata:
            palla ferma nell'area piccola (rinvio) o alla bandierina (corner) o alla linea
            laterale (rimessa); oppure ST `Goal Kick` / `Corner Kick` / `Throw-in`
            → con conferma diventa ANCHOR
emit:   D dall'istante stimato di uscita al restart_kick
plus:   linea di fondo: restart_team da R-GOALKICK / R-CORNER (chi ha toccato per ultimo: R-LAST)
forza:  z dell'angolo e della velocità del tiro; distanza estrapolata oltre la linea / σ
        della traiettoria (cresce con il tempo di estrapolazione: oltre ~1 s l'incertezza è alta);
        fattore di conferma della ripresa
edge:   tiro sopra la traversa senza Z: identico a un goal in 2D → il calcio d'inizio decide
        (D-GOAL); senza calcio d'inizio è D-SHOT
        tiro parato e trattenuto: il portiere è sul tratto estrapolato → la regola non scatta
        palla respinta in campo: ritrovata dentro prima di T_SHOT_OUT → nessun D
param:  V_SHOT, ANGLE_SHOT, T_SHOT_OUT (a priori, da aggiungere a params.yaml)
```

### D-SP · set piece dello Smart Tagging confermato — ANCHOR

```text
when:  ST `SET PIECES` attivo sul frame
       and ball.still_for >= T_STILL_RESTART               # oppure palla non visibile e collective_slowdown
emit:  D
span:  dal primo frame con la condizione al restart_kick
       # NON tutto l'intervallo ST: dura in mediana 35 s e contiene anche il gioco dopo la ripresa
conf:  0.9
```

### D-WHISTLE · fischio (falli, fuorigioco, infortuni) — SOFT → ANCHOR se confermato

```text
when:  collective_slowdown <= SLOWDOWN_RATIO  per >= 1 s
       and (ball.still_for >= T_STILL_RESTART or not ball.visible)
emit:  D
span:  dall'inizio del rallentamento al restart_kick
conf:  0.6 ; 0.85 se seguito da una ripresa riconosciuta (restart_kick con palla ferma prima)
edge:  giro palla lento: giocatori lenti ma palla in movimento → la condizione sulla palla lo esclude
       portiere con palla in mano: palla ferma + rallentamento → falso D (vedi §9, domanda Q3)
note:  lo ST ha solo 5 `Free Kick`: quasi tutti i falli si trovano solo da qui
```

v0.4 — proposte del team:

```text
variante A (palla ferma):   collective_slowdown di ENTRAMBE le squadre
                            and palla ferma (o tenuta in mano) per T_WHISTLE (3–5 s, a priori)
variante B (palla calciata via): collective_slowdown di entrambe le squadre
                            and la palla si allontana dall'ultimo possessore SENZA che nessuno
                            la insegua (nessun giocatore che accelera verso di lei)
                            # dopo un fischio la palla spesso si muove: viene calciata via o
                            # passata all'arbitro → "palla in movimento" non esclude il D
span:   retroattivo: il D comincia all'inizio del rallentamento, non dopo i 3–5 s di attesa
        (si lavora offline sull'intera partita)
edge:   gestione lenta dei difensori (giro palla a centrocampo): palla in movimento tra
        compagni → è possesso, non D. Distinzione: in B la palla NON arriva a un compagno che la
        controlla (control_episode) e il rallentamento riguarda entrambe le squadre
```

### D-GAP · interruzione lunga senza palla — SOFT

```text
when:  ball.missing_for >= T_GAP_LONG
       and collective_slowdown <= SLOWDOWN_RATIO (mediana sul tratto)
emit:  D
conf:  0.5
note:  sostituzioni, infortuni, VAR, palla fuori inquadratura; da validare con la distribuzione
v0.4:  il buco da solo NON basta: la palla manca nel 41% dei frame valutati, spesso in pieno
       gioco (palle alte, mischie). "Palla assente = D" è proprio l'errore della baseline sul D.
       Il rallentamento collettivo di entrambe le squadre è obbligatorio.
```

### D-EDGE · palla ferma vicino alla linea con giocatori fermi — SOFT

```text
when:  near_line(ball.pos, 1.5 m) and ball.still_for >= T_STILL_RESTART
       and collective_slowdown <= SLOWDOWN_RATIO
emit:  D
conf:  0.6
note:  la sola vicinanza alla linea non basta (6,4% dei frame, spesso gioco normale)
```

### D-HOLD · persistenza del D — meccanismo, non regola

```text
una volta aperto un tratto D da una regola ANCHOR, resta D fino al primo restart_kick
o fino a un control_episode (P-DRIB) in OPEN_PLAY con palla in movimento.
```

---

## 6. Regole H/A (possesso)

Valutate solo con `context == OPEN_PLAY`, salvo le RESTART.

### P-DRIB · conduzione — ANCHOR

```text
when:  control_episode(team, s, e) con (e - s) >= T_CONTROL
       and cos(ball.vel, nearest[team].vel) >= 0.8 quando ball.speed > 1 m/s
emit:  team
span:  [s, e]
conf:  0.85
edge:  contrasto: nearest[other].dist <= R_CONTROL per parte dell'episodio → conf 0.5
```

### P-PASS · sequenza di passaggi — ANCHOR

```text
when:  catena pass_event(team, …) consecutivi con ≥ N_PASS_ANCHOR ricezioni
       e nessun control_episode di other in mezzo
emit:  team
span:  dal primo control_episode della catena all'ultimo (retroattivo: si lavora offline)
conf:  0.9
edge:  palla persa in volo: se il buco è ≤ T_PASS_MAX e la ricezione è di team, conta lo stesso
```

### P-FREE · possesso non contestato — ANCHOR

```text
when:  nearest[team].dist <= R_CONTROL
       and nearest[other].dist >= R_FREE_OPP
       and ball.speed <= V_PASS
       per >= T_CONTROL
emit:  team
conf:  0.85
note:  copre l'impostazione dal basso; il compagno vicino esclude la palla vagante dopo un rinvio
```

### P-PROG · progressione con palla — SOFT

```text
when:  per >= T_PROG:
         ball avanza verso opp_goal_x(team) (componente x della velocità * attack_dir > 0)
         and nearest_team == team per >= 80% dei frame
         and nearest[team].dist <= 2 * R_CONTROL
emit:  team
conf:  0.6
edge:  lancio lungo: la palla avanza ma nessuno è vicino → escluso dalla condizione su nearest
```

### P-GOAL · azione che porta al goal — ANCHOR (deriva da D-GOAL)

```text
when:  D-GOAL con scorer_side definito
emit:  scorer_side
span:  dall'ultimo cambio di possesso rilevato prima di f_goal fino a f_goal
       # mai una finestra fissa all'indietro
conf:  0.8
```

### P-SHAPE-OPEN · forma delle squadre a palla in movimento — SOFT (peso basso)

```text
scope: OPEN_PLAY
when:  sempre (è un voto continuo)
score(team) = w1 * (n_in_opp_half[team] - n_in_opp_half[other])
            + w2 * (spread[team] - spread[other])
            + w3 * (velocità media di team verso opp_goal)
emit:  argmax(score) con conf proporzionale a |score diff|, max 0.4
note:  la forma si adatta con secondi di ritardo dopo un cambio di possesso:
       è debole proprio sulle transizioni → serve a riempire i buchi della palla, non a decidere
       un cambio. Il motore temporale non accetta un cambio sostenuto solo da regole SOFT.
```

### 6.2 Regole TURNOVER (cambio di possesso)

Sono le regole che contano di più per spellF1: decidono **dove** finisce un possesso. Principio:
un cambio richiede **due** evidenze indipendenti — un *segnale di recupero* (questo paragrafo) e
un *controllo* della nuova squadra (P-DRIB, P-FREE, P-PASS) entro `T_TURN_CONFIRM`. Il segnale
dice **quando**, il controllo dice **se**. Senza conferma il segnale viene scartato (inerzia).

Le direzioni sono sempre **relative alla squadra** (`attack_dir`), così valgono in entrambi i tempi.

```text
dir_ball(f)          = versore della velocità media della palla in [f − T_REV_WIN, f]
dir_ball_after(f)    = idem in [f, f + T_REV_WIN]
toward(team, v)      = v.x * attack_dir[team][half]            # > 0 = verso la porta attaccata da team
```

#### T-REV · inversione repentina della direzione della palla — TURNOVER

La tua regola. H attacca verso destra, passaggio intercettato: la palla torna verso sinistra.

```text
when:  angle(dir_ball(f), dir_ball_after(f)) >= ANGLE_REV
       and toward(team_prev, dir_ball(f))  > 0         # prima andava verso la porta di chi aveva palla
       and toward(team_prev, dir_ball_after(f)) < 0    # dopo va verso la sua porta
       and nearest_team(f) == other(team_prev)         # all'inversione c'è un avversario sulla palla
       and nearest[other].dist(f) <= R_CONTROL
emit:  turnover(f, new_team = other(team_prev))
confirm: control_episode(new_team) che inizia entro T_TURN_CONFIRM
forza: z da angolo, distanza dell'avversario, durata del controllo successivo
```

Perché da sola **non basta** (falsi positivi da escludere):

| Situazione | Inversione? | Come la si esclude |
|---|---|---|
| Retropassaggio / cambio gioco di team_prev | sì | all'inversione la palla è vicina a un compagno, non a un avversario |
| Rinvio/respinta della difesa senza controllo | sì | manca il `control_episode` di conferma → inerzia |
| Palo, traversa, parata respinta | sì | nessun giocatore vicino all'inversione; se poi la difesa controlla → T-LOOSE |
| Tiro parato e trattenuto dal portiere | palla si ferma | P-FREE/portiere: vedi Q3 |
| Intercetto che prosegue nella stessa direzione (es. anticipo e fuga in avanti) | **no** | T-INTERCEPT, sotto |

#### T-INTERCEPT · passaggio interrotto da un avversario — TURNOVER

Copre gli intercetti senza inversione (la direzione non è l'unico segnale).

```text
when:  pass_event(team_prev) iniziato (release) ma la ricezione avviene vicino a other:
         il primo control_episode dopo il release è di other(team_prev), entro T_PASS_MAX
emit:  turnover(f_inizio_controllo, other(team_prev))
forza: z da distanza di controllo e durata
```

#### T-TACKLE · contrasto vinto — TURNOVER

```text
when:  control_episode(team_prev) → tratto conteso (entrambi entro R_CONTROL)
       → control_episode(other) che dura ≥ T_CONTROL
emit:  turnover(inizio del controllo di other, other)
note:  un contrasto senza controllo successivo (palla che schizza via) non cambia nulla
```

#### T-LOOSE · palla vagante recuperata — TURNOVER

```text
when:  tratto con nessun giocatore entro R_CONTROL per ≥ 0.5 s (palla libera, anche dopo una respinta)
       → primo control_episode di other(team_prev)
emit:  turnover(inizio controllo, other)
note:  se il primo controllo è di team_prev → nessun cambio (corner respinto che torna a chi batte)
```

#### T-TEAMFLIP · inversione collettiva del movimento delle squadre — TURNOVER (SOFT)

Utile soprattutto quando la palla **non è visibile**: quando il possesso cambia, entrambe le
squadre cambiano verso di corsa (chi attaccava rientra, chi difendeva riparte).

```text
v_team(team, f)  = media di toward(team, vel) dei giocatori di team
when:  in una finestra di T_TEAM_FLIP:
         v_team(team_prev) passa da > +v0 a < −v0       # chi aveva palla torna indietro
         and v_team(other)  passa da < −v0 a > +v0      # chi difendeva va avanti
emit:  turnover(punto medio della finestra, other)
forza: z dall'ampiezza delle due variazioni
note:  la forma si adatta in ritardo → istante poco preciso. Serve a dire SE c'è stato un cambio
       durante un buco della palla; l'istante va preso, se c'è, da un segnale palla
       (T-REV/T-INTERCEPT) o dal primo controllo dopo il buco.
```

#### Combinazione nel motore

```text
per ogni candidato turnover(f, new_team):
    forza_tot = combinazione delle forze dei segnali entro ±T_REV_WIN (più segnali = più forte)
    accettato se:
        conferma di controllo di new_team entro T_TURN_CONFIRM
        or (palla non visibile and T-TEAMFLIP con forza alta and ST di transizione coerente, §8)
    istante del cambio = il più precoce tra i segnali palla accettati (non l'inizio del controllo)
```

---

## 7. Regole RESTART (chi riprende dopo un D)

Etichettano il possesso che **parte** al `restart_kick`. Il tratto D resta D.

### R-SET · forma delle squadre a palla ferma — RESTART

Distinzione rispetto a P-SHAPE-OPEN: a palla ferma la disposizione **anticipa** chi riprende
(gli attaccanti salgono in area per corner e punizioni), invece di inseguire il possesso.

```text
scope: DEAD, con palla ferma nella metà offensiva di una delle due squadre
when:  n_in_opp_box[team] >= N_PUSH and n_in_opp_box[team] > n_in_opp_box[other]
       # contare per DIREZIONE: i difensori stanno nella PROPRIA area, quindi non vengono contati
emit:  restart_team = team
conf:  0.75
edge:  punizione a centrocampo / rimessa: nessuno sale → la regola si astiene
```

### R-CORNER · calcio d'angolo — RESTART

```text
when:  near_corner(ball.pos, 2 m) and ball.still_for >= T_STILL_RESTART
       (conf più alta se ST `Corner Kick` attivo)
emit:  restart_team = squadra che attacca quella porta  (attack_dir)
conf:  0.9
note:  la preparazione del corner (giocatore che va alla bandierina, sistemazione) è D
       → vedi domanda Q2 in §9
```

### R-GOALKICK · rinvio dal fondo — RESTART

```text
when:  palla ferma nell'area piccola di una porta, dopo D-OUT o D-SHOT dal fondo
       (conf più alta se ST `Goal Kick`)
plus (v0.4, proposta del team): disposizione da rinvio
       - il giocatore più vicino alla palla è il più arretrato della squadra che difende
         quella porta (il portiere)
       - difensori di quella squadra vicino all'area
       - avversari FUORI dall'area di rigore (Laws 16: obbligatorio fino al calcio)
emit:  restart_team = squadra che difende quella porta
conf:  0.85 ; 0.95 con la disposizione da rinvio
```

### R-TOUCH · primo tocco dopo la ripresa — RESTART (generale, copre rimesse e punizioni)

```text
when:  restart_kick a f_r
emit:  restart_team = squadra di nearest_team nella finestra [f_r - 0.5 s, f_r]
conf:  0.8
note:  risolve il `Team = N/A` di tutte le rimesse dello ST
```

### R-KO · kick-off — RESTART

```text
emit:  restart_team = squadra con il giocatore più vicino alla palla al restart_kick di D-KO
       # dopo un goal è la squadra che ha subìto: si può usare come controllo di P-GOAL
conf:  0.95
```

### R-THROW · rimessa laterale — RESTART

Il regolamento obbliga chi batte la rimessa a stare **fuori dal campo**: è un segnale geometrico
forte, e risolve il `Team = N/A` dello ST.

```text
when:  palla ferma o tenuta vicino alla linea laterale (near_line laterale ≤ 1.5 m) in DEAD
       and esiste un giocatore con |y − linea| oltre R_THROW_OUT FUORI dal campo, vicino alla palla
emit:  restart_team = squadra di quel giocatore
forza: z dalla distanza oltre la linea e dalla vicinanza alla palla
note:  il tracker può perdere chi sta fuori campo → se manca, fallback a R-LAST e R-TOUCH
```

Nota v0.4 (domanda del team: "la rimessa è un cambio di possesso?"). Per la metrica non conta:
la sequenza è sempre `possesso → D → possesso` e CHALLENGE.md dice che `H H D D H H` sono **due**
possessi. Basta etichettare bene il D e la squadra che riprende. Di solito riprende l'avversario
di chi ha toccato per ultimo (R-LAST), ma non va dato per scontato: R-THROW (chi sta fuori dalla
linea) lo verifica.

### R-WALL · barriera / distanza regolamentare — RESTART

Su punizioni e corner gli avversari devono stare a 9,15 m: un **anello di giocatori a ~9,15 m**
dalla palla ferma indica la squadra che **difende**, quindi chi riprende è l'altra.

```text
when:  DEAD and ball.still_for >= T_STILL_RESTART
       and n_within_ring(team, WALL_DIST) >= 2 and n_within(team, 7 m) == 0
       and n_within(other, 3 m) >= 1                       # qualcuno dell'altra squadra sulla palla
emit:  restart_team = other
note:  sulle rimesse la distanza regolamentare è 2 m → la regola non scatta (corretto)
```

### R-LAST · chi ha toccato per ultimo prima dell'uscita — RESTART

```text
when:  D-OUT a f_out
emit:  restart_team = other(squadra di nearest_team nell'ultimo tratto con palla vicina a un giocatore prima di f_out)
       eccezione: uscita dalla linea di fondo
         ultimo tocco della squadra che attacca quella porta → R-GOALKICK (riprende chi difende)
         ultimo tocco della squadra che difende            → R-CORNER  (riprende chi attacca)
forza: bassa se l'ultimo giocatore vicino era lontano (> R_CONTROL) o se erano vicini due avversari
note:  "ultimo tocco" con il tracking è approssimato → regola di supporto, non decisiva
```

### Consenso sulle riprese

Ogni ripresa ha più regole indipendenti. Il motore le combina invece di sceglierne una:

```text
per ogni restart_kick f_r:
    candidati = [(rule, team, q·s) for rule in R-* if rule.fires(f_r)]
    restart_team = argmax_team Σ q·s
    agreement    = quota di peso sul vincitore
    if agreement < 0.7: marcare la ripresa come "incerta" nel report
    # le riprese incerte sono il primo posto da guardare nel debug, mai da correggere a mano
```

Coppie di regole che si controllano a vicenda: R-THROW ↔ R-LAST (rimesse), R-CORNER ↔ R-WALL ↔
R-SET (corner), R-KO ↔ P-GOAL (dopo un goal riprende chi ha subìto), R-TOUCH con tutte.

---

## 8. Regole Smart Tagging

Prerequisito (task Francesco): intervalli ST in frame, `Team` normalizzato a H/A/N/A, verifica
dell'ipotesi "fasi a coppie" (POSSIBLE_PATHS.md §3.3).

### S-PHASE · fase tattica — SOFT

```text
POSSESSION_CODES = {BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING TRANSITION,
                    BALL IN FINAL THIRD, BALL IN THE BOX, LONG BALL}        # da confermare
DEFENDING_CODES  = {DEFENDING IN DEFENSIVE THIRD, DEFENDING IN MIDDLE THIRD,
                    DEFENDING IN ATTACKING THIRD, DEFENSIVE TRANSITION}

when:  ST con code in POSSESSION_CODES e Team ∈ {H, A}   → emit Team
       ST con code in DEFENDING_CODES  e Team ∈ {H, A}   → emit other(Team)   # se l'ipotesi regge
conf:  0.4 ; 0.6 se la fase gemella (stessi start/end, Team opposto) è presente e coerente
note:  le fasi non sono possessi: non decidono i confini, aiutano a riempire
       `PLAYERS IN THE BOX` e `GOALS`: esclusi da S-PHASE, usati da D-GOAL / R-SET
```

### S-SP · tipo di set piece — usato da D-SP e dalle RESTART

```text
Type = Corner Kick → rinforza R-CORNER ; Goal Kick → R-GOALKICK ;
Kick Off (*) → D-KO ; Throw-in / Free Kick → R-TOUCH (Team spesso N/A)
```

### S-TRANS · transizioni — TURNOVER (SOFT)

`ATTACKING TRANSITION` / `DEFENSIVE TRANSITION` descrivono per definizione il momento successivo a
un recupero palla: sono il segnale ST più vicino a un cambio di possesso.

```text
when:  inizio di un ATTACKING TRANSITION con Team = T  (o DEFENSIVE TRANSITION con Team = other(T))
emit:  turnover candidato verso T, istante = inizio del tag, tolleranza ± qualche secondo
uso:   rinforza T-REV / T-INTERCEPT / T-TACKLE vicini; da solo non basta (serve controllo, §6.2)
       durante un buco della palla, insieme a T-TEAMFLIP, può bastare
```

### S-FLIP · cambio di squadra tra fasi consecutive — TURNOVER (SOFT)

```text
when:  fase con possesso di T (S-PHASE) seguita entro pochi secondi da fase con possesso di other(T),
       senza SET PIECES in mezzo
emit:  turnover candidato verso other(T) nella finestra tra le due fasi
note:  se in mezzo c'è un SET PIECES, il cambio passa per un D → lo gestiscono le RESTART
```

### S-ZONE · zona della palla — controllo di coerenza

```text
BALL IN FINAL THIRD / BALL IN THE BOX con Team = T:
    → T attacca (voto SOFT a T)
    → controllo: la palla ATD, se visibile, deve stare nel terzo/area offensivo di T
      (attack_dir). Se non ci sta → il tag è dubbio: abbassa la qualità stimata di S-* (§4.1)
PLAYERS IN THE BOX con Team = T → input per R-SET a palla ferma
LONG BALL con Team = T → possesso di T durante il volo; nessun cambio finché qualcuno non controlla
```

### S-GOAL · goal del tagging — solo come conferma

```text
GOALS: mai una regola a sé. Rinforza D-GOAL solo se l'ATD vede la palla in porta.
Kick Off (After Goal) con Team = T: conferma che T ha subìto → controllo incrociato con P-GOAL
GOALS senza kick-off successivo nell'ATD → D-GOAL-VOID (goal annullato o falso positivo del tag)
```

### S-QUALITY · affidabilità dello Smart Tagging

Lo ST è automatico: ci sono falsi positivi (es. più `GOALS` dei goal convalidati, per goal annullati o
errori) e falsi negativi (solo 5 `Free Kick`). Regole d'uso:

1. **Mai da solo per un'ANCHOR.** Ogni regola S-* è SOFT o rinforza una regola ATD.
2. **Qualità stimata per code**, non per tutto lo ST: con il label model (§4.1) ogni code ha la
   sua `q` (es. `SET PIECES` affidabile per il D, `PROGRESSION` meno per i confini).
3. **Controlli di coerenza automatici** nel report:
   - code di zona coerente con la posizione della palla ATD;
   - coppie di fasi (se l'ipotesi regge) con `Team` opposti;
   - `Kick Off (After Goal)` preceduto da palla in porta;
   - set piece con palla ferma nell'ATD all'inizio dell'intervallo.
   La percentuale di coerenza per code diventa parte di `q`.

---

## 9. Aggregazione e inerzia (il motore)

```text
# 1. contesto
for f in frames:
    context[f] = DEAD if any ANCHOR D attiva su f (con D-HOLD) else OPEN_PLAY

# 2. voti (peso = qualità della regola × forza dell'attivazione, §4.1)
for f in frames:
    votes[f] = [(rule.id, rule.emit, rule.q * rule.strength(f)) for rule in RULES if rule.fires(f)]
    p[f]     = combine(votes[f])        # v0: voto pesato normalizzato su {H, A, D}
                                         # v1: label model (accuratezze stimate da accordi/disaccordi)
    conf[f]  = max(p[f])                 # esportata nel report: dove il motore è incerto

# 2b. candidati di cambio possesso (§6.2), accettati solo con conferma di controllo

# 3. ancore
anchor[f] = label se esiste un voto ANCHOR con conf >= CONF_ANCHOR e nessun ANCHOR in conflitto
            CONFLICT se due ANCHOR discordano  → log, nessuna ancora su f

# 4. inerzia (v0, isteresi) — per ogni tempo, in ordine
state = None
for f in graded frames of half:
    if anchor[f] == D:                       state = D
    elif state == D and restart_kick(f):     state = restart_team(f) or argmax_HA(p[f])
    elif anchor[f] in {H, A} and anchor[f] != state:
        # cambio: serve evidenza di controllo sostenuta, non un singolo frame
        if anchor sostenuta per >= INERTIA_MIN:  state = anchor[f] (retrodatato all'inizio dell'ancora)
    elif anchor[f] is None and state is None:
        state = argmax_HA(p[f]) se max >= CONF_SOFT_START
    label[f] = state if state is not None else X
    # NB: solo regole SOFT non bastano mai a cambiare state

# 5. v1: sostituire il passo 4 con HSMM/Viterbi, con le ancore come vincoli
```

---

## 10. Edge case e come li gestisce il motore

| Caso | Comportamento atteso | Meccanismo |
|---|---|---|
| Corner/punizione respinti, palla che torna a chi ha battuto | possesso invariato | inerzia: nessun controllo avversario |
| Respinta della difesa senza controllo | possesso invariato | il rinvio non crea un `control_episode` |
| Duello aereo vinto, nessun controllo dopo | possesso invariato | inerzia |
| Duello aereo vinto, controllo dopo | cambio al momento del controllo | P-DRIB / P-FREE dell'altra squadra |
| Palla in aria dopo un lancio di A | resta A finché B non controlla | conta l'ultimo **controllo**, non l'ultimo tocco |
| Palla in aria senza Z | dedotta: palla persa / picco di velocità / convergenza di giocatori | primitive §2.3 (`ball_z` è sempre vuota) |
| Contrasto prolungato (palla contesa) | possesso a chi l'aveva | P-DRIB con conf ridotta, inerzia |
| Autogol | possesso prima del goal = squadra che attaccava quella porta | D-GOAL / P-GOAL via attack_dir |
| Goal annullato | possesso di chi ha segnato fino alla porta, D fino alla ripresa effettiva, poi chi riprende | D-GOAL-VOID + P-GOAL + consenso R-* |
| Retropassaggio / cambio gioco (palla che inverte direzione) | possesso invariato | T-REV richiede un avversario sulla palla |
| Intercetto con fuga in avanti (nessuna inversione) | cambio | T-INTERCEPT |
| Palo / parata respinta | invariato finché qualcuno non controlla | T-LOOSE |
| Cambio di possesso con palla non visibile | cambio nel buco, istante incerto | T-TEAMFLIP + S-TRANS |
| Portiere con palla in mano | possesso del portiere (da verificare) | Q3 |
| Rimessa laterale (`Team = N/A`) | D fino al lancio, poi chi lancia | D-OUT + R-TOUCH |
| Buco lungo della palla in gioco | trascina l'ultimo possesso, salvo D-GAP | inerzia; politica X da decidere |

---

## 11. Domande da girare alla GT pubblica (significato, non soglie) — task Gio

Uso consentito: capire cosa significano le etichette. Le risposte vanno scritte qui in forma
qualitativa ("il D inizia quando…"), **mai** come valori numerici da copiare nelle soglie.

- **Q1** — Dopo un fallo: il D comincia al contatto/fischio o solo quando la palla si ferma?
- **Q2** — Corner: la preparazione (giocatore alla bandierina) è D? Il D finisce alla battuta?
- **Q3** — Portiere con palla in mano / rinvio dalle mani: è possesso della sua squadra o D?
- **Q4** — Rimessa laterale: il D va dall'uscita della palla al lancio?
- **Q5** — Respinte e rimpalli: la GT cambia possesso sul tocco o solo dopo un controllo?
- **Q6** — Palla persa dal tracker a lungo durante il gioco: che etichetta ha la GT in quel tratto?
- **Q8** — In un cambio di possesso, la GT mette il confine sul tocco dell'avversario (intercetto)
  o più tardi, quando il controllo è stabilito? Decide l'istante restituito da §6.2.
- **Q7** — Durata tipica dei D brevi (falli) rispetto a quelli lunghi (goal, sostituzioni): solo
  come ordine di grandezza per la plausibilità, non per le soglie.

---

## 12. Dipendenze e passaggio al motore

### Dipendenze dagli altri task

| Serve a | Da chi | Cosa |
|---|---|---|
| §2.1, §2.3 | Gio | unità della velocità; slot vuoti/duplicati; scambi di slot; buchi della palla; conferma che `ball_z` è assente ovunque |
| §8 | Francesco | ST → frame, `Team` normalizzato, verifica fasi a coppie, sovrapposizioni |
| §11 | Gio | risposte Q1–Q7 |

### Struttura del codice (v0.5: ✅ esiste · ⬜ da fare)

```text
src/tikitalia/
├─ config/
│  ├─ params.yaml        ✅ tabella §3: valore, unità, origine, motivazione (unica fonte delle soglie)
│  ├─ rules.yaml         ✅ registro: classe, kind, scope, prior_q, parametri dichiarati
│  └─ geometry.yaml      ✅ misure del campo
├─ preprocessing.py      ✅ ATD/ST → Match in metri, pulizia palla, velocità della palla
├─ geometry.py           ✅ Pitch: linee, porte, aree
├─ features.py           ✅ primitive §2.3 per frame (array numpy, vettorizzato) — FEATURES.md
├─ events.py             🟡 calcio d'inizio, attack_dir ✅ · control_episode, pass_event, restart_kick ⬜
├─ rules/
│  ├─ base.py            ✅ Rule / RuleOutput (label, strength, episodes)
│  ├─ registry.py        ✅ classi registrate, istanziate da rules.yaml
│  ├─ strength.py        ✅ z-score, saturazione, fattore di durata
│  ├─ dead.py            🟡 D-OUT, D-GOAL ✅ · altre D ⬜
│  ├─ possession.py      ⬜ P-*
│  ├─ turnover.py        ⬜ T-* (§6.2)
│  ├─ restart.py         ⬜ R-* + consenso
│  └─ smart_tagging.py   ⬜ S-*
├─ pipeline.py           ✅ config → Match → context (eventi + feature) → uscite delle regole
├─ engine.py             ⬜ aggregazione + inerzia (§9) → etichette per frame — MOTORE.md
├─ submission.py         ⬜ scrittura del CSV
└─ report.py             🟡 report per D-OUT, D-GOAL
```

Ogni regola è una funzione **vettorizzata** sugli array delle feature (199k righe): niente loop
per frame in Python, salvo il passo di inerzia (o Numba).

### Test

- **Test unitari su scenari sintetici** costruiti a mano (una palla ferma al centro con le squadre
  nelle proprie metà → D-KO scatta). Sono dati inventati, non frame di BAR-ATH: nessun problema di
  regole.
- **Report per regola** su BAR-ATH: copertura, conflitti, accordo con ST (metrica M2).
- **Precisione delle regole solo-ATD** su partite open con etichette (M1), quando sarà pronto.
- Prima di ogni consegna: `validate_submission.py` e controllo di allineamento sui 5 minuti pubblici.

---

## 13. Aperte

- Peso relativo ANCHOR / SOFT nel voto v0, e soglie `CONF_ANCHOR`, `CONF_SOFT_START`.
- Politica `X` sui buchi lunghi della palla in gioco.
- `INERTIA_MIN`: a priori o dalla distribuzione delle durate degli episodi di controllo?
- Quando passare dal voto pesato al label model, e dall'isteresi all'HSMM.
- **Scala dello z-score per condizione.** Nei test su BAR-ATH lo z con σ = rumore della palla
  (0,1 m) satura sempre (D-OUT mediana 32, D-GOAL "palla persa in area" 85–95): misura la
  precisione del sensore, non l'incertezza della condizione. Proposta: σ_i = max(σ sensore,
  tolleranza semantica a priori della condizione), es. ~2 m per "dentro l'area", ~1 m per
  "oltre la linea"; da decidere insieme e documentare in params.yaml.
  v0.5: confermato dalle feature — anche σ dei giocatori stimato su BAR-ATH finisce sul pavimento
  (0,10 m): il tracking corretto è già smussato e il "rumore del sensore" non è misurabile. Da
  decidere **prima** della forza di P-DRIB / `control_episode` (FEATURES.md §4.7).

---

## 14. Proposte del team (27/10) e dove stanno

| Proposta | Regola | Stato |
|---|---|---|
| Dopo un goal: GOALS dello ST + palla che esce verso la linea di porta | D-GOAL (v0.4, dal calcio d'inizio all'indietro) | riscritta |
| Tiro che esce: direzione, velocità, posizione; uscita entro n secondi | D-SHOT | nuova |
| Buchi lunghi senza palla | D-GAP, solo con rallentamento collettivo | precisata |
| Rallentamento collettivo + palla ferma 3–5 s, oppure palla calciata via | D-WHISTLE varianti A e B | estesa |
| Un tiro che esce è palla morta? | Sì, Laws 9 → D-OUT / D-SHOT | risposta |
| Rimessa laterale per l'altra squadra | R-THROW + R-LAST; per la metrica è sempre possesso → D → possesso | precisata |
| Palla ferma in area piccola, difensori vicino, portiere che calcia | R-GOALKICK con disposizione da rinvio | estesa |
| ≥ 2 passaggi tra giocatori della stessa squadra | P-PASS (`N_PASS_ANCHOR` = 2 ricezioni) | c'era |
| Direzione + velocità + posizione di palla e giocatore vicino | P-DRIB (conduzione) | c'era |
| Transizioni dello ST | S-TRANS | c'era |
| Calcio d'inizio dopo un goal / inizio 2° tempo | D-KO + R-KO, D dall'inizio del blocco valutato al calcio | precisata |
