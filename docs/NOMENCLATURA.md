# Nomenclatura: valutazione delle regole di pseudo-labelling

Riferimento per i termini usati in `REGOLE_PSEUDO_LABEL.md` (§4.1, §9) e in
`ESEMPI_VALUTAZIONE_REGOLE.md`. In ogni voce: cosa è, a cosa si riferisce (regola, condizione o frame)
e come si ottiene. I termini del motore (ancora, inerzia, stato) sono al §6, quelli delle feature
al §7 (spiegazione completa in `FEATURES.md`).

---

## 1. Struttura di una regola

**Regola** (`r`)
: Un'unità di voto. Ha un identificativo (es. `P-DRIB`), un tipo, una o più condizioni in AND, e
  un'etichetta che emette quando tutte le condizioni sono vere. Un OR si scrive come due regole
  separate.

**Tipo di regola**
: - `ANCHOR`: etichetta un tratto; il motore non la ribalta se il peso supera `CONF_ANCHOR`.
  - `SOFT`: voto pesato, serve a riempire i tratti scoperti e a rompere la parità.
  - `RESTART`: non etichetta il tratto di palla ferma, dice quale squadra riprende il gioco.
  - `TURNOVER`: propone l'istante di un cambio di possesso; il motore lo accetta solo se
    confermato (vedi *finestra di conferma*).

**Condizione** (`C_i`)
: Un confronto `feature ≷ soglia`, valutato su un frame. Può essere:
  - *istantanea*: basta che valga al frame `f`;
  - *di durata* ("per ≥ T"): deve valere di continuo per almeno `T` secondi. Una condizione di
    durata può riguardare una sola feature o un blocco di altre condizioni ("C1, C2 e C3 insieme
    per ≥ 0,4 s").

**Feature** (`x_i`)
: La grandezza misurata dal tracking su cui si basa una condizione: distanza palla–giocatore,
  velocità della palla, numero di giocatori in area, ecc. È una misura, mai una decisione:
  vedi §7.

**Soglia** (`θ_i`)
: Il valore con cui si confronta la feature. Ha un nome (es. `R_CONTROL`), un valore iniziale e
  un'origine dichiarata (*a priori* o *dai dati*) nella tabella dei parametri.

**`T`** (durata minima)
: Proprietà di una **condizione** di durata, non della regola. Una regola può avere più condizioni
  di durata, ognuna con il proprio `T`, oppure nessuna.

**Emissione** (`ℓ_r(f)`)
: L'etichetta che la regola `r` emette al frame `f` quando scatta: `H`, `A` o `D` (per le RESTART,
  la squadra che riprende; per le TURNOVER, la squadra verso cui avviene il cambio).
  **Non è l'errore.**

---

## 2. Forza dell'attivazione (per frame)

**Margine** (`m_i`)
: Di quanto il valore della feature sta dentro la soglia, con il segno scelto in modo che
  `m_i > 0` voglia dire "condizione soddisfatta".
  - condizione `x ≤ θ` → `m = θ − x`
  - condizione `x ≥ θ` → `m = x − θ`

**Rumore** (`σ_i`)
: L'errore di misura tipico della feature, cioè di quanto il tracking "tremola". Si stima senza
  etichette:
  1. si smussa la traccia (es. media mobile su 5 frame);
  2. si calcola il residuo `grezza − smussata`;
  3. `σ` è la deviazione standard del residuo.

  Regole di composizione:
  - distanza tra due oggetti: `σ = √(σ_a² + σ_b²)`;
  - grandezze derivate (velocità, accelerazione, angoli di direzione): si stima sul residuo della
    grandezza derivata, che è più rumorosa della posizione;
  - conteggi (n giocatori, n passaggi): `σ = 1`.

  Su BAR-ATH il σ stimato così finisce sul pavimento (`SIGMA_POS_FLOOR` = 0,10 m) sia per la palla
  sia per i giocatori: il tracking è già smussato e il rumore del sensore non si vede. Aperto in
  REGOLE §13: usare `σ = max(σ sensore, tolleranza semantica della condizione)`.

**z-score** (`z_i`)
: `z_i = m_i / σ_i`. Di quanti "errori di misura" il valore sta dentro la soglia. Con rumore circa
  gaussiano, la probabilità che la condizione risulti vera solo per colpa del rumore è circa
  `Φ(−z)`: 16% a z = 1, 2,3% a z = 2, 0,13% a z = 3.

**Saturazione** (`Z_SAT`)
: Il livello di z oltre il quale la condizione è considerata certa e la forza smette di crescere.
  Valore a priori: `Z_SAT = 3`, perché oltre 3σ la probabilità di un falso dovuto al rumore è
  trascurabile (0,13%). È unico per tutte le regole, per non introdurre un parametro libero per
  regola. Si verifica con la calibrazione (§4): se la precisione continua a crescere oltre z = 3
  lo si alza, se è già piatta da z = 2 lo si abbassa.

**Fattore di durata** (`d_i`)
: Solo per le condizioni di durata: `d_i = min(1, durata_osservata / (2·T_i))`. Vale 0,5 appena
  superato `T`, 1 da `2T` in su. Per le condizioni istantanee `d_i = 1`.

**Forza della condizione** (`s_i`)
: `s_i = clip(z_i / Z_SAT, 0, 1) · d_i`

**Condizione più debole**
: Quella con la forza `s_i` più bassa. Le condizioni sono in AND: se una è appena soddisfatta,
  basta un po' di rumore per farla cadere e con lei cade la regola.

**Forza dell'attivazione** (`s_r(f)`)
: `s_r(f) = min_i s_i`. Misura per frame: quanto è netta *questa* attivazione.

---

## 3. Qualità della regola (per regola)

**Etichetta vera** (`y(f)`)
: L'etichetta corretta del frame. Disponibile solo su partite etichettate esterne (M1); sulla
  partita di gara si usa un riferimento sostitutivo (consenso o label model).

**Errore**
: La regola `r` sbaglia al frame `f` se scatta e `ℓ_r(f) ≠ y(f)`. Si conta in modo diverso per
  tipo di regola:
  - `ANCHOR` / `SOFT`: per frame (e, in parallelo, per episodio);
  - `RESTART`: una volta per ripresa (squadra giusta o sbagliata);
  - `TURNOVER`: per evento, con la *tolleranza di valutazione* `Δt`.

  I frame entro ±k da un cambio di possesso vero si escludono o si pesano poco, perché anche la
  GT lì è ambigua.

**Episodio**
: Un tratto continuo di frame in cui la regola scatta. Contare gli errori anche per episodio evita
  che un solo sbaglio su un tratto lungo (centinaia di frame a 25 fps) affondi la regola.

**Copertura**
: Frazione dei frame in cui la regola scatta. Dice quanto spesso vota, non quanto è affidabile.
  **Non entra nel peso**: una regola frequente pesa già di più perché vota più spesso.

**Qualità** (`q_r`)
: La precisione della regola: `q_r = P(y = ℓ_r | r scatta) = corretti_r / attivazioni_r`.

**Prior** (`conf_prior`, `n₀`)
: La qualità dichiarata a priori nella scheda della regola, con un peso `n₀` espresso in
  "osservazioni fittizie" (es. 30 episodi). Si combina con i conteggi osservati:

  ```text
  q_r = (conf_prior · n₀ + corretti_r) / (n₀ + attivazioni_r)
  ```

  Con poche attivazioni `q` resta vicino al prior, con molte è dominata dai dati.

**Fonti dei `corretti`**, in ordine di affidabilità:
: 1. **M1**: etichette vere su altre partite. Unico errore "vero"; vale per le regole solo-ATD.
  2. **Consenso leave-one-out**: il riferimento per `r` è l'uscita del motore senza `r`, solo sui
     frame dove quel consenso è sicuro (`conf₋ᵣ(f) ≥ soglia`). Limite: non vede gli errori comuni
     a tutte le regole.
  3. **Label model** (Dawid–Skene / Snorkel): stima tutte le `q` insieme con l'EM dalla matrice
     di accordi e disaccordi. Va dichiarato quali regole sono correlate (stesse feature).

  I 5 minuti di GT pubblica servono solo al controllo finale e **non** si usano per stimare `q`.

---

## 4. Peso del voto e calibrazione

**Peso** (`w_r(f)`)
: `w_r(f) = q_r · s_r(f)`. Quanto conta il voto di `r` al frame `f`.

**Peso ottimale** (`w_ott`)
: In un voto pesato, il peso ideale di un votante con precisione `p` sono i suoi log-odds. Con
  K = 3 classi: `w_ott = ln( (K−1)·p / (1 − p) ) = ln( 2p / (1 − p) )`. Vale 0 quando `p = 1/3`
  (la regola tira a caso).

  Da dove viene: una regola con precisione `p` ha ragione con probabilità `p` e, se gli errori si
  dividono a metà tra le altre due classi, sbaglia verso ciascuna con probabilità `(1 − p)/2`. Il
  suo voto moltiplica le quote della classe votata per `p / ((1 − p)/2) = 2p / (1 − p)`. Con regole
  indipendenti i fattori si moltiplicano; passando al logaritmo si **sommano**, ed è per questo
  che il peso da sommare è il log-odds.

**Indipendenza**
: Assunzione di `w_ott`: gli errori di regole diverse non sono correlati. Due regole che usano le
  stesse feature (es. P-DRIB e P-FREE) violano l'assunzione: sommandone i log-odds si conta due
  volte la stessa evidenza. Vanno dichiarate correlate (e il label model le tratta a parte).

**Calibrazione**
: Il controllo che il peso dato a un voto corrisponda a quanto quel voto vale davvero.

  Il peso `q · s` è costruito a tavolino (z-score, saturazione, prodotto): è una promessa ("questo
  voto vale tanto"). La calibrazione confronta la promessa con la frequenza osservata con cui la
  regola ha ragione. È come per un previsore del meteo: è calibrato se, nei giorni in cui dice
  "70% di pioggia", piove davvero circa 7 volte su 10.

  Risponde a due domande per ogni regola:
  1. **La forza porta informazione?** Se la precisione cresce con `s`, un'attivazione netta
     sbaglia meno di una al limite. Se resta piatta, la forza è rumore.
  2. **La forma del peso è giusta?** Anche se `s` informa, il prodotto può dare troppo o troppo
     poco peso a certe fasce, e allora un voto mal pesato può ribaltare la decisione del motore.

  Procedura: si dividono le attivazioni in fasce di `s`, si calcola la precisione `p` per fascia,
  la si converte in `w_ott` e si confrontano le due curve normalizzate sull'ultima fascia (conta
  la forma, non la scala). Si fa su M1 o sul consenso leave-one-out, **mai sui 5 minuti
  pubblici**. Esiti possibili:
  - forme simili → il prodotto va bene;
  - fasce basse sottopesate → pavimento `s' = s₀ + (1 − s₀)·s`;
  - precisione piatta → lo z-score non è informativo per quella regola, si usa `s = 1`;
  - in generale si può stimare direttamente `p(z) = sigmoid(a_r + b_r·z)` e usare
    `w = ln(2p(z) / (1 − p(z)))`.

---

## 5. Cambio di possesso e tempo

Due intervalli di tempo diversi, da non confondere.

**Finestra di conferma** (1,5 s)
: **Fa parte della regola.** Un segnale TURNOVER (es. l'inversione della palla, T-REV) dice
  *quando* potrebbe esserci un cambio; il motore lo accetta solo se la nuova squadra prende il
  controllo entro 1,5 s. Altrimenti lo scarta.

**Tolleranza di valutazione** (`Δt`, es. ±0,5 s)
: **Serve solo a misurare l'errore**, non entra nel motore. Un cambio proposto è corretto se cade
  entro ±Δt da un cambio vero non ancora abbinato. Ogni cambio vero si abbina a una sola
  proposta.

**Precisione** (eventi)
: proposte abbinate / proposte.

**Richiamo** (eventi)
: cambi veri abbinati / cambi veri.

---

## 6. Motore: stato, ancora, inerzia

**Motore**
: Il codice che trasforma le uscite di tutte le regole in **una sola etichetta per frame**
  (`H`, `A`, `D`, `X`). Le regole non si conoscono tra loro; il motore non sa niente di calcio.
  Scorre i frame di ogni tempo **in ordine** e decide guardando due cose: i voti delle regole
  su quel frame **e lo stato del frame precedente**.

**Stato** (`state`)
: L'etichetta corrente del motore mentre scorre i frame: chi ha il possesso adesso (`H`/`A`), o
  palla morta (`D`). All'inizio di ogni tempo vale `None` (nessuna memoria) e non attraversa
  mai l'intervallo. Diventa l'etichetta del frame: `label[f] = state`.

**Ancora**
: Un frame in cui il motore è **sicuro** dell'etichetta, perché una regola `ANCHOR` scatta con peso
  `w = q · s ≥ CONF_ANCHOR` e nessun'altra `ANCHOR` la contraddice. È un'**evidenza**: dice cosa
  si vede su quel frame. Le ancore sono poche (copertura bassa) e precise.
  - Ancora `D` → lo stato passa a `D` subito.
  - Ancora `H`/`A` diversa dallo stato → il cambio si accetta solo se l'ancora dura almeno
    `INERTIA_MIN`; in quel caso il cambio si **retrodata** all'inizio dell'episodio che l'ha prodotta.
  - Ancore in disaccordo → `CONFLICT`: si registra nel report e il frame si tratta come senza
    ancora.

**Inerzia**
: La regola del motore per i frame **senza ancora**: lo stato resta quello che era. È una
  **memoria**, non un'evidenza: non dice cosa si vede, dice che senza prove contrarie il possesso
  non cambia. Traduce CHALLENGE.md: un tocco, una deviazione o un duello aereo senza controllo
  successivo non cambiano il possesso. I voti `SOFT` non la rompono mai: servono solo a scegliere
  lo stato quando è ancora `None`.

**In breve**

| | Ancora | Inerzia |
|---|---|---|
| cos'è | evidenza su un frame | memoria tra un'evidenza e l'altra |
| da dove viene | una regola `ANCHOR` con peso alto | lo stato del frame precedente |
| cosa fa | fissa o cambia lo stato | mantiene lo stato |
| quanti frame copre | pochi (dove il possesso è "facile") | tutti gli altri |
| se sbaglia | errore localizzato, si vede nel report per regola | trascina l'errore fino all'ancora successiva |

**Esempio pratico** (`INERTIA_MIN` = 1 s)

Azione di 12 secondi. H imposta da dietro, perde un pallone conteso, poi A intercetta e calcia fuori.

```text
t (s)  cosa succede in campo                         ancora            stato   perché
0–2    difensore H conduce palla                     H (P-DRIB)        H       ancora
2–3    passaggio H in volo, nessuno vicino           —                 H       inerzia
3–4    centrocampista H riceve e controlla           H (P-DRIB)        H       ancora, stessa squadra
4–4,4  contrasto: A tocca la palla, che schizza via  A per 0,4 s       H       ancora troppo breve
                                                                               (< INERTIA_MIN): tocco,
                                                                               non controllo
4,4–6  palla vagante, tracking la perde              —                 H       inerzia
6–8    A raccoglie e conduce                         A (P-DRIB)        A       ancora ≥ 1 s → cambio,
                                                                               retrodatato a t = 6
8–9    A lancia lungo, palla in volo                 —                 A       inerzia
9–11   palla oltre la linea laterale                 D (D-OUT)         D       ancora D
11     rimessa: batte H (R-THROW / R-TOUCH)          ripresa H         H       fine del D
11–12  H riceve la rimessa                           H (P-DRIB)        H       ancora
```

Da notare:
- tra t = 4,4 e 6 nessuna regola dice niente e la palla non si vede: l'etichetta `H` viene
  **solo** dall'inerzia. Se a t = 5 A avesse davvero preso il controllo senza che il tracking
  lo vedesse, l'errore durerebbe fino all'ancora di t = 6. È il prezzo dell'inerzia, ed è il
  motivo per cui servono le regole TURNOVER per i buchi della palla (T-TEAMFLIP, S-TRANS);
- l'ancora `A` a t = 4–4,4 è vera (A ha toccato la palla) ma **non** basta: un'ancora breve è
  un'evidenza che il motore registra senza cambiare lo stato;
- il cambio a `A` si scopre a t = 7 (dopo `INERTIA_MIN`), ma si scrive da t = 6: senza
  retrodatazione la metrica punirebbe 1 s di ritardo su ogni cambio.

---

## 7. Feature, eventi e regole

Spiegazione completa, formule e valori su BAR-ATH: `FEATURES.md`.

**Feature** (strato)
: Una misura per frame calcolata una sola volta da `features.py` e condivisa da tutte le regole.
  Non contiene soglie né etichette: le uniche costanti sono finestre di calcolo e guardie di
  validità, in `params.yaml`. NaN = non misurabile (mai zero).

**Evento**
: Un fatto con un inizio e una fine, ricavato applicando soglie alle feature nel tempo:
  `control_episode` ("H controlla la palla da 1,3 s"), `pass_event`, `restart_kick`. Sta tra le
  feature e le regole (`events.py`).

**Esempio dei tre strati sullo stesso frame**
: - feature: "il più vicino di H è a 0,8 m, `rel_speed` = 0,3 m/s";
  - evento: "`control_episode` di H iniziato 1,3 s fa";
  - regola: "P-DRIB emette H con forza 0,7".

**Giocatore più vicino** (`nearest_*`)
: Per ogni squadra, la traccia più vicina alla palla in quel frame, con distanza, posizione e
  velocità. È un candidato portatore, non il portatore: lo diventa con un `control_episode`.

**Gap** (`nearest_gap`)
: Quanto l'altra squadra è più lontana dalla palla: `|d_H − d_A|`. Misura quanto il pallone è
  conteso. Con un gap sotto il rumore della distanza, "chi è più vicino" lo decide il rumore.

**Velocità relativa** (`rel_speed`) e **allineamento** (`align_cos`)
: `|v_palla − v_giocatore|` e coseno tra le due velocità. Palla e giocatore che si muovono insieme
  (relativa ~0, coseno ~1) = conduzione.

**Velocità di chiusura** (`closing_speed`) e **di inseguimento** (`chase_speed`)
: Due velocità "verso la palla" diverse, con `u` = versore giocatore → palla:
  - chiusura: `(v_giocatore − v_palla) · u`, di quanto si accorcia la distanza. ~0 per chi conduce;
  - inseguimento: `v_giocatore · u`, il massimo sulla squadra. Chi corre verso una palla più veloce
    di lui insegue anche se la distanza cresce.

**Pressione** (`pressure_dist`)
: Distanza tra il giocatore più vicino alla palla e l'avversario più vicino a lui. L'unica distanza
  giocatore–giocatore usata per il possesso.

**Relativo alla direzione d'attacco** (`fwd`)
: `fwd = (x − L/2) · attack_dir`: > 0 vuol dire "nella metà della porta che la squadra attacca",
  in entrambi i tempi. Base di `centroid_fwd`, `n_opp_half`, aree, `v_forward`.

**Rallentamento collettivo** (`slowdown`)
: Velocità mediana dei giocatori diviso il suo riferimento (mediana degli ultimi `REF_WIN` s), per
  H, A e tutti. < 1 = più lenti del solito. Indica l'**inizio** di una sosta, non la sua durata:
  nelle soste più lunghe di `REF_WIN` anche il riferimento scende e il rapporto torna verso 1.
