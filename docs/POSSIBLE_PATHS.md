# Strade percorribili — MTAH27 Phase 1 (Ball Possession)

Documento di decisione per il team, prima di partire con lo sviluppo.
Domanda di partenza: **costruiamo un validation set su altre partite, oppure un modello
unsupervised che si allena direttamente su BAR-ATH?**

Risposta breve: **non sono alternative.** Sono due assi diversi:

- **Asse 1 — come impara il modello** (regole, unsupervised sulla partita, supervisionato su altre partite, …)
- **Asse 2 — come scegliamo tra due versioni del modello** (a priori, metriche interne, validation set esterno)

Un modello unsupervised elimina il bisogno di *etichette per addestrare*, non il bisogno di *un
metro per decidere*. La vera decisione è quale combinazione dei due assi adottare.

> **Aggiornamento 24/10 — ipotesi di lavoro del team.** Partiamo da un'etichettatura **a regole**:
> durante preprocessing, cleaning e feature extraction definiamo regole sui dati frame-by-frame
> dell'ATD, incrociate con lo Smart Tagging, che producono H/A/D dove sono affidabili. Le regole
> diventano un set di feature/pseudo-etichette su cui poggia il modello temporale. È consentito
> (RULES.md, "What remains fine": *pseudo-labels produced by an algorithm, provided no person
> decided any of them*), a patto di rispettare i vincoli del §3.5. Dettaglio nel **§3**; nella
> mappa delle strade corrisponde a **S1 + S4** (§4).
>
> Le regole in dettaglio (pseudocodice, parametri, confidenza, motore) sono in
> [docs/REGOLE_PSEUDO_LABEL.md](docs/REGOLE_PSEUDO_LABEL.md): quel file è il riferimento aggiornato,
> il catalogo del §3.3 qui sotto è la versione iniziale.

> Fonti: RULES.md, CHALLENGE.md, DATA.md, EVALUATION.md, FAQ.md, SUBMISSION.md di questa repo,
> più `cache/atd_columns.txt` e `cache/profile.txt` (output degli script in `scripts/`).
> Il contenuto dei file in `data/` è descritto da DATA.md; i dati della partita non vanno caricati
> su LLM in cloud (Termini e Condizioni, citati in FAQ.md).

---

## 1. Cosa c'è in `data/` e cosa permette

| File | Dimensione | Contenuto | Cosa ci si può fare |
|---|---|---|---|
| `tactical/BAR-ATH_tactical_atd.txt` + `.xml` | 187 MB | x, y (0–1), velocità di 46 tracce e della palla, 199.319 frame; squadra di ogni traccia; **nessuna identità**; palla assente nel 41% dei frame giocati | **Fonte principale.** Feature per H/A/D, forma delle squadre, fit unsupervised, ricostruzione self-supervised della palla |
| `tactical/BAR-ATH_tactical_smart_tagging.xml` | 255 KB | 638 intervalli automatici (fasi con squadra, set piece con tipo), in secondi → `round(s*25)` = frame | Indizi per **D** (ogni set piece è una ripresa) e per **H/A** (fasi con `Team`). Rumoroso; `Team = N/A` su tutte le rimesse |
| `tactical/BAR-ATH_tactical.mp4` | 4,9 GB | video di riferimento, 25 fps, 1920×1080 | Debug visivo; eventuale detector della palla pre-addestrato su altre partite |
| `tactical/BAR-ATH_tactical_homography.csv` | 22 MB | matrice 3×3 campo→pixel per frame, valida al 100% | Serve solo se si usa il video (proiettare tracking sul video o palla dal video al campo) |
| `broadcast/*` (video 13 GB, ATD 135 MB, omografia 15 MB) | — | feed TV, **non sincronizzato** col tattico, omografia valida solo al 65,8% | Possibile recupero della palla, previo allineamento. Priorità bassa |
| `evaluation/gt/…_public.csv` (in repo) | 1,9 MB | 5 minuti etichettati, 17 possessi | **Solo verifica** di allineamento e significato delle etichette. Niente training né tuning |

Dettagli dell'ATD rilevanti per le regole (da `cache/atd_columns.txt`):

- `ball_state` e `ball_owning_team_id` esistono come colonne kloppy ma sono **vuote al 100%**: nessuna
  scorciatoia, lo stato della palla va dedotto.
- `ball_speed` e `ball_z` sono vuote: **la velocità della palla va derivata** dalle posizioni
  (differenze tra frame, con smoothing). Niente altezza della palla.
- Unità della velocità dei giocatori: il metadata XML dichiara `speed_kph`, DATA.md dice m/s.
  **Da verificare** (confronto con la velocità derivata dalle posizioni) prima di fissare soglie.

## 2. Vincoli che delimitano le strade

| | Su BAR-ATH | Su altre partite |
|---|---|---|
| Usare i dati come input | ✅ | ✅ |
| Fit unsupervised / self-supervised / pseudo-etichette algoritmiche | ✅ (RULES.md, "What remains fine") | ✅ |
| Regole/euristiche generali scritte da noi (vedi §3.5) | ✅ | ✅ |
| Etichette fatte a mano, per training **o** validazione | ❌ in qualsiasi quantità | ✅ in qualsiasi quantità, se accessibili agli organizzatori e documentate |
| Dati prodotti da persone (eventi, tracking corretto, annotazioni, risultato reale) | ❌ | ✅ |
| GT pubblica 5 minuti | solo verifica | — |
| Costanti/regole legate a frame specifici visti a occhio | ❌ | — |

Inoltre: la submission deve essere **rigenerabile identica** dalla repo (regola 3), ranking su
**spellF1** (macroF1 come spareggio), differenze di spellF1 sotto ~0,04 sono rumore.

---

## 3. Ipotesi di lavoro: feature + regole + Smart Tagging

### 3.1 Idea

Ogni regola è una **funzione di etichettatura** (*labeling function*, LF): guarda le feature di un
frame (o di una finestra) e restituisce `H`, `A`, `D` oppure **si astiene**. Le regole non devono
coprire tutto: devono essere **affidabili dove si esprimono**. I frame su cui nessuna regola è
sicura vengono decisi dal modello temporale (§3.4), che propaga le ancore delle regole.

Tre principi:

1. **Una regola = un'evidenza, non una sentenza.** L'output di ogni LF entra come feature
   (voto + confidenza), non come etichetta finale. Le regole "dure" (sovrascrivono tutto) sono
   riservate ai casi quasi certi: palla fuori dal campo, kick-off.
2. **Lo Smart Tagging è una fonte di LF come le altre.** È automatico e rumoroso (DATA.md): lo
   si incrocia con l'ATD, non lo si copia. Un tag vale di più quando l'ATD lo conferma.
3. **Ogni LF è generale e documentata**: condizione, soglie, motivazione calcistica, copertura
   misurata. Finisce così com'è nel documento tecnico.

### 3.2 Pipeline

```
ATD raw ──► 1. preprocessing / cleaning ──► 2. feature per frame ──┐
Smart Tagging ──► intervalli → frame, mappatura code→lato ─────────┤
                                                                   ▼
                                       3. labeling functions (H/A/D/astieni + confidenza)
                                                                   ▼
                                       4. aggregazione (voto pesato / label model)
                                                                   ▼
                                       5. modello temporale (isteresi o HSMM/Viterbi) → segmenti
                                                                   ▼
                                       6. politica X + submission (199.319 righe)
```

**1. Preprocessing e cleaning**
- Allineamento: `video_frame = frame_id − 1`; Smart Tagging `round(s·25)`; solo frame valutati
  per le statistiche (`half_video_blocks`).
- Coordinate in metri (105,3 × 68,0, da `games.py`).
- **Direzione d'attacco per tempo**, calcolata dai dati: ad es. baricentro x di ciascuna squadra
  nei primi secondi di ogni tempo (al kick-off ogni squadra è nella propria metà). Mai cablata a
  mano: serve a tutte le regole "metà campo propria / porta attaccata".
- Palla: rimozione dei salti fisicamente impossibili (> ~40 m/s, 0,14% dei frame), velocità e
  accelerazione derivate con smoothing; flag "fuori dal box 0–1" (4,19% dei frame con palla).
- Tracce: tracklet molto brevi, duplicati (> 11 giocatori di una squadra nel 12,9% dei frame),
  slot sempre vuoti (22, 23, 44).
- Smart Tagging: intervalli in frame, `Team` normalizzato (`Barcelona`→H, `Atletic Club`→A,
  `N/A` esplicito), confronto case-insensitive.

**2. Feature per frame** (bozza, da estendere)

> Aggiornamento 28/10: le feature di prossimità, forma e ritmo sono implementate in
> `src/tikitalia/features.py`. Descrizione, formule e valori su BAR-ATH: [FEATURES.md](FEATURES.md).
> Le feature Smart Tagging e di contesto restano da fare (le prime con il join ATD–ST, le seconde
> nel motore). La tabella qui sotto è la bozza iniziale.

| Gruppo | Feature |
|---|---|
| Palla | visibile sì/no; x, y; velocità e accelerazione derivate; ferma da N frame; fuori campo; vicino a una linea; in area; distanza dal centro del campo; tempo dall'ultimo frame visibile |
| Prossimità | distanza del giocatore H più vicino e dell'A più vicino; squadra più vicina; n. giocatori per squadra entro 2 m / 5 m; da quanti frame la stessa squadra è la più vicina; velocità relativa giocatore–palla |
| Forma squadre | baricentro x/y per squadra; ampiezza (sd x + sd y); n. giocatori nella propria metà; linea difensiva; velocità media di squadra e sua componente verso la porta attaccata |
| Smart Tagging | code attivi; squadra "in possesso" secondo il tag (vedi §3.3); set piece attivo e tipo; tempo dall'inizio/fine del tag |
| Contesto | ultima squadra con evidenza forte di controllo e da quanto; ultima LF "dura" scattata |

**3. Labeling functions** — catalogo in §3.3.

**4. Aggregazione.** Si parte da un voto pesato con pesi a priori; poi si può passare a un
*label model* (stile Snorkel) che stima l'accuratezza di ciascuna LF **dagli accordi e disaccordi
tra LF**, senza etichette: è apprendimento unsupervised, quindi consentito su BAR-ATH.

**5. Modello temporale.** Le probabilità per frame vanno in un livellatore che produce segmenti:
isteresi (cambio di possesso solo dopo N frame di evidenza contraria) come primo passo, HSMM con
Viterbi (S2) come passo successivo. È qui che si guadagna su spellF1.

**6. Politica X.** `X` sui frame non valutati. Sui frame valutati con palla persa a lungo si può
astenersi, sapendo che spellF1 riempie con l'ultima etichetta: va bene dentro un possesso, costa
caro a cavallo di un cambio (CHALLENGE.md).

### 3.3 Catalogo iniziale delle regole

Le soglie indicate sono **valori di partenza a priori**, da fissare con i criteri del §3.5.

#### Dead ball (D)

| LF | Condizione | Output | Fonte | Affidabilità attesa |
|---|---|---|---|---|
| D1 · palla fuori | palla visibile fuori dal box 0–1 (oltre un margine di tolleranza) | D | ATD | alta (attenzione al rumore vicino alle linee) |
| D2 · palla ferma + set piece | `SET PIECES` attivo **e** palla ferma (< ~0,5 m/s) da ≥ 1 s | D | ATD + ST | alta |
| D3 · kick-off | palla ferma vicino al centro **e** tutte e due le squadre nella propria metà | D fino al primo movimento della palla | ATD (+ ST `Kick Off`) | molto alta |
| D4 · goal | vedi regola proposta n. 2, §3.4 | D | ATD + ST `GOALS` | alta se confermata dall'ATD |
| D5 · ripresa da fermo | palla ferma vicino a linea laterale/bandierina/area piccola con giocatori fermi | D | ATD | media |

#### Possesso H/A

| LF | Condizione | Output | Fonte | Affidabilità attesa |
|---|---|---|---|---|
| P1 · controllo prolungato | giocatore di una squadra entro ~1–1,5 m dalla palla per ≥ ~1 s, palla a velocità compatibile con conduzione | squadra del giocatore | ATD | alta (è il proxy "keeps the ball" di CHALLENGE.md) |
| P2 · sequenza di passaggi | ≥ 2 ricezioni consecutive (palla veloce → palla vicina a un giocatore) della stessa squadra | quella squadra | ATD | alta (proxy "more than two passes") |
| P3 · chi riprende il gioco | alla fine di un D, squadra del primo giocatore che tocca la palla | quella squadra | ATD | alta |
| P4 · tipo di set piece | `Corner Kick` → chi attacca quella porta; `Goal Kick` → chi difende quella porta; `Kick Off (After Goal)` → chi ha subìto; `Free Kick`/`Throw-in` → squadra del giocatore sulla palla | squadra che riprende | ST + ATD (direzione d'attacco) | alta; risolve il `Team = N/A` delle rimesse |
| P5 · fase tattica | code ST attivo con `Team` (es. `BUILD UP`, `PROGRESSION`, `CREATING CHANCES`) | squadra in possesso secondo il tag | ST | media: sono fasi, non possessi |
| P6 · forma delle squadre | regola proposta n. 1, §3.4 | H o A | ATD | bassa da sola: solo come feature/voto debole |
| P7 · continuità senza palla | palla non visibile, nessuna evidenza contraria | ultima squadra in possesso (o astensione) | contesto | dipende dalla durata del buco |

**Smart Tagging: le fasi sembrano a coppie.** In `cache/profile.txt` i conteggi e le durate mediane
coincidono a due a due: `PROGRESSION` ↔ `DEFENDING IN MIDDLE THIRD` (66, 17,6 s),
`CREATING CHANCES` ↔ `DEFENDING IN DEFENSIVE THIRD` (30, 30,0 s), `BUILD UP` ↔
`DEFENDING IN ATTACKING THIRD` (22, 20,5 s), `ATTACKING TRANSITION` ↔ `DEFENSIVE TRANSITION`
(24, 15,4 s); e il 41,5% dei frame è coperto da tag di entrambe le squadre. Ipotesi: ogni fase è
descritta due volte, dal lato di chi attacca e da quello di chi difende. Se è così, per i code
`DEFENDING …` la squadra **in possesso è l'avversaria** di `Team`, e la coppia fornisce una doppia
conferma. **Da verificare con uno script** (stessi start/end, `Team` opposti) prima di scrivere P5.

### 3.4 Le due regole proposte, riviste

#### Regola n. 1 — posizione e velocità delle squadre → possesso

> *Se i giocatori H sono nella propria metà campo, la palla anche, e le velocità di giocatori e
> palla sono sotto/sopra una soglia, allora il possesso è H.*

**Utilizzabile, ma come evidenza debole (P6), non come regola dura.** Il motivo: "palla e squadra H
nella metà di H" descrive due situazioni opposte con la stessa geometria:
- H costruisce dal basso → possesso **H**;
- A attacca e H difende schierata → possesso **A**.

Le feature di posizione da sole non le distinguono; la differenza sta in **come** è disposta la
squadra e in **chi** è legato alla palla. Come renderla più discriminante:
- aggiungere il rapporto con la palla: la squadra più vicina e da quanto tempo (P1);
- forma: chi ha la palla tende ad allargarsi, chi difende a compattarsi (in `profile.txt` l'ampiezza
  mediana è 29,1 m per H contro 25,9 m per A, ma va guardata frame per frame, non in media);
- direzione del movimento: la squadra in possesso avanza verso la porta attaccata, quella senza
  palla scappa verso la propria;
- calcolare "propria metà" con la direzione d'attacco **per tempo** (§3.2), perché al secondo
  tempo le squadre invertono il campo;
- confermare con lo Smart Tagging: `BUILD UP` con `Team = Barcelona` (o la coppia
  `DEFENDING IN ATTACKING THIRD` di A, se l'ipotesi del §3.3 regge) rafforza H.

Le soglie di velocità vanno fissate dopo aver chiarito l'unità (kph o m/s, §1).

#### Regola n. 2 — goal → D dopo, squadra che segna prima

> *Dopo un goal la palla è D; prima del goal il possesso è H o A secondo la squadra a cui
> appartiene l'evento goal.*

**Valida e forte, con quattro precisazioni:**
1. **Il momento del goal va trovato nell'ATD.** Gli intervalli `GOALS` hanno durata mediana
   169 s: comprendono azione, esultanza e ripresa. Il D inizia quando la palla supera la linea di
   porta tra i pali (x oltre 0/1, y entro la larghezza della porta, oppure la palla sparisce lì);
   finisce con il kick-off (`Kick Off (After Goal)` + D3).
2. **"Prima del goal" va limitato.** La regola fissa solo la parte finale dell'azione: dall'ultimo
   cambio di possesso rilevato dagli altri segnali fino al goal. Non va estesa a ritroso per una
   durata fissa.
3. **Autogol.** In quel caso la squadra in possesso prima del goal può essere quella che subisce.
   Più robusto: la squadra in possesso è quella che **attacca la porta** in cui entra la palla
   (con la direzione per tempo) e il `Team` del tag serve da conferma; se i due segnali non
   concordano, la LF si astiene.
4. **Il tag `GOALS` può sbagliare.** È automatico (4 istanze). Si accetta un goal solo se l'ATD lo
   conferma: palla nella porta **e** kick-off a centrocampo nei secondi successivi. Il risultato
   reale della partita **non** si usa per controllare: è un dato prodotto da persone sulla
   partita (RULES.md, regola 2).

Generalizzazione utile: la stessa logica vale per **ogni ripresa** (P3, P4). Dopo ogni D c'è "un
possessore chiaro" (CHALLENGE.md): le riprese sono le ancore più affidabili che abbiamo.

### 3.5 Vincoli sulle regole (per restare in regola)

- **Niente frame specifici.** Le condizioni sono generali ("palla ferma vicino alla bandierina"),
  mai costanti legate a momenti precisi (RULES.md, regola 1).
- **Soglie: a priori o dai dati, mai dalla GT.** Due modi leciti: un valore con significato
  calcistico dichiarato ("controllo = 1 s entro 1,5 m"), oppure un valore ricavato senza etichette
  dalla distribuzione dei dati (quantili, mixture model). I 5 minuti pubblici non si usano per
  scegliere le soglie né per scegliere tra regole (RULES.md, regola 2; EVALUATION.md).
- **Niente dati umani sulla partita**: event data, risultato, formazioni come input.
- **Deterministiche e versionate**: stesso codice + stessi dati = stesso file (regola 3).
- **Documentate**: per ogni LF condizione, motivazione, copertura, conflitti.

### 3.6 Come si misura senza etichette

- **Per LF:** copertura (% frame su cui si esprime), sovrapposizione e **conflitti** con le altre
  LF, accordo con lo Smart Tagging. Un tasso di conflitto alto indica una regola da rivedere.
- **Plausibilità del risultato:** numero di possessi e durata media (la GT pubblica ha 17 possessi
  in 5 minuti, media 14,2 s: è un ordine di grandezza da controllare, non un obiettivo da
  ottimizzare), quota di D.
- **Validazione esterna (M1):** le LF che usano solo l'ATD si possono misurare su altre partite con
  etichette (precisione per regola). Quelle basate sullo Smart Tagging no, perché lì non c'è.
- **GT pubblica:** solo per controllare che le etichette significhino ciò che crediamo (es. quando
  comincia il D di una rimessa).

---

## 4. Asse 1 — come impara il modello

> L'ipotesi di lavoro del §3 corrisponde a **S1 + S4**: le regole di S1 diventano le LF di S4, e
> il modello temporale di S2 si appoggia sopra. S3 resta il componente per la palla mancante.

### S1 · Euristiche + smoothing evoluto ("baseline++")
Regole esplicite sull'ATD: giocatore più vicino con isteresi (cambio solo dopo N frame di
controllo), `D` da palla fuori campo / palla ferma / set piece dello Smart Tagging, `X` dove la
palla manca (lo scorer lo riempie con l'ultima etichetta, quindi il possesso viene "trascinato").

- **Usa:** ATD, Smart Tagging.
- **Pro:** veloce, spiegabile, nessun training.
- **Contro:** molte soglie; senza un metro sono scelte alla cieca. Tetto di prestazione basso.
- **Stato:** **adottata** come base, nella forma strutturata del §3 (catalogo di LF).

### S2 · Modello a stati unsupervised su BAR-ATH (HMM / HSMM con EM)
Tre stati latenti H/A/D; osservazioni = feature per frame (squadra vicina alla palla, distanza,
velocità e posizione palla, baricentro/ampiezza delle squadre, Smart Tagging). Baum-Welch stima
le emissioni sui 199k frame senza etichette, Viterbi restituisce **segmenti** — ciò che spellF1
premia. Una versione semi-Markov impone una durata minima ai possessi.

- **Usa:** ATD, Smart Tagging.
- **Pro:** si adatta al rumore reale della partita; la penalità di transizione controlla
  direttamente il numero di possessi; consentito esplicitamente.
- **Contro:** gli stati latenti vanno "ancorati" al significato H/A/D (vincoli sulle emissioni);
  EM ha ottimi locali; restano iperparametri (durata minima, peso dello Smart Tagging).
- **Legame col §3:** le LF forniscono proprio le **ancore**: i frame su cui scatta una regola dura
  fissano lo stato, le altre LF diventano osservazioni.

### S3 · Ricostruzione self-supervised della palla
Nel 59% dei frame la palla è visibile: lì si ha la coppia *configurazione dei giocatori →
posizione della palla*. Si allena un modello (GBM o piccola rete) e lo si usa per stimare la
palla nel 41% mancante. Non è un classificatore H/A/D: è un **componente** che migliora S1/S2.

- **Usa:** ATD.
- **Pro:** attacca la difficoltà principale; ha **una metrica interna** (errore su frame visibili
  tenuti da parte) che non richiede etichette.
- **Contro:** la palla manca proprio nelle situazioni difficili (mischie, palle alte), dove la
  ricostruzione è meno affidabile → distribuzione diversa tra train e uso.

### S4 · Weak supervision / self-training con pseudo-etichette
"Funzioni di etichettatura" automatiche che si esprimono solo dove sono sicure (palla fuori campo
→ D; giocatore attaccato alla palla > 1 s → sua squadra; set piece → D). Si addestra un
classificatore sui frame coperti e lo si applica agli altri.

- **Usa:** ATD, Smart Tagging.
- **Pro:** sfrutta segnali forti, generalizza ai frame ambigui; consentito (nessuna persona decide).
- **Contro:** il classificatore impara anche gli errori delle regole (confirmation bias); il
  guadagno non è misurabile senza un metro esterno.
- **Stato:** **adottata** come struttura (§3). Il classificatore addestrato sulle pseudo-etichette
  è un passo opzionale successivo, da valutare dopo il voto pesato / label model.

### S5 · Supervisionato su altre partite, applicato a BAR-ATH
Partite open con tracking + eventi (es. Metrica sample-data: 3 partite, coordinate 0–1, la Game 3
nello stesso formato EPTS dell'ATD) → etichette H/A/D dagli eventi → tracking degradato per
somigliare all'ATD → modello addestrato lì e applicato qui.

- **Usa:** ATD solo per misurare le statistiche di degradazione e come input finale.
- **Pro:** etichette vere, risultato misurabile.
- **Contro:** domain shift (tracking corretto vs automatico, partite diverse); la definizione di
  possesso degli eventi può differire da quella della GT; **niente Smart Tagging** sulle partite
  open; lavoro di costruzione.

### S6 · Video: detector della palla pre-addestrato
Un detector addestrato su altre partite applicato al video tattico, con l'omografia per riportare
la palla sul campo. Recupera parte del 41% mancante.

- **Usa:** video tattico + omografia.
- **Pro:** l'informazione mancante è nel video; consentito.
- **Contro:** pesante (GPU, 199k frame a 1080p), fuori dal cuore del problema. Da valutare solo
  se S2/S3 si fermano per colpa della palla.

### S7 · Fusione con il broadcast
Allineare il broadcast al tattico e usarne l'ATD dove il tattico perde la palla.

- **Contro:** nessuna mappatura fornita, tagli e replay, omografia valida al 65,8%. Ultima priorità.

---

## 5. Asse 2 — come scegliamo tra due versioni

### M0 · Nessun metro esterno: parametri a priori + plausibilità
Parametri con significato calcistico, fissati prima di vedere i risultati e dichiarati nel
documento tecnico ("un possesso dura almeno 1 s"). Controlli di plausibilità: numero e durata
dei possessi rispetto alla letteratura, accordo con lo Smart Tagging.

- **Costo:** zero. **Rischio:** alto — EVALUATION.md mostra che sbagliare la sola finestra di
  smoothing della baseline sposta spellF1 di parecchi punti.

### M1 · Validation set su altre partite
Come in S5, ma usato **solo per misurare**, anche per modelli unsupervised (si rifà il fit EM su
ogni partita di validazione, così si valuta il *metodo*).

- **Verifica del metro:** EVALUATION.md pubblica gli spellF1 ufficiali delle 4 baseline sulla
  partita intera (smooth50 0.2433 > smooth125 0.2242 > smooth25 0.1996 > raw 0.0486). Se il
  validation set degradato riproduce **lo stesso ordine**, è un buon sostituto.
- **Uso:** una partita esclusa alla volta, più seed di degradazione, `--bootstrap 1000`.
- **Costo:** medio. **Limite:** misura bene il confronto tra varianti, non il punteggio assoluto;
  la parte Smart Tagging resta poco misurabile.
- **Per le LF del §3:** misura la precisione delle regole solo-ATD (D1–D3, D5, P1–P3, P6, P7).

### M2 · Metriche interne su BAR-ATH, senza etichette
Ogni componente si misura per conto suo:
- ricostruzione della palla (S3): errore su frame con palla visibile nascosti di proposito;
- dead ball: accordo con i set piece dello Smart Tagging e con la palla fuori dalle coordinate 0–1;
- modello a stati: stabilità tra seed/inizializzazioni EM, numero e durata dei possessi;
- **labeling functions (§3.6):** copertura, conflitti, accordo con lo Smart Tagging.

- **Costo:** basso. **Limite:** misura i pezzi, non il risultato finale H/A/D.

### Sempre: GT pubblica come controllo
Prima di ogni consegna: `validate_submission.py` + verifica che allineamento e significato delle
etichette tornino sui 5 minuti. Mai per scegliere.

---

## 6. Confronto

| Strada | Dati BAR-ATH usati | Impegno | Misurabile con | Spiegabilità | Note |
|---|---|---|---|---|---|
| **S1 + S4 regole come LF (§3)** | ATD, ST | medio | M0, M1 (solo LF ATD), M2 | alta | **ipotesi di lavoro adottata** |
| S1 euristiche | ATD, ST | basso | M0, M1 | alta | confluita nel §3 |
| S2 HMM/HSMM + EM | ATD, ST | medio | M1, M2 | alta | struttura a segmenti = spellF1; si appoggia sulle LF |
| S3 palla self-supervised | ATD | medio | **M2** (interna) | media | componente per S1/S2/S4 |
| S4 pseudo-etichette | ATD, ST | medio | M1 | media | confluita nel §3; rischio di imparare i propri errori |
| S5 supervisionato esterno | ATD (input) | alto | M1 | media | domain shift, niente ST |
| S6 video | video, omografia | molto alto | M1, M2 | media | solo se la palla è il collo di bottiglia |
| S7 broadcast | broadcast | alto | M2 | bassa | ultima |

---

## 7. Combinazioni realistiche

**Percorso A — solo BAR-ATH (§3 regole/LF → S2, + S3; misurato con M0 + M2)**
Nessun dato esterno. Si parte subito: preprocessing, feature, catalogo di LF, aggregazione,
livellatore temporale. Le scelte di alto livello si basano su parametri a priori e metriche interne.
Rischio: decidere alla cieca le scelte che contano di più (durata minima, penalità di transizione,
pesi delle LF).

**Percorso B — stesso modello, più validation set (come A, misurato con M1 + M2)**
Il modello resta a regole/unsupervised su BAR-ATH; il validation set serve solo per scegliere
(soglie delle LF solo-ATD, smoothing) e per dimostrare nel documento tecnico che il metodo batte la
baseline.

**Percorso C — supervisionato esterno (S5 [+ S3], misurato con M1)**
Massima misurabilità, massimo rischio di domain shift.

### Proposta da discutere
Partire con il **percorso A nella forma del §3** (sblocca subito tutto il team: ognuno può
scrivere e misurare LF in parallelo) e costruire **in parallelo, con una persona, un M1 leggero**
(Metrica sample-data + degradazione). A un checkpoint fissato — ad es. **1 novembre** — si decide:
- M1 riproduce l'ordine ufficiale delle baseline → si passa al percorso B;
- non lo riproduce → si resta su A e il tempo va su S3/S2.

Primi passi concreti per il percorso A:
1. Script di preprocessing (direzione d'attacco per tempo, palla pulita, velocità derivate, unità).
2. Script di verifica dell'ipotesi "fasi ST a coppie" (§3.3).
3. Prime LF ad alta affidabilità: D1, D3, D4, P1, P3, P4.
4. Tabella copertura/conflitti per LF (§3.6).
5. Aggregazione + isteresi → prima submission a regole, validata con `validate_submission.py`.

Scadenza consegna: **17 novembre 2026**.

---

## 8. Decisioni da prendere come team

1. Percorso A, B o C (o la proposta ibrida)? *Ipotesi attuale: A nella forma del §3, con M1 in parallelo.*
2. Chi lavora su cosa (preprocessing/cleaning, feature, LF dead ball, LF possesso, Smart Tagging,
   palla mancante, modello temporale, validazione, documento tecnico)?
3. Data del checkpoint e criterio per passare da A a B.
4. Soglie delle LF: quali fissare a priori e quali ricavare dalle distribuzioni (§3.5)?
5. Aggregazione: voto pesato semplice o label model? Classificatore sulle pseudo-etichette sì/no?
6. Politica `X` sui buchi lunghi della palla: astenersi o trascinare l'ultima squadra?
