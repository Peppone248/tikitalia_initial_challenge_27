# Modello di sequenza bidirezionale per i cambi di possesso — design

*12/11/2026 · autore: Francesco (con Claude) · stato: da approvare*

## 1. Perché

Il modello del team (`combined_algorithm`: pulizia del tracking → alberi → HMM) arriva a 0,66 di
spellF1 sulle partite esterne. L'oracolo di REPORT_B §20 dice dove sono i punti rimasti: **i cambi di
possesso che il modello non trova** (+0,21 al massimo), più che la loro data (+0,02) o la palla
morta. Il modello attuale prevede circa 303 possessi a partita contro 386 veri: è **sotto-segmentato**.

L'idea di Francesco: il possesso ha una memoria fortissima. Quindi la memoria va messa nella
**struttura** (l'HMM, che c'è già), e serve un modello che guardi **avanti e indietro** nel tempo
(il controllo spesso si capisce *dopo* il tocco) e che dia un segnale esplicito su **dove il possesso
cambia**. È la ricetta della letteratura di *temporal action segmentation* (MS-TCN, ASRF), che usa
proprio la metrica dello spellF1: F1 sui segmenti con soglie di IoU.

## 2. Obiettivi, non-obiettivi, anti-obiettivi

**Obiettivi**
- Un modello temporale **bidirezionale** che, per ogni frame, dà:
  - `P(H)`, `P(A)`, `P(D)`;
  - `P(cambio)`: la probabilità che entro ±k frame ci sia un cambio di possesso H↔A.
- Un decodificatore HMM in cui `P(cambio)` rende i cambi H↔A più probabili dove il modello li vede.
- Una misura onesta su partite esterne: aiuta o no, rispetto allo stesso pipeline senza modello di
  sequenza?
- Le probabilità salvate in un formato documentato, così Beppe può provarle dentro
  `combined_algorithm`.

**Non-obiettivi**
- Sostituire `combined_algorithm`. Se il modello aiuta, entra come **ingrediente** (emissione o
  feature), non come pipeline nuova.
- Prevedere la posizione della palla. Ball Radar sbaglia in media di circa 3,7 m, più del raggio di
  controllo di 1,5 m. Si prevede direttamente il possesso.
- Usare il possesso passato come input. Il modello imparerebbe a copiarlo, e in uso reale gli errori
  si trascinerebbero avanti.

**Anti-obiettivi** (un fallimento anche se "funziona")
- Qualsiasi scelta presa guardando BAR-ATH o i 5 minuti pubblici (RULES.md §1–2).
- Un CSV che non si rigenera identico su un'altra macchina (RULES.md §3).
- Arrivare al 16 senza consegna perché si è inseguito questo modello. È **a tempo**: vedi §8.

## 3. Vincoli

- **Calendario:** oggi è il 12, la consegna chiude il 16 sera.
- **Etichette:** solo da altre partite, documentate nel PDF tecnico (cosa, come, da chi, quanto).
- **Riproducibilità:** l'addestramento può usare PyTorch. L'**inferenza** che produce il CSV va in
  numpy, con input e uscite arrotondati su una griglia, come già fa il team (`exact.py`). I pesi si
  salvano nella repo, sono piccoli.
- **Dati di gara sul PC di Francesco:** ancora assenti. Il codice si scrive e si testa su dati
  sintetici e su IDSSE, e va provato su BAR-ATH quando arrivano i dati.
- **Codice di `combined_algorithm`:** non è su git. Il modulo nuovo deve stare in piedi da solo.
  L'integrazione la prova Beppe.

## 4. Approcci considerati

| Approccio | Pro | Contro | Esito |
|---|---|---|---|
| **A. Autoregressivo causale**: predice il possesso a t dal possesso a t−1 e dalle feature passate | è l'idea di partenza, semplice | impara a copiare il frame prima; non usa il futuro; gli errori si propagano | scartato |
| **B. TCN bidirezionale con due uscite (classi + cambio), decodificato da un HMM** | usa passato e futuro; attacca i cambi mancati; si innesta sull'HMM esistente; inferenza facile in numpy | serve un set di dati esterni degradati come BAR-ATH | **scelto** |
| C. GRU / LSTM bidirezionale | potente sulle sequenze | inferenza deterministica in numpy più delicata; più lento | riserva, se la TCN non basta |
| D. Solo HSMM con distribuzioni di durata | piccolo, nessun addestramento profondo | non porta segnale nuovo sui cambi | fuori da questo lavoro |

**Scelto B** (Francesco ha approvato il 12/11 la versione: memoria nell'HMM, modello bidirezionale
che cerca i cambi, più partite esterne).

## 5. Design

### 5.1 Dove sta

Una nuova cartella di primo livello **`sequence_model/`**, come `ball_detection/` di Gio,
indipendente dal codice non pubblicato. Dalla repo riusa:
- lo scorer ufficiale `evaluation/possession_metrics.py`;
- `HMM_first_imp/validation/fetch_idsse.py` dal branch `approach-b-broadcast`, copiato, per
  scaricare e convertire IDSSE.

```text
sequence_model/
  README.md
  requirements.txt          numpy, pandas, pyyaml, torch (solo addestramento)
  config.yaml               ogni costante con il suo perché
  data/                     (ignorato da git) partite convertite in .npz
  seqmodel/
    datasets.py             caricamento IDSSE / Metrica / SkillCorner -> formato comune
    degrade.py              tracking "come BAR-ATH": buchi di palla, tracce spezzate
    features.py             feature per frame, simmetriche rispetto allo scambio H/A
    model.py                TCN bidirezionale (PyTorch) + export dei pesi
    infer.py                stessa rete in numpy, deterministica
    decode.py               HMM a 3 stati con durate minime e cambio guidato da P(cambio)
    evaluate.py             leave-one-match-out, scorer ufficiale, più seed
  scripts/
    fetch_data.py · train.py · run_lomo.py · predict.py
  tests/
```

### 5.2 Dati (in quest'ordine)

1. **IDSSE**, 7 partite: possesso e stato della palla per frame a 25 fps, già usate dal team. È la
   fase 1.
2. **Metrica**, 3 partite di esempio: le etichette vanno ricavate dagli eventi con un algoritmo. È la
   fase 2.
3. **SkillCorner open data**, 9 partite: gruppo in possesso per frame, ma tracking da broadcast a
   10 fps. Fase 3, solo se c'è tempo.

Formato comune per partita: `px, py` (frame × giocatori, NaN dove manca), `team` per giocatore,
`ball_x, ball_y` (NaN dove non vista), `period`, `label` ∈ {H, A, D, X}, dimensioni del campo,
direzione d'attacco per tempo (ricavata dalle posizioni: dove stanno i portieri).

Etichette di addestramento e di misura: quelle **"controllate"** del team (REPORT_VALIDATION):
un cambio più corto di 1 s tra due possessi dell'altra squadra è un tocco, non un cambio
(CHALLENGE.md). Così i numeri restano confrontabili con quelli dei report.

### 5.3 Degrado "come BAR-ATH"

Senza degrado il modello imparerebbe da una palla sempre visibile, mentre su BAR-ATH manca nel 41%
dei frame. Due regole a priori, con valori tondi, gli stessi principi dei report di Beppe:
- **la palla sparisce soprattutto a gioco fermo:** ogni tratto di palla morta è nascosto con
  probabilità alta (sd8: 0,8); a palla viva si tolgono buchi con lunghezze da una distribuzione a
  priori, così la palla risulta vista circa nel 55–60% dei frame valutati;
- **tracce dei giocatori spezzate:** qualche % di frame mancanti a tratti.

Almeno **due seed** di degrado: REPORT_B §19.5 mostra che un solo seed non decide differenze
sotto ~0,005.

**La palla di Gio non entra nella fase 1.** REPORT_B §27: dare un rilevatore a un modello non
addestrato su quell'input peggiora il punteggio. Sarà una variante successiva, con un degrado che
simula il rilevatore.

### 5.4 Feature per frame

Tutte cambiano segno se si scambiano H e A, oppure sono neutre (palla vista, velocità della palla).
Circa 25 numeri:
- **palla:** vista sì/no, secondi da quando non si vede, velocità, distanza dalla linea più vicina,
  x nella direzione d'attacco di H;
- **vicinanza alla palla, H meno A:** distanza del giocatore più vicino, giocatori entro 3 / 8 m;
- **forma delle squadre, H meno A:** baricentro, linea difensiva, giocatore più avanzato, ampiezza,
  lunghezza, velocità in avanti, velocità media;
- **ritmo collettivo:** velocità media di tutti (rileva le interruzioni senza la palla);
- **maschera D semplice:** palla vista fuori dalle linee; tutti lenti con la palla non vista.

Le transizioni ("dopo meno prima") non servono come feature: la TCN bidirezionale le vede da sola.

### 5.5 Modello

- **TCN non causale:** convoluzioni dilatate sul tempo, che guardano avanti e indietro. Campo
  ricettivo di circa ±10 s a 25 fps. Pochi canali (32–64) e 6–8 livelli.
- **Due uscite:**
  - classi H / A / D, softmax;
  - cambio H↔A entro ±k frame (k ≈ 12, mezzo secondo), sigmoide.
- **Simmetria H/A:** ogni sequenza si vede anche con le squadre scambiate. L'uscita si rende esatta
  mediando f(x) e swap(f(swap(x))), così il modello non può imparare "la casa ha più palla".
- **Addestramento:** PyTorch, seed fissi, finestre di circa 60 s, loss pesata sulla rarità dei
  cambi, early stop sulla partita di validazione interna.
- **Inferenza:** gli stessi pesi in numpy, con convoluzioni scritte a mano, input arrotondati a 2^-16
  e uscite arrotondate a 3 decimali, come fa il team.

### 5.6 Decodifica

Un HMM a 3 stati (H, A, D) con catene di Bakis per le durate minime: 0,8 s per il possesso, 2 s per
la palla morta, come nel modello del team.
- **Emissioni:** w · (log P(classe) − log prior).
- **Transizioni H↔A variabili nel tempo:** probabilità base × (1 + β · P(cambio)). β e w si scelgono
  in leave-one-match-out.
- **Altre transizioni:** contate sulle partite di addestramento.

Il meccanismo delle transizioni variabili esiste già nel v1 del team, quindi Beppe lo conosce.

### 5.7 Misura (`run_lomo.py`)

Leave-one-match-out su IDSSE: si addestra su 6, si valuta sulla settima, a turno. Scorer
ufficiale, spellF1 e macroF1, due seed di degrado. Confronti, tutti sugli stessi dati:

| Riga | Cosa misura |
|---|---|
| baseline50 | il riferimento ufficiale (≈ 0,21 sui dati degradati del team) |
| classificatore per frame (stesse feature, nessuna sequenza) + stesso HMM | **il termine di paragone vero**: isola il valore del modello di sequenza |
| TCN, solo classi + HMM | il valore del guardare avanti e indietro |
| TCN, classi + cambio + HMM | il valore del ramo dei cambi |

Oltre allo spellF1 si guardano: numero di possessi previsti contro veri, cambi trovati entro ±1 s
(`transF1@25`), quota di palla morta.

**Criterio per tenerlo:** la TCN completa batte il classificatore per frame di **almeno 0,01 di
spellF1**, su almeno 5 partite su 7, con **entrambi** i seed. Altrimenti si dichiara un risultato
negativo e lo si scrive nel registro.

Il confronto con `combined_algorithm` (0,659) **non è diretto**: il degrado è diverso. Per questo il
modulo esporta le probabilità e Beppe le prova nella sua pipeline sullo stesso benchmark.

### 5.8 Uscite

- `predict.py --match <npz o ATD>` → `probs.npz` con `p_h, p_a, p_d, p_change` per frame, più un
  CSV `frame,label` dopo la decodifica.
- Per BAR-ATH: lettura dell'ATD con lo stesso caricatore del team (`tikitalia`), ed esportazione di
  `probs.npz` per Beppe.

## 6. Conformità (va nel documento tecnico)

- **Etichette:** solo IDSSE (DFL, CC-BY 4.0), Metrica (dati di esempio pubblici, etichette da
  eventi con un algoritmo) e, se usata, SkillCorner open data. Nessuna etichetta di BAR-ATH.
- **Degrado:** costanti tonde a priori, non stimate da BAR-ATH, perché la domanda 4 agli
  organizzatori è ancora aperta.
- **Nessun numero di frame o di slot nel codice.**
- **I 5 minuti pubblici:** una sola lettura finale, come controllo della struttura.

## 7. Cosa decide chi implementa

Numero esatto di canali e livelli; schema dell'early stop; valori di griglia per β e w; dettagli del
caricatore Metrica. Ogni scelta va nel `config.yaml` con il suo perché.

## 8. Tempi (time-box)

| Giorno | Obiettivo |
|---|---|
| 12–13/11 | IDSSE scaricato; formato comune; degrado; feature; classificatore per frame + HMM in leave-one-match-out |
| 13–14/11 | TCN + inferenza numpy + ramo dei cambi; tabella completa con 2 seed |
| 14/11 sera | **decisione**: si tiene o no (criterio §5.7). Se no, ci si ferma e si aiuta la consegna |
| 15/11 | se si tiene: `probs.npz` su BAR-ATH, prova di Beppe in `combined_algorithm` |
| 16/11 | solo correzioni sicure; riga nel documento tecnico |

## 9. Scelte fatte da Claude (da controllare)

- Cartella separata `sequence_model/` invece che dentro `src/tikitalia/`: non dipende dal codice non
  pubblicato e non tocca il codice di nessuno.
- TCN invece di GRU: inferenza deterministica in numpy molto più semplice.
- La palla di Gio esclusa dalla fase 1 (REPORT_B §27).
- Soglia di 0,01 di spellF1 per tenerlo: sopra il rumore di un seed, sotto il guadagno tipico delle
  idee che hanno funzionato (+0,05 per le transizioni).
- Il documento non è committato: lo si committa quando Francesco lo chiede.
