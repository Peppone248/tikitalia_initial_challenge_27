# Esempi: come si valuta una regola

Esempi passo per passo per i termini di `NOMENCLATURA.md`. **Tutti i numeri sono illustrativi**:
nessuno viene da misure sui dati della gara.

---

## Esempio 1 · Forza di un'attivazione (P-DRIB)

Al frame `f` la regola P-DRIB ("un giocatore conduce la palla") scatta per il numero 7 del
Barcellona ed emette `ℓ_r(f) = H`.

### Condizioni

| | Condizione | Tipo | Soglia | Valore al frame f |
|---|---|---|---|---|
| C1 | distanza palla–giocatore ≤ `R_CONTROL` | istantanea | 1,5 m | 0,8 m |
| C2 | velocità palla ≤ `V_MAX` | istantanea | 8 m/s | 3 m/s |
| C3 | avversario più vicino ≥ `D_PRESS` | istantanea | 1,0 m | 1,4 m |
| C4 | C1, C2 e C3 insieme per ≥ `T` | di durata | 0,4 s (10 frame) | 0,56 s (14 frame) |

### Rumore

- σ_palla = 0,20 m, σ_giocatore = 0,15 m (residuo tra traccia grezza e smussata)
- distanza tra due oggetti: σ = √(0,20² + 0,15²) = **0,25 m** (vale per C1 e C3)
- velocità della palla (derivata, più rumorosa): **σ_v = 1 m/s**

### Calcolo

| | Margine | z = margine / σ | clip(z / 3) | d | s_i |
|---|---|---|---|---|---|
| C1 | 1,5 − 0,8 = 0,7 m | 2,8 | 0,93 | 1 | 0,93 |
| C2 | 8 − 3 = 5 m/s | 5,0 | 1,00 (saturata) | 1 | 1,00 |
| C3 | 1,4 − 1,0 = 0,4 m | 1,6 | 0,53 | 1 | 0,53 |
| C4 | (blocco C1–C3) | min = 1,6 | 0,53 | 0,56 / 0,8 = 0,7 | **0,37** |

- C2 è **saturata**: z = 5 supera `Z_SAT = 3`, quindi vale 1 come se fosse z = 3.
- La **condizione più debole** è C4: il blocco è appena soddisfatto per via di C3 (avversario
  vicino) e dura da poco.

```text
s_r(f) = min(0,93; 1,00; 0,53; 0,37) = 0,37
w_r(f) = q_r · s_r(f) = 0,85 · 0,37 = 0,31
```

La regola vota `H` con peso 0,31. Il `q = 0,85` è quello calcolato nell'esempio 2.

### Stesso giocatore, mezzo secondo dopo

Si è allontanato dall'avversario (1,9 m) e il blocco dura da 0,9 s:

- C3: margine 0,9 m → z = 3,6 → s = 1 (saturata)
- C4: z = min(2,8; 5,0; 3,6) = 2,8 → 0,93 · min(1; 0,9 / 0,8) = 0,93
- s = 0,93 → w = 0,85 · 0,93 = **0,79**

---

## Esempio 2 · Qualità di una regola (P-DRIB)

### Senza dati

Nella scheda: `conf_prior = 0,80`, `n₀ = 30` episodi. Prima di qualsiasi misura `q = 0,80`.

### Con le partite M1 (etichette vere)

P-DRIB scatta in 200 episodi; in 172 l'etichetta emessa è quella vera.

```text
q = (0,80 · 30 + 172) / (30 + 200) = 196 / 230 = 0,85
```

La precisione osservata da sola sarebbe 172 / 200 = 0,86; il prior la tira appena verso 0,80
perché gli episodi sono molti.

### Confronto con una regola rara (R-WALL, barriera)

`conf_prior = 0,80`, `n₀ = 30`. Scatta in 8 riprese, tutte corrette.

```text
q = (0,80 · 30 + 8) / (30 + 8) = 32 / 38 = 0,84
```

Anche con il 100% di successi resta vicino al prior: 8 casi sono troppo pochi per fidarsi.
Ecco dove entra la frequenza: **non cambia la qualità, decide quanto ci fidiamo della stima.**

### Per frame invece che per episodio

Gli stessi 200 episodi sono 4000 frame, di cui 3389 corretti: precisione per frame 0,847.
Si riportano tutti e due i numeri; se divergono molto, la regola sbaglia soprattutto su pochi
episodi lunghi (o su molti brevi).

---

## Esempio 3 · Verifica di `w = q · s` (calibrazione)

Le 4000 attivazioni di P-DRIB su M1, divise in fasce di forza:

| fascia di s | attivazioni | corretti | precisione p |
|---|---|---|---|
| 0–0,25 | 600 | 372 | 0,62 |
| 0,25–0,5 | 900 | 702 | 0,78 |
| 0,5–0,75 | 1000 | 890 | 0,89 |
| 0,75–1 | 1500 | 1425 | 0,95 |
| **totale** | **4000** | **3389** | **0,85** |

### Peso ottimale per fascia

`w_ott = ln(2p / (1 − p))`, poi tutto normalizzato sull'ultima fascia:

| centro fascia s | p | w_ott | w_ott norm. | q·s norm. | con pavimento s₀ = 0,2 |
|---|---|---|---|---|---|
| 0,125 | 0,62 | 1,18 | **0,32** | **0,14** | 0,33 |
| 0,375 | 0,78 | 1,96 | 0,54 | 0,43 | 0,56 |
| 0,625 | 0,89 | 2,78 | 0,76 | 0,71 | 0,78 |
| 0,875 | 0,95 | 3,64 | 1,00 | 1,00 | 1,00 |

### Lettura

- Dalla fascia centrale in su il prodotto segue bene il peso ottimale.
- Nella fascia bassa lo sottostima: le attivazioni deboli hanno ragione il 62% delle volte (quasi
  il doppio del caso, 33%) ma con `q·s` quasi non votano.
- Con il pavimento `s' = 0,2 + 0,8·s` le due curve coincidono quasi ovunque.

### Caso negativo (P-SHAPE-OPEN)

| fascia di s | 0–0,25 | 0,25–0,5 | 0,5–0,75 | 0,75–1 |
|---|---|---|---|---|
| precisione | 0,58 | 0,60 | 0,59 | 0,61 |

La precisione non cresce con la forza: lo z-score di questa regola non porta informazione. Si
usa `s = 1` e il peso dipende solo da `q` (oltre al tetto 0,4 già previsto per la regola).

---

## Esempio 4 · Cambio di possesso

### Finestra di conferma (dentro il motore)

| Istante | Segnale | Controllo della nuova squadra | Esito |
|---|---|---|---|
| 30:12,0 | T-REV: la palla inverte verso la porta del Barcellona | giocatore dell'Athletic in controllo a 30:13,1 (dopo 1,1 s) | **accettato** |
| 31:40,0 | T-REV: la palla inverte verso la porta del Barcellona | nessun giocatore dell'Athletic in controllo entro 31:41,5; la riprende il Barcellona | **scartato** (era un retropassaggio) |

### Tolleranza di valutazione (solo per misurare l'errore, Δt = ±0,5 s)

Cambi veri (M1): 12:03,7 · 14:10,2 · 15:55,0
Cambi proposti dalla regola: 12:03,4 · 12:05,0 · 14:10,5 · 17:20,0

| Proposto | Cambio vero più vicino libero | Scarto | Esito |
|---|---|---|---|
| 12:03,4 | 12:03,7 | 0,3 s | corretto |
| 12:05,0 | — (12:03,7 già abbinato, e comunque a 1,3 s) | — | errore |
| 14:10,5 | 14:10,2 | 0,3 s | corretto |
| 17:20,0 | — | — | errore |

Il cambio vero delle 15:55,0 non è stato trovato.

```text
precisione = 2 / 4 = 0,50
richiamo   = 2 / 3 = 0,67
```

---

## Esempio 5 · Qualità senza etichette (consenso leave-one-out)

Sulla partita di gara non ci sono etichette. Per stimare P-DRIB:

1. si fa girare il motore **senza** P-DRIB e si tengono solo i frame con `conf₋ᵣ(f) ≥ 0,7`;
2. P-DRIB scatta in 3000 di questi frame e concorda col consenso in 2610;
3. `corretti = 2610`, `attivazioni = 3000` (in frame; lo stesso si fa per episodi) e si applica la
   formula con il prior dell'esempio 2.

Precisione stimata 2610 / 3000 = 0,87. Il numero è ottimista se P-DRIB sbaglia negli stessi punti
delle regole che condividono le sue feature (es. P-FREE): per questo le regole correlate vanno
dichiarate, e il label model le tratta a parte.
