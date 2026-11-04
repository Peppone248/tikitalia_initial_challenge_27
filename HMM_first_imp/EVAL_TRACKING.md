# Tracking della valutazione pubblica — v0 → v1 → C

Generato da `scripts/track_public_eval.py` il 2026-10-30 · ground truth `BAR-ATH_possession_gt_public.csv` · scorer ufficiale `evaluation/possession_metrics.evaluate` (lo stesso di `run_evaluation.py --bootstrap 1000`).

```powershell
.\venv\Scripts\python.exe HMM_first_imp\scripts\track_public_eval.py
```
Per aggiungere una versione: una voce in `config/eval_tracking.yaml`, poi rilancia.

> **Come leggerlo.** La finestra pubblica è 5 minuti: 7500 frame giudicati, **17 possessi veri** di 14.2 s in media. Differenze di spellF1 sotto **~0.36** (l'intervallo bootstrap più largo) possono essere fortuna. Serve a controllare che l'output abbia senso, **non a scegliere i parametri**: quello si fa su IDSSE (ultima colonna della tabella 1).

## 1. Il quadro in una tabella

| versione | spellF1 [5–95%] | macroF1 [5–95%] | possessi (pred / veri) | durata media s (pred / vera) | frammentazione (1 = giusta) | spellF1 IDSSE (7 partite) |
|---|---|---|---|---|---|---|
| v0 | 0.432 [0.284–0.579] | 0.773 [0.594–0.828] | 21 / 17 | 12.2 / 14.2 | 1.43 | 0.289 |
| v1 | 0.345 [0.166–0.524] | 0.740 [0.523–0.825] | 12 / 17 | 21.2 / 14.2 | 0.76 | 0.256 |
| c_v2 | 0.459 [0.294–0.624] | 0.802 [0.585–0.872] | 17 / 17 | 15.0 / 14.2 | 0.95 | 0.316 |
| **c_v3** | 0.512 [0.325–0.688] | 0.830 [0.593–0.896] | 15 / 17 | 16.6 / 14.2 | 0.86 | 0.319 |
| baseline_smooth125 | 0.424 [0.267–0.594] | 0.581 [0.448–0.665] | 16 / 17 | 18.5 / 14.2 | 0.81 | – |
| baseline_smooth50 | 0.326 [0.214–0.447] | 0.621 [0.460–0.742] | 26 / 17 | 10.8 / 14.2 | 1.43 | – |

- **spellF1** (classifica): possessi indovinati come pezzi interi (inizio, fine, squadra).
- **macroF1**: etichetta giusta frame per frame, media di H, A, D.
- **Frammentazione** = segmenti previsti / veri: sopra 1 spezza, sotto 1 unisce.

## 2. Palla morta (D)

| versione | F1 D | precision (D detta e giusta) | recall (D vera trovata) | D vera chiamata H/A | palla viva chiamata D | F1 viva/morta |
|---|---|---|---|---|---|---|
| v0 | 0.639 | 76% | 55% | 658 (26 s) | 258 (10 s) | 0.926 |
| v1 | 0.741 | 85% | 66% | 504 (20 s) | 170 (7 s) | 0.946 |
| c_v2 | 0.746 | 86% | 66% | 504 (20 s) | 153 (6 s) | 0.947 |
| c_v3 | 0.812 | 87% | 76% | 350 (14 s) | 169 (7 s) | 0.958 |
| baseline_smooth125 | 0.146 | 100% | 8% | 1353 (54 s) | 0 (0 s) | 0.899 |
| baseline_smooth50 | 0.323 | 66% | 21% | 1155 (46 s) | 164 (7 s) | 0.899 |

D vera nella finestra: 1469 frame (59 s). La **recall** dice quanta palla morta troviamo; la **precision** quanto ci si può fidare quando diciamo D.

## 3. Squadra (H / A)

| versione | F1 H | F1 A | squadra giusta a palla viva | H chiamato A | A chiamato H | quota possesso casa (pred vs vera) |
|---|---|---|---|---|---|---|
| v0 | 0.879 | 0.799 | 87% | 383 (15 s) | 136 (5 s) | 53% vs 57% (err 0.045) |
| v1 | 0.754 | 0.726 | 76% | 904 (36 s) | 365 (15 s) | 48% vs 57% (err 0.089) |
| c_v2 | 0.850 | 0.810 | 86% | 343 (14 s) | 364 (15 s) | 57% vs 57% (err 0.000) |
| c_v3 | 0.864 | 0.813 | 85% | 392 (16 s) | 319 (13 s) | 55% vs 57% (err 0.027) |
| baseline_smooth125 | 0.786 | 0.811 | 89% | 282 (11 s) | 409 (16 s) | 62% vs 57% (err 0.052) |
| baseline_smooth50 | 0.778 | 0.762 | 83% | 413 (17 s) | 419 (17 s) | 62% vs 57% (err 0.048) |

## 4. Possessi interi (spellF1 per soglia di sovrapposizione)

| versione | IoU 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | @0.5 presi / mancati / inventati |
|---|---|---|---|---|---|---|
| v0 | 0.632 | 0.474 | 0.474 | 0.316 | 0.263 | 9 / 8 / 12 |
| v1 | 0.483 | 0.414 | 0.414 | 0.276 | 0.138 | 6 / 11 / 6 |
| c_v2 | 0.588 | 0.588 | 0.529 | 0.353 | 0.235 | 9 / 8 / 8 |
| c_v3 | 0.625 | 0.625 | 0.562 | 0.437 | 0.312 | 9 / 8 / 6 |
| baseline_smooth125 | 0.606 | 0.606 | 0.424 | 0.242 | 0.242 | 7 / 10 / 9 |
| baseline_smooth50 | 0.512 | 0.465 | 0.326 | 0.233 | 0.093 | 7 / 10 / 19 |

IoU = quanto il possesso previsto si sovrappone a quello vero. Se lo spellF1 cala molto da 0.3 a 0.7 i possessi sono quelli giusti ma con i **confini imprecisi**.

## 5. Momenti di cambio

| versione | cambi pred / veri | F1 ±0.5 s (presi) | F1 ±1 s (presi) | ±1 s mancati / inventati | latenza media / mediana (frame; < 0 = in anticipo) |
|---|---|---|---|---|---|
| v0 | 29 / 20 | 0.082 (2) | 0.286 (7) | 13 / 22 | -10.9 / -23.0 |
| v1 | 15 / 20 | 0.171 (3) | 0.171 (3) | 17 / 12 | 5.3 / 7.0 |
| c_v2 | 19 / 20 | 0.154 (3) | 0.205 (4) | 16 / 15 | -4.5 / -6.5 |
| c_v3 | 17 / 20 | 0.162 (3) | 0.270 (5) | 15 / 12 | 0.0 / -3.0 |
| baseline_smooth125 | 16 / 20 | 0.111 (2) | 0.167 (3) | 17 / 13 | -7.7 / -3.0 |
| baseline_smooth50 | 29 / 20 | 0.082 (2) | 0.082 (2) | 18 / 27 | 4.0 / 4.0 |

## 6. Partita intera (BAR-ATH, nessuna ground truth)

Dal `run_summary.json` di ogni run: controlli di plausibilità, non punteggi. Riferimento della finestra pubblica: 3.4 possessi/min, palla morta 20% dei frame.

| versione | H / A / D % | possessi | durata poss. media / mediana / p90 s | interruzioni | durata D media / mediana / p90 s | possessi < 1 / 3 / 5 s | flip H-A-H < 5 s | cambi H↔A a palla viva | stessa squadra spezzata da D < 2 s |
|---|---|---|---|---|---|---|---|---|---|
| v0 | 46.3 / 32.4 / 21.3 | 329 (3.1/min) | 15.2 / 8.9 / 35.1 | 145 | 9.3 / 5.6 / 14.8 | 17 / 92 / 122 | 72 | 184 | 0 |
| v1 | 44.7 / 28.8 / 26.5 | 205 (1.9/min) | 22.8 / 17.0 / 49.2 | 124 | 13.6 / 7.8 / 25.4 | 0 / 19 / 32 | 11 | 81 | 0 |
| c_v2 | 45.6 / 27.9 / 26.5 | 244 (2.3/min) | 19.1 / 12.7 / 43.4 | 122 | 13.8 / 8.2 / 25.5 | 6 / 29 / 50 | 22 | 121 | 0 |
| c_v3 | 43.5 / 23.7 / 32.8 | 228 (2.2/min) | 18.7 / 13.3 / 43.0 | 124 | 16.8 / 11.6 / 35.7 | 8 / 31 / 48 | 18 | 104 | 0 |

## 7. Cosa cambia da una versione alla successiva

| metrica | v0 → v1 | v1 → c_v2 | c_v2 → c_v3 |
|---|---|---|---|
| spellF1 | -0.087 | +0.114 | +0.054 |
| macroF1 | -0.032 | +0.062 | +0.027 |
| F1 D | +0.102 | +0.005 | +0.066 |
| recall D | +0.105 | +0.000 | +0.105 |
| squadra a palla viva | -0.110 | +0.096 | -0.003 |
| spellF1 @IoU 0.7 | -0.125 | +0.097 | +0.077 |
| F1 cambi ±1 s | -0.114 | +0.034 | +0.065 |
| possessi (veri 17) | 21 → 12 | 12 → 17 | 17 → 15 |
| spellF1 IDSSE (appaiato) | -0.033 | +0.060 | +0.016 |

Una variazione pubblica sotto ~0.36 di spellF1 non è una prova da sola: va letta insieme alla riga IDSSE.

## 8. Versioni e file

| versione | data | cosa | CSV | SHA-256 |
|---|---|---|---|---|
| v0 | 2026-10-29 | HMM di Bakis per frame, NN (MLP) + regole, config_v0 | `outputs/run_v0/tikitalia.csv` | `a82ff7f159db…` |
| v1 | 2026-10-30 | v0 + team gate, NN team-blind (random-feature), D-WHISTLE anchor, turnover rules | `outputs/run/tikitalia.csv` | `9f1797edaf6c…` |
| c_v2 | 2026-10-30 | approccio C: D dall'HMM v1, confini solo agli eventi, squadra per segmento (lambda 0.5) | `outputs/run_c/tikitalia.csv` | `23a6ad7c1f9c…` |
| c_v3 | 2026-10-30 | C + D da Smart Tagging (calci piazzati), D-SP spenta, finestra portiere, meno pesi (IDSSE con tagging simulato: v2 0.303 nelle stesse condizioni) | `outputs/run_c3/tikitalia.csv` | `6a7baedd1245…` |

## 9. Lettura (scritta a mano, 30/10/2026)

**v0 → v1: palla morta migliore, possessi peggiori.**
- La palla morta migliora davvero: F1 D da 0.64 a 0.74, recall da 55% a 66%, D mancata da 26 s a 20 s.
  Sulla partita intera le interruzioni passano da 145 molto corte (mediana 5.6 s) a 124 più lunghe
  (mediana 7.8 s).
- La squadra peggiora: a palla viva da 87% a 76%, con 36 s di H chiamati A. La quota casa scende
  al 48% contro un vero 57%.
- v1 unisce troppo: 12 possessi contro 17, durata 21 s contro 14 s. Sulla partita intera i
  possessi passano da 329 a 205.
- Risultato: spellF1 −0.09 sulla finestra, −0.03 su IDSSE (v0 migliore su 7/7 partite).

**v1 → C (v2): stessa palla morta, squadra e struttura recuperate.**
- La palla morta è quella di v1 (C la prende dall'HMM): recall identica (66%), stessi 504 frame
  mancati. Cambia poco anche sulla partita intera (26.5% di D, 122 interruzioni).
- La squadra torna al livello di v0 (86% a palla viva) e la quota casa diventa esatta (57%).
- I possessi tornano al numero giusto: 17 su 17, durata 15.0 s, frammentazione 0.95. Sulla
  partita intera 244 possessi, a metà tra v0 (329, troppo spezzettato: 92 sotto i 3 s) e v1 (205).
- Risultato: spellF1 +0.11 sulla finestra, +0.06 su IDSSE; +0.027 su v0 in IDSSE (7/7).

**Cosa resta uguale in tutte le versioni (i limiti attuali).**
1. *Palla morta troppo corta.* Anche la migliore trova il 66% della D vera (20 s mancati su 59 s).
   Sulla partita intera la D è al 26.5%, mentre di solito la palla è ferma per il 35–45% del tempo.
2. *Confini imprecisi.* Il numero di cambi è giusto (19 su 20), ma entro ±1 s ne prendiamo 4.
   Lo spellF1 scende da 0.59 (IoU 0.3) a 0.24 (IoU 0.7).
3. *Sulla partita intera probabilmente troppi pochi possessi.* 2.3/min contro i 3.4/min della
   finestra pubblica (stima su soli 5 minuti); su IDSSE 1302 contro 2705.

**Prossimi passi** (da misurare su IDSSE, non qui): allungare le interruzioni fino alla ripresa
vera e anticiparne l'inizio; aggiungere confini dove la palla non si vede (inversione di direzione
di entrambe le squadre).

**C (v2) → C (v3): palla morta da Smart Tagging, portiere, meno pesi.**
- Palla morta: recall da 66% a 76% nella finestra, D mancata da 20 s a 14 s; sulla partita intera
  la D passa dal 26.5% al 32.8% e le interruzioni durano di più (mediana da 8.2 a 11.6 s).
- Squadra e struttura quasi invariate: 15 possessi su 17 (v2: 17), quota casa 55% contro 57%.
- Su IDSSE (tagging simulato, stesse condizioni): +0.016 spellF1 sullo sporco (6/7), 0.000 sul
  pulito. Dettagli in REPORT_C_v3.md.
