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
