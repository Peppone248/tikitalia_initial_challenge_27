# Improvements da discutere — prossima call

Proposte nate dalla lettura di `evaluation/possession_metrics.py`, `EVALUATION.md`, `RULES.md` e
`SUBMISSION.md`. Sono tutte **consentite dal regolamento**. Per ognuna c'è una domanda da chiudere
in call.

I numeri marcati *(sim)* vengono da una simulazione con il codice ufficiale della metrica su una
partita **sintetica**: sono solo ordini di grandezza, non misure su BAR-ATH.

---

## A. Dalla metrica: gratis o quasi

### A1. Mai scrivere `X` sui frame valutati

Riempire ogni `X` con l'ultima etichetta data, come fa già il valutatore prima di calcolare lo
spellF1.

- **spellF1:** identico, perché il valutatore farebbe lo stesso riempimento.
- **macroF1** (primo criterio di parità): può solo migliorare. Una `X` conta sempre come errore,
  un'etichetta copiata a volte è giusta.

> **Da decidere:** lo mettiamo come ultimo passo fisso della pipeline, prima di
> `validate_submission.py`?

### A2. Retrodatare i cambi: i ritardi costano più di tutto

Dalla tabella ufficiale in `EVALUATION.md`: tutto perfetto ma 8 frame di ritardo → 0,89;
1 s di ritardo → 0,65. Un frame di scarto è gratis.

Le nostre regole confermano un cambio *dopo* averlo visto (`T_CONTROL` = 1 s, finestra di
conferma `T_TURN_CONFIRM` = 1,5 s), quindi senza correzione il ritardo è sistematico.

- ogni cambio di possesso va spostato all'**inizio** dell'episodio di controllo che lo conferma;
- ogni D va fatto partire all'istante dell'uscita della palla o del rallentamento collettivo,
  non a quando la regola ne è sicura.

È una scelta di progetto, non una taratura sulla GT.

> **Da decidere:** che evento usiamo come istante del cambio (primo tocco? inizio del controllo?).
> Si lega alla Q8 di `REGOLE_PSEUDO_LABEL.md` §11.

### A3. Un cambio inventato e uno mancato costano quasi uguale

*(sim)* Circa 0,4 punti di spellF1 ciascuno. Un cambio falso aggiunge un possesso spurio e spezza
quello vero; uno mancato fonde due possessi in uno che non si abbina bene a nessuno dei due.

Conseguenza: **non conviene essere prudenti**. Un cambio candidato va accettato se, a livello di
segmento, è più probabile che sia vero che falso. Questo cambia la taratura della finestra di
conferma e del peso delle regole TURNOVER.

> **Da decidere:** soglia di accettazione dei cambi intorno al 50% a livello di segmento, invece
> di una soglia alta "di sicurezza"?

### A4. Il D conta solo come separatore

Lo spellF1 considera solo i possessi (sequenze di `H` o di `A`): l'interno di un D non entra nel
punteggio. Contano **la presenza** del D e **i suoi bordi**.

Il caso pericoloso sono i D brevi tra due possessi della stessa squadra (fallo, poi riprende chi
aveva la palla): `H H D H H` sono due possessi. Se il D si perde, due possessi veri diventano uno
lungo che non si abbina a nessuno dei due.

> **Da decidere:** priorità alle regole che trovano i D brevi (D-WHISTLE, D-EDGE) rispetto a
> quelle che migliorano l'interno dei D lunghi?

### A5. Frame non valutati liberi, e i D prima del calcio d'inizio del secondo tempo

Sui frame non valutati si può scrivere qualsiasi cosa. Nel secondo tempo, tra l'inizio della
valutazione e il calcio d'inizio, ci sono quasi 3.000 frame valutati tutti `D`: se D-KO rileva il
calcio d'inizio, li prende da solo.

Il frame del calcio d'inizio è pubblicato nelle FAQ: va rilevato con l'algoritmo e il valore
pubblicato usato solo come verifica, non scritto nel codice.

> **Da decidere:** `scripts/diagnose_kickoff.py` copre già questo caso?

---

## B. Dai dati: più lavoro, ma molto valore

### B1. Un rilevatore di palla nostro sul video tattico

La palla manca nel 41% dei frame valutati, e soprattutto in quelli interessanti (aree affollate,
contrasti, palle alte). È il problema principale della challenge.

Le regole consentono esplicitamente di migliorare o sostituire i dati forniti con algoritmi
("improving that data, or replacing it with your own, is entirely legitimate"). Un modello di
rilevamento pre-addestrato, eseguito **in locale**, può recuperare parte di quei frame.

Vincolo: i termini e condizioni vietano di caricare video o dati su LLM o servizi cloud di terzi.

> **Da decidere:** chi ci lavora, con quale modello, e quanto tempo gli diamo prima di valutare se
> conviene (scadenza 17 novembre).

### B2. L'ATD broadcast come seconda fonte per la palla

Il broadcast ha un proprio ATD (palla visibile nel 50,5% dei suoi frame), ma su una timeline
diversa: tagli, replay, zoom, nessuna tabella di corrispondenza. Va allineato al video tattico con
un algoritmo (per esempio confrontando le posizioni dei giocatori). Impegnativo, ma consentito.

> **Da decidere:** vale lo sforzo rispetto a B1? Chi stima quanti buchi della palla tattica
> coprirebbe?

### B3. Un set di validazione da altre partite

È l'unico modo onesto per scegliere tra due versioni del sistema: `EVALUATION.md` mostra che i
5 minuti pubblici scelgono la variante sbagliata anche con la baseline banale, e le regole vietano
di usarli per tarare.

- etichettare a mano **altre partite** è permesso in qualsiasi quantità: bastano una o due partite,
  etichettando solo i cambi di possesso e i D;
- in alternativa, verificare se i dati open di Metrica nello stesso formato EPTS hanno eventi da
  cui ricavare il possesso con un algoritmo;
- per avvicinare il dato al nostro, si aggiunge rumore artificiale (palla tolta a tratti come
  nell'ATD di BAR-ATH).

Va dichiarato nel documento tecnico: cosa è stato etichettato, come, da chi e quanto.

> **Da decidere:** quali partite, chi etichetta, e con quale strumento.

---

## Ordine proposto

1. **v0:** A1–A5. Costano poco e vanno dentro la prima versione del motore.
2. **Subito dopo:** B3, perché senza un set di validazione non possiamo confrontare le versioni.
3. **In parallelo, se c'è una persona libera:** B1, il punto che può spostare di più il punteggio.
4. **Solo se avanza tempo:** B2.
