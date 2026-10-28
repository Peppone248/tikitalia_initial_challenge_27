# Il motore

Cosa fa, cosa riceve, cosa restituisce e come decide. Termini: [NOMENCLATURA.md](NOMENCLATURA.md)
§6. Pseudocodice originale: [REGOLE_PSEUDO_LABEL.md](REGOLE_PSEUDO_LABEL.md) §9. Stato
dell'implementazione: [STATO_LAVORI.md](STATO_LAVORI.md) §5. Lo strato a monte delle regole (le
misure che le regole leggono): [FEATURES.md](FEATURES.md).

---

## 1. In una frase

Il motore prende le uscite di **tutte** le regole e decide **una sola etichetta per frame**
(`H`, `A`, `D`, `X`), scorrendo la partita in ordine e ricordando lo stato del frame precedente.

Le regole sono testimoni: ognuna dice cosa ha visto su certi frame, spesso tace, a volte si
contraddice con le altre. Il motore è il giudice: mette insieme le testimonianze e, dove nessuno
dice niente, tiene la decisione precedente.

```text
stato[f] = decidi( voti_pesati[f], stato[f-1] )
```

## 2. Dove sta nella pipeline

```text
dati grezzi ─► Match ─► features ─► events ─► regole ─► MOTORE ─► CSV
               misure    misure      fatti      opinioni  decisione
               pulite    derivate    nel tempo  pesate    unica
```

| Strato | Domanda a cui risponde | Soglie? | Etichette? |
|---|---|---|---|
| `preprocessing.py` → `Match` | dove sono palla e giocatori, in metri? | solo pulizia (`V_JUMP`) | no |
| `features.py` ([FEATURES.md](FEATURES.md)) | quanto distano, quanto sono sparsi, quanto corrono? | no | no |
| `events.py` | quando succede qualcosa (controllo, passaggio, ripresa)? | sì | no |
| `rules/` | cosa significa per il possesso, e quanto è netto? | sì | sì, una per regola |
| `engine.py` | qual è l'etichetta finale di ogni frame? | poche, del motore | sì, una sola |

Due separazioni da non rompere:
- le **regole non si conoscono tra loro**: aggiungerne una non tocca le altre né il motore;
- il **motore non sa niente di calcio**: legge etichette, pesi e tipi, non posizioni.

## 3. Ingresso e uscita

```text
IN   outputs: dict[rule_id → RuleOutput]   label[n] ('H'|'A'|'D'|''), strength[n], episodes
     rules:   prior_q, kind, scope          da rules.yaml
     match:   n, fps, half, graded          solo la cornice della partita
     params:  CONF_ANCHOR, INERTIA_MIN, CONF_SOFT_START
OUT  labels:  array[n] di 'H'|'A'|'D'|'X'
     conf:    array[n] in [0,1]             dove il motore è incerto (per il report)
     log:     conflitti tra ancore, cambi accettati e scartati
```

| Ingresso | Cosa usa il motore | Perché |
|---|---|---|
| `label` | il voto della regola su ogni frame | `''` = astensione, **mai** un voto contrario |
| `strength` (`s`) | quanto è netta questa attivazione | peso del voto |
| `episodes` | inizio e fine di ogni attivazione | retrodatare i cambi, scrivere il log |
| `prior_q` (`q`) | quanto la regola ci azzecca in generale | peso del voto: `w = q · s` |
| `kind` | cosa può fare il voto | solo `ANCHOR` cambia lo stato |
| `scope` | in quale contesto il voto vale | `OPEN_PLAY` non vota durante un D |
| `half`, `graded` | dove lavorare e dove azzerare la memoria | l'inerzia non attraversa l'intervallo |

## 4. Le cinque fasi

### 4.1 Voti

Ogni attivazione diventa un voto con peso `w = q · s`, sommato in una matrice frame × {H, A, D}.

```python
score = np.zeros((n, 3))                      # colonne H, A, D
for rid, out in outputs.items():
    w = rules[rid].prior_q * out.strength
    for k, lab in enumerate("HAD"):
        m = out.label == lab
        score[m, k] += w[m]
```

### 4.2 Ancore

Un frame ha un'ancora se una regola `ANCHOR` scatta con `w ≥ CONF_ANCHOR`.

- una sola ancora, o più ancore concordi → `anchor[f] = etichetta`
- ancore discordi → `CONFLICT`, nel log; il frame si tratta come senza ancora
- nessuna ancora → `None`

Ordine obbligato: **prima le ancore D**, perché da loro dipende il contesto (`DEAD` /
`OPEN_PLAY`) che filtra i voti H/A con `scope`.

### 4.3 Inerzia

È una macchina a stati che gira **una volta per tempo**, sui soli frame valutati:

```text
per ogni tempo:
    state = None                                   # memoria azzerata
    per ogni frame f del tempo, in ordine:
        se anchor[f] == D                                   → state = D
        se state == D e c'è una ripresa a f                 → state = squadra che riprende
        se anchor[f] ∈ {H,A}, ≠ state, e dura ≥ INERTIA_MIN → state = anchor[f]   (cambio)
        se state è None e un voto SOFT supera CONF_SOFT_START → state = quel voto
        altrimenti                                           → state invariato   (inerzia)
        label[f] = state
```

- Solo le `ANCHOR` cambiano lo stato. I voti `SOFT` scelgono solo lo stato iniziale.
- Un'ancora H/A più breve di `INERTIA_MIN` è un tocco, non un controllo: si registra e basta.

### 4.4 Post-processing (scelte legate alla metrica, non al calcio)

- **Retrodatazione:** un cambio accettato si sposta all'**inizio dell'episodio** che l'ha prodotto,
  letto da `episode["start"]`. Il motore se ne accorge con `INERTIA_MIN` di ritardo, ma la metrica
  punisce molto i ritardi (EVALUATION.md: 1 s di ritardo → 0,65).
- **Nessuna `X` sui frame valutati:** ogni `X` diventa l'ultima etichetta data in quel tempo
  (IMPROVEMENTS A1). Sui frame non valutati resta `X`.

### 4.5 Scrittura

`frame,label`, 199.319 righe, poi `evaluation/validate_submission.py`.

## 5. Esempio

### 5.1 Giocattolo (12 frame, un tempo)

`INERTIA_MIN` = 3 frame, `CONF_ANCHOR` = 0,5. Nelle celle: etichetta e forza `s`.

```text
frame          0    1    2    3    4    5    6    7    8    9    10   11
P-DRIB  q=.85  H.9  H.9  H.9  ·    ·    A.8  A.8  ·    ·    ·    ·    ·
D-OUT   q=.95  ·    ·    ·    ·    ·    ·    ·    ·    D1   D1   ·    ·
P-SHAPE q=.40  H.5  H.5  H.5  H.5  A.6  A.6  A.6  A.6  ·    ·    A.6  A.6   (SOFT)
R-TOUCH        ·    ·    ·    ·    ·    ·    ·    ·    ·    ·    A    ·     (ripresa)

w ancore       .77  .77  .77  ·    ·    .68  .68  ·    .95  .95  ·    ·
anchor         H    H    H    ·    ·    A    A    ·    D    D    ·    ·
state          H    H    H    H    H    H    H    H    D    D    A    A
perché         anc  anc  anc  ine  ine  corta corta ine anc  anc  rip  ine
```

- frame 4: P-SHAPE vota A, ma è SOFT → l'inerzia tiene H;
- frame 5–6: ancora A di 2 frame, sotto `INERTIA_MIN` → tocco, nessun cambio;
- frame 8–9: D-OUT con `w = 0,95` → D;
- frame 10: R-TOUCH chiude il D con A.

Se l'ancora A dei frame 5–6 fosse durata 3 frame, il cambio sarebbe accettato e scritto **dal
frame 5** (retrodatazione), non dal frame 7 in cui il motore se ne accorge.

### 5.2 Un'azione di gioco (12 secondi, `INERTIA_MIN` = 1 s)

```text
t (s)    in campo                                   ancora          stato  perché
0–2      difensore H conduce                        H (P-DRIB)      H      ancora
2–3      passaggio H in volo                        —               H      inerzia
3–4      compagno H controlla                       H (P-DRIB)      H      ancora
4–4,4    A tocca, la palla schizza via              A per 0,4 s     H      troppo breve: tocco
4,4–6    palla vagante, il tracking la perde        —               H      inerzia
6–8      A raccoglie e conduce                      A (P-DRIB)      A      ≥ 1 s → cambio da t = 6
8–9      lancio lungo di A                          —               A      inerzia
9–11     palla oltre la linea laterale              D (D-OUT+HOLD)  D      ancora D
11       rimessa battuta da H                       ripresa H       H      fine del D
11–12    H controlla                                H (P-DRIB)      H      ancora
```

Tra t = 4,4 e 6 l'etichetta viene **solo** dall'inerzia. Se A avesse preso palla a t = 5 senza che
il tracking lo vedesse, l'errore durerebbe un secondo: è il costo dell'inerzia, e il motivo per cui
servono T-TEAMFLIP e S-TRANS nei buchi della palla.

## 6. Parametri del motore

Vanno in `params.yaml` come gli altri, con origine e motivazione.

| Nome | Valore iniziale | Origine | Ruolo |
|---|---|---|---|
| `CONF_ANCHOR` | 0,5 | a priori | peso minimo perché un voto ANCHOR fissi lo stato |
| `INERTIA_MIN` | 1 s | a priori | durata minima di un'ancora H/A per cambiare possesso |
| `CONF_SOFT_START` | da definire | a priori | peso minimo dei voti SOFT per lo stato iniziale |

## 7. Struttura del codice

```text
src/tikitalia/engine.py
    collect_votes(outputs, rules)                          -> score[n, 3]
    find_anchors(outputs, rules, params)                   -> anchor[n], conflicts
    restarts(outputs, rules)                               -> {frame: team}
    run_inertia(anchor, score, restarts, match, params)    -> label[n], changes
    postprocess(label, changes, match)                     -> label[n]
    decide(outputs, rules, match, params)                  -> labels, conf, log
src/tikitalia/submission.py
    write_csv(labels, path)
```

Ogni funzione si testa da sola con `RuleOutput` costruiti a mano (`tests/scenarios.py`). L'esempio
5.1 è il primo test di `run_inertia`.

## 8. Cosa NON va nel motore

- **Logica calcistica** ("se la palla è alla bandierina…"): è una regola.
- **Feature o posizioni**: il motore non legge `Match` oltre la cornice (`n`, `fps`, `half`).
- **Soglie sparse nel codice**: solo quelle del §6, in `params.yaml`.
- **Tarature sui 5 minuti pubblici** (RULES.md §2).

## 9. Evoluzione

| Versione | Voti | Decisione | Note |
|---|---|---|---|
| v0 | somma pesata `q · s` | isteresi con `INERTIA_MIN` | questo documento |
| v1 | label model (Dawid–Skene): `q` stimate da accordi e disaccordi | HSMM / Viterbi con le ancore come vincoli | stesse firme delle funzioni del §7 |

## 10. Decisioni aperte

- Istante del cambio: primo tocco o inizio del controllo? (REGOLE §11 Q8, IMPROVEMENTS A2)
- Soglia di accettazione dei cambi: prudente o intorno al 50% a livello di segmento? (IMPROVEMENTS A3)
- Politica sui buchi lunghi della palla in gioco.
- `INERTIA_MIN` a priori o dalla distribuzione delle durate dei controlli.
