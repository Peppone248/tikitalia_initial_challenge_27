# Primi passi — cosa lanciare e in che ordine

Checklist per chi parte da zero sulla repo del team. Comandi per **Windows PowerShell**, lanciati
dalla root della repo (`tikitalia_initial_challenge_27\`). Su macOS/Linux cambiano solo l'attivazione
del venv e il modo di impostare le variabili d'ambiente.

> I dati della partita restano **solo in locale**: non vanno committati, né caricati su LLM in cloud
> (Termini e Condizioni). Vedi [TEAM.md](TEAM.md) §2 e [RULES.md](RULES.md).

---

## 0. Clonare la repo (una volta)

```powershell
git clone https://github.com/Peppone248/tikitalia_initial_challenge_27.git
cd tikitalia_initial_challenge_27
```

## 1. Ambiente Python (una volta)

Python **3.9+** (consigliato 3.11).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-starter.txt     # numpy, pandas, kloppy, matplotlib, notebook
```

Se PowerShell blocca l'attivazione: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
Da qui in poi ogni comando va lanciato **con il venv attivo**.

## 2. Mettere i dati in `data\`

Scompatta il bundle ricevuto all'iscrizione in `data\tactical\` e `data\broadcast\` con i nomi
indicati in [TEAM.md](TEAM.md) §2 (rinomina i video se arrivano come `*-001.mp4`).
In alternativa lasciali dove sono e punta la variabile d'ambiente:

```powershell
$env:MTAH_DATA_DIR = "D:\percorso\ai\dati"   # vale solo per la sessione corrente
```

## 3. Controllare i dati

```powershell
python scripts\check_data.py
```

Atteso: `ok` su tutti i 5 file tactical e `All tactical assets present.` (broadcast è opzionale).

## 4. Lo scorer funziona (non servono dati)

```powershell
python evaluation\run_evaluation.py
```

Atteso: la classifica delle 4 baseline sui 5 minuti pubblici (`baseline_smooth125` prima con
spellF1 0.4242). Se esce questa tabella, ambiente e scorer sono a posto.

## 5. Il caricamento dei dati funziona

```powershell
python starter\load_atd.py              # tracking: 199319 record, squadre H/A, frame allineati
python starter\load_smart_tagging.py    # 638 istanze di Smart Tagging
```

Il primo legge ~190 MB di tracking: ci mette un po'.

## 6. Rigenerare la baseline dai dati

```powershell
python starter\baseline.py
python evaluation\run_evaluation.py
```

Riscrive i 4 CSV in `evaluation\results\` partendo dal tracking (1–2 minuti) e li riscora: i numeri
devono coincidere con quelli del punto 4. Se coincidono, la vostra pipeline dati → submission è
corretta.

Questi 4 file sono versionati: se `git status` li mostra modificati (es. fine riga), ripristinali
con `git restore evaluation/results/` — **non committarli**.

## 7. Validare una submission

```powershell
python evaluation\validate_submission.py evaluation\results\baseline_smooth50.csv
```

Fa esattamente i controlli dello scorer ufficiale (199,319 righe, header, etichette) e stampa
numero e durata dei possessi. Va lanciato **su ogni file prima di consegnarlo**.

## 8. Notebook guidati

```powershell
jupyter notebook
```

Aprili in quest'ordine, dalla cartella `starter\`:

1. `01_smart_tagging.ipynb` — il tagging tattico come intervalli di frame
2. `02_tracking_atd.ipynb` — il tracking, con grafici (palla assente sul 41% dei frame giocati)
3. `03_first_submission.ipynb` — dai dati a una submission scorata

## 9. Leggere i documenti (in quest'ordine)

[CHALLENGE.md](CHALLENGE.md) → [DATA.md](DATA.md) → [EVALUATION.md](EVALUATION.md) →
[RULES.md](RULES.md) → [SUBMISSION.md](SUBMISSION.md) → [FAQ.md](FAQ.md)

---

## Da ricordare

- **I 5 minuti pubblici servono solo a verificare la pipeline**: niente training, niente tuning,
  niente scelta tra due versioni (RULES.md, regola 2).
- Le submission di prova vanno in `evaluation\results\` (ignorati da git, tranne le 4 baseline).
- Il proprio lavoro si fa su un branch: `git switch -c <nome>/<argomento>`, poi PR verso `master`.
