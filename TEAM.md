# Tikitalia — More than a Hack 2027 (Phase 1: Ball Possession)

Team repository for the MTAH27 challenge: **for every frame of the tactical video, label who has the ball** (`H` / `A` / `D` / `X`).
The official challenge material (Metrica Sports) is merged in from upstream, so read
[README.md](README.md) → [CHALLENGE.md](CHALLENGE.md) → [DATA.md](DATA.md) → [EVALUATION.md](EVALUATION.md) first.

**Deadline: November 17, 2026.**

---

## 1. Setup (once per person)

```bash
git clone https://github.com/Peppone248/tikitalia_initial_challenge_27.git
cd tikitalia_initial_challenge_27
python -m venv .venv
# Windows:   .venv\Scripts\activate
# mac/linux: source .venv/bin/activate
pip install -r requirements-starter.txt
```

## 2. Get the data — it is NOT in git

The match data is **gated** (each of us must enrol and accept the terms) and **must not be
redistributed**. It is also far too big for GitHub (videos 5–14 GB, tracking files 140–200 MB).
`.gitignore` keeps everything in `data/` out of the repo except its README. Never force-add it.

1. Download the bundle from the link you received after enrolment.
2. Put it in `data/` with exactly this layout (rename the videos if they arrive as `*-001.mp4`):

```
data/
├─ tactical/                                  the reference timeline (row i of a submission = frame i)
│  ├─ BAR-ATH_tactical.mp4
│  ├─ BAR-ATH_tactical_atd.xml
│  ├─ BAR-ATH_tactical_atd.txt
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml
└─ broadcast/                                 optional, NOT synchronised with tactical
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

   Keeping the data elsewhere (another disk)? Set `MTAH_DATA_DIR` to that folder instead:
   `$env:MTAH_DATA_DIR = "D:\mtah2027"` (PowerShell) or `export MTAH_DATA_DIR=/path` (bash).

3. Check it:

```bash
python scripts/check_data.py           # every asset found, with sizes
python evaluation/run_evaluation.py    # scores the 4 baselines on the public 5 minutes
```

## 3. Repository layout

```
games.py, evaluation/, starter/, *.md   official challenge material (upstream — don't edit, see §5)
docs/                                   start at docs/README.md: every report in order
src/tikitalia/                          our code: features, models, post-processing
HMM_first_imp/, combined_algorithm/     reports of each approach (the code follows once cleaned up)
notebooks/                              exploration; name them NN_initials_topic.ipynb
scripts/                                command-line entry points (check_data.py, ...)
evaluation/results/                     put submission CSVs here to score them (ignored by git)
data/                                   the match data — local only, ignored by git
```

Code that is reused by more than one notebook goes in `src/tikitalia/`, not copy-pasted between notebooks.
To import it from a notebook, put this at the top:

```python
import sys; sys.path[:0] = ["..", "../src"]
from games import asset_paths
import tikitalia
```

## 4. Workflow

- `master` is shared and should always run. Work on a branch: `git switch -c <name>/<topic>`, open a PR, merge.
- Before submitting anything: `python evaluation/validate_submission.py evaluation/results/<file>.csv`.
- Ranked on **spellF1** (whole possessions recovered); macroF1 is the tie-break. See EVALUATION.md.
- Do not commit: anything in `data/`, videos, full-match ground truth, large intermediate files
  (put caches in `cache/` or `outputs/`, both ignored).
- Stage files by name or folder, and read `git status` before committing. The submission repository
  has to be public (SUBMISSION.md), and what enters the history stays there.
- No manual labelling or correcting of the data — see RULES.md.

## 5. Pulling updates from the organisers

The official repo is the `upstream` remote. When Metrica updates the docs or the scorer:

```bash
git remote add upstream https://github.com/metrica-sports/more-than-a-hack-2027.git   # once
git fetch upstream
git merge upstream/main
```

Leave the upstream files as they are so these merges stay conflict-free; put our changes in `src/`, `notebooks/`, `scripts/`.

## Licence

Our code: [LICENSE](LICENSE). Challenge code from Metrica Sports: [LICENSE-metrica.txt](LICENSE-metrica.txt).
The match data is covered by neither — it is licensed to each participant under the competition terms.
