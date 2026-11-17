# changepoint_model: ball possession from tracking (submission v4.2)

Labels every video frame of the provided match as H / A / D (X outside the two halves) from the
automatic tracking, with two rules from the Laws of the Game on top. How it works, step by step:
[HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Reproduce the submission

From the repository root, with the match bundle in `data/` (or `MTAH_DATA_DIR`) and `ffmpeg` on
the PATH:

```
python -m venv venv
venv\Scripts\pip install -r changepoint_model\requirements.txt
venv\Scripts\python.exe changepoint_model\predict.py --out outputs\tikitalia.csv
venv\Scripts\python.exe evaluation\validate_submission.py outputs\tikitalia.csv
```

Expected: SHA-256 `8682095eaa072212e2b0be95e8acb87713c1f7312d474598ce694e263a04491d`, about one
minute. The first run extracts the audio of the tactical video once
(`data/tactical/BAR-ATH_tactical_audio.flac`, 16 kHz mono FLAC). Produced with Python 3.10.4,
the pinned versions of `requirements.txt` and ffmpeg 9.0.2.

`python changepoint_model/scripts/check_determinism.py` runs the pipeline twice and compares the
SHA-256 with `determinism_reference.json` (reference machine).

## Pipeline

```
tracking ─► features (134 per frame) ─► GRU: P(home | ball in play)          ─┐
                                    └─► boosted trees: P(dead), P(change),     ├─► HMM (H, A, D), hazards from the
                                        P(stop), P(restart) within ±0.5 s     ─┘   boundary heads, Viterbi
                                                                                        │
Smart Tagging kick-off tag ─────────────────────────────► Law 8: D before each kick-off ┤
audio of the tactical video ─► whistle blasts ──────────► Law 5: D from a whistle that  ┤
                                                          slows the players down        ▼
                                                                                   tikitalia.csv
```

| file | role |
|---|---|
| `predict.py` | the submission: provided match -> `frame,label` CSV |
| `cpm/loaders/` | EPTS (provided ATD, Metrica) and IDSSE / PFF readers, one `Match` format |
| `cpm/features.py` | the 134 frame features (66 change sign when the teams are swapped, 68 do not) |
| `cpm/models.py`, `cpm/seq.py` | team logistic, boosted trees, bidirectional GRU (numpy inference) |
| `cpm/pipeline.py` | prepare -> predict -> decode |
| `cpm/decode.py` | HMM with minimum durations and time-varying hazards (numba Viterbi) |
| `cpm/kickoff.py` | Law 8, from the Smart Tagging "Kick Off (Start)" tag |
| `cpm/whistle.py` | whistle detector (audio) and the whistle rule |
| `cpm/degrade.py`, `cpm/labels.py`, `cpm/evaluate.py` | training utilities, labels from events, official scorer |
| `models/cpm_v4_hybrid.pkl` | the submitted model; `cpm_v2_all18.pkl` holds the trees it was built from |
| `scripts/train.py`, `scripts/make_hybrid.py` | training from external data |
| `scripts/eval_lomo.py`, `scripts/eval_hybrid.py` | leave-one-match-out evaluation |
| `data_prep/` | download / conversion of IDSSE and PFF to the `.npz` the loaders read |
| `tests/` | `python -m pytest changepoint_model/tests` |

## Training data (external only)

| source | matches | labels from | licence / access |
|---|---|---|---|
| Metrica Sports sample data | 3 | the published events (rule in `cpm/labels.py`) | public, attribution (`LICENSE-metrica.txt`) |
| IDSSE, Bundesliga 2022/23 (Bassek et al., Sci. Data 2025) | 7 | DFL ball-status and possession flags | CC BY 4.0, `data_prep/fetch_idsse.py` |
| PFF FC World Cup 2022 (broadcast tracking) | 8 | PFF game events | free registration, `data_prep/fetch_pff.py` |

No label of the provided match is used anywhere, and no person labelled any frame of it. The
model files are committed; to retrain:

```
git clone --depth 1 https://github.com/metrica-sports/sample-data ext\metrica
python changepoint_model\data_prep\fetch_idsse.py
python changepoint_model\data_prep\unpack_pff.py <PFF download>\*.zip
python changepoint_model\data_prep\fetch_pff.py --n 8
python changepoint_model\scripts\train.py --metrica ext\metrica\data --idsse data\idsse --pff data\pff --scenario degraded --seed 2027 --subsample 4 --tune-matches 6 --gammas 4 8 16 --ws 0.25 0.5 --out changepoint_model\models\cpm_v2_all18.pkl
python changepoint_model\scripts\make_hybrid.py --trees changepoint_model\models\cpm_v2_all18.pkl --metrica ext\metrica\data --idsse data\idsse --pff data\pff --out changepoint_model\models\cpm_v4_hybrid.pkl
```

A GRU retrained with another torch version or machine may not reproduce the committed weights;
prediction from the committed weights is deterministic everywhere.

## Validation (leave one match out, official scorer, external matches)

| | spellF1 | baseline (smooth 50) |
|---|---|---|
| trees only (v2), 8 PFF matches (real broadcast tracker) | 0.583 | 0.365 |
| trees only (v2), 7 IDSSE / 3 Metrica matches | 0.657 / 0.733 | 0.265 / 0.325 |
| hybrid (v4) vs trees (v2), 6 held-out matches | 0.653 vs 0.637 (better on 6 of 6) | |

All choices were made on these external evaluations. On the public 5-minute window, scored as a
check only, v4.2 reaches 0.691 spellF1 (official baseline 0.326).

## Rules (RULES.md, FAQ, organisers' mail of 03/11)

- Inputs: the provided tracking, the Smart Tagging kick-off tag and the audio of the provided
  video, all read by code. No hand labels, no human-produced data on the provided match, no
  correction of its tracking.
- Training and every choice of model or constant: external data or a priori physical constants
  (metres, seconds). Nothing is trained, fitted or calibrated on the provided match, its video or
  its ground truth.
- The two rules come from the Laws of the Game (Law 8: play starts with the kick-off; Law 5: play
  stops at the referee's whistle). The kick-off rule is a no-op on the external matches, whose
  periods start at the kick-off. The whistle detector uses the frequency band published for
  referee whistles in broadcast audio (Raventos et al., SpringerPlus 2015, arXiv 1411.6496); its
  other constants are a priori physical values (dB, seconds, m/s).
