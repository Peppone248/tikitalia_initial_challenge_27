# changepoint_model — possession from the *changes*, not only the state

A self-contained module (numpy, scikit-learn, numba; torch only to train the recurrent heads)
that labels every frame of the provided match H / A / D, built around one observation from the
team's reports: the points we lose are the **changes of possession that never get cut**
(REPORT_B §20: +0.21 spellF1 if every missing change were found, +0.16 if the possessions that
cover several true ones were cut; §24: raising the HMM's switch rate everywhere only adds wrong
cuts). The frame-wise classifiers of the earlier models have no notion of a boundary; this module
gives them one.

## The idea in one picture

```text
tracking ──► features (odd / even under the team swap)
                │
                ├─► P(home | alive)      antisymmetric logistic regression (odd features)
                ├─► P(dead)              boosted trees (even features + |odd|)
                └─► P(boundary ±0.5 s)   three boosted heads: team change, stop, restart
                                          (inputs: even features, |odd|, and the trajectory of
                                           P(home) around the frame — "stacking")
                │
                ▼
     3-state HMM (H, A, D) with minimum durations whose transition hazard at frame t
     is the boundary probability at t  ──► Viterbi ──► labels
```

Or, with `--heads seq`, one bidirectional GRU replaces the five heads and outputs the same six
quantities per frame; the decoder is unchanged. Prediction never needs torch (numpy forward pass).

### Why a boundary head (state of the art, and what we take from it)

Frame-based temporal segmentation has the same failure as ours: over-smoothing merges short
segments. The literature answers with a **separate boundary branch**: ASRF (Ishikawa et al.,
WACV 2021) predicts frame classes *and* action boundaries and refines the segmentation with the
boundaries; MS-TCN++ and boundary-aware cascade networks do the same with multi-stage refinement.
We keep the cheap part of that idea: a boundary classifier trained on "a change within ±k frames",
turned into a per-frame hazard of the HMM (`decode.py`). Where the hazard is high a cut is cheap
even if the class posteriors do not swing (the 1–2 s possessions we lose); elsewhere the HMM holds.

The team head follows the team's own best practice (REPORT_B §14–16): odd features, no intercept,
so P(home | teams swapped) = 1 − P(home) exactly. Control votes, contacts (Vidal-Codina et al.
2022) and the shape transitions ("after minus before") are the features that mattered there.

## Layout

```text
cpm/match.py        one Match object, whatever the source; attack direction inferred
cpm/loaders/        idsse.py (the team's .npz), metrica.py (CSV + EPTS), epts.py (FIFA/EPTS raw
                    text + metadata XML: the provided ATD and Metrica game 3, same parser)
cpm/labels.py       labels from event data by one mechanical rule (Metrica vocabulary)
cpm/degrade.py      a priori degradation: ball hidden in stoppages, long live gaps, broken tracks
cpm/features.py     per-frame features, odd / even
cpm/models.py       TeamModel (symmetric logistic), EvenModel (hgb | logistic), boundary targets
cpm/decode.py       time-varying-hazard HMM, Viterbi (numba)
cpm/seq.py          bidirectional GRU heads (torch to train, numpy to predict)
cpm/pipeline.py     prepare / fit / predict / decode
cpm/evaluate.py     the official scorer, the official baseline
cpm/kickoff.py      Law 8: D before each period's kick-off (Smart Tagging kick-off tag)
cpm/rule_layers.py  the team's precise dead-ball layers (HMM_first_imp), for evaluation
scripts/eval_lomo.py    leave-one-match-out on external matches
scripts/diag_changes.py where the changes are lost
scripts/train.py        final models from external matches
scripts/predict_atd.py  the submission: ATD -> tikitalia.csv
tests/                  pytest
```

## Reproduce the submission (v4.2)

```powershell
# 0. the referee's whistle from the match audio, with the literature profile (REPORT_B 18 and 31:
#    the audio is extracted once with ffmpeg into data\, nothing is calibrated on it)
.\venv\Scripts\python.exe HMM_first_imp\scripts\detect_whistles.py --profile config_b/whistle_profile_literature.json --out outputs/whistles/BAR-ATH_whistles_lit.json
# 1. label the provided match: v4 hybrid model + D before each kick-off (Law 8) + whistle R7a
.\venv\Scripts\python.exe changepoint_model\scripts\predict_atd.py --models changepoint_model\models\cpm_v4_hybrid.pkl --whistle HMM_first_imp\outputs\whistles\BAR-ATH_whistles_lit.json --out outputs\cpm_v42\tikitalia.csv
# 2. check the format
.\venv\Scripts\python.exe evaluation\validate_submission.py outputs\cpm_v42\tikitalia.csv
```

Expected: SHA-256 `8682095eaa072212e2b0be95e8acb87713c1f7312d474598ce694e263a04491d`, 292 possessions,
D 44.4 % of the graded frames; two runs give the same file. Needs the team's `HMM_first_imp` and
`src/tikitalia` (rule code, whistle) next to this module, and the Smart Tagging file (kick-offs).

The model files are committed, so training is optional. To retrain from external data only:

```powershell
git clone --depth 1 https://github.com/metrica-sports/sample-data ext\metrica
.\venv\Scripts\python.exe changepoint_model\scripts\train.py --metrica ext\metrica\data --idsse data\idsse --pff data\pff --scenario degraded --seed 2027 --subsample 4 --tune-matches 6 --gammas 4 8 16 --ws 0.25 0.5 --out changepoint_model\models\cpm_v2_all18.pkl
.\venv\Scripts\python.exe changepoint_model\scripts\make_hybrid.py --trees changepoint_model\models\cpm_v2_all18.pkl --metrica ext\metrica\data --idsse data\idsse --pff data\pff --out changepoint_model\models\cpm_v4_hybrid.pkl
```

A GRU retrained on another machine or torch version may not give the same weights; the committed
weights give the same labels everywhere (numpy forward pass).

## Rules

- RULES.md: nothing is trained, tuned or chosen on the provided match or on the public window;
  every constant is a priori or chosen on external matches (leave one match out); corrections
  come out of code applied to the whole match; the data stays out of git.
- Mail of 03/11: the public window is scored once, after the choice, as a structure check only.
- External data: Metrica sample data (public GitHub repository, README asks for attribution),
  IDSSE (CC BY 4.0), PFF FC World Cup 2022 (registration).

See `VERSIONS.md` for every version tried and its numbers, and `REPORT.md` for the reading.

## Determinism check (RULES.md 3)

The same command must give the same file on every machine. On the reference machine, once:

```
python changepoint_model/scripts/check_determinism.py --write-reference
```

On any other machine (fresh venv, `pip install -r changepoint_model/requirements.txt`, data bundle in
`data/`, ffmpeg on the PATH, optionally the Metrica sample data in `ext/metrica`):

```
python changepoint_model/scripts/check_determinism.py
```

It runs four steps twice each (Metrica sample game 3, the provided match without and with the whistle,
the whistle detection itself) and compares the SHA-256 of every output with
`determinism_reference.json`. The report in `outputs/determinism/` holds hashes and versions only.
The numerical libraries are pinned to one thread inside `predict_atd.py`.
