# sequence_model (package `seqmodel`)

Bidirectional temporal model for H/A/D possession labels, trained on external matches (IDSSE).

Setup: `python -m venv .venv`, then `pip install -r requirements.txt`.
Tests (from the repo root): `python -m pytest sequence_model/tests -q`.

## Prediction on one match (`scripts/predict.py`)

```
python sequence_model/scripts/predict.py --weights W.npz --match <idsse .npz | atd .xml> --out DIR [--beta B --w W]
```

`--match` is an IDSSE `.npz` or the provided ATD `.xml` (the `.txt` with the same stem must sit next to it; kloppy is needed only for this).
Defaults: `beta 0`, `w 1`, default transition rates and prior.

Outputs in `DIR`:
- `probs.npz`: `p_h`, `p_a`, `p_d`, `p_change`, all float32 with shape `(T,)`, one row per **video frame** (row i = frame i, ATD `frame_id - 1`).
  `p_h + p_a + p_d = 1` (to rounding, 3 decimals); `p_change` in [0, 1] is the probability of a direct H<->A possession change near the frame.
  Rows outside the graded halves are model outputs on untracked/irrelevant frames and should be ignored (use `labels.csv`: X there).
- `labels.csv`: `frame,label`, `H`/`A`/`D` on graded frames, `X` elsewhere (ATD: outside the two halves in `games.py`).

BAR-ATH (run where the data is; the data must never be uploaded to cloud services):

```
python sequence_model/scripts/predict.py --weights W.npz --match data/tactical/BAR-ATH_tactical_atd.xml --out runs/barath
python evaluation/validate_submission.py runs/barath/labels.csv
```
Two runs give byte-identical files (check SHA-256).
