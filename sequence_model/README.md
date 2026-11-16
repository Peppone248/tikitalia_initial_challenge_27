# sequence_model (package `seqmodel`)

Bidirectional temporal model (TCN) for H/A/D possession labels, trained on external matches (IDSSE). Results: [RESULTS.md](RESULTS.md).
All commands run from the repo root; `sequence_model/.venv/Scripts/python.exe` is the module's virtualenv.

## Setup

```
python -m venv sequence_model/.venv
sequence_model/.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
sequence_model/.venv/Scripts/python.exe -m pip install -r sequence_model/requirements.txt
sequence_model/.venv/Scripts/python.exe -m pytest sequence_model/tests -q
```
(On Linux/macOS use `sequence_model/.venv/bin/python`.)

## 1. Data

```
sequence_model/.venv/Scripts/python.exe sequence_model/scripts/fetch_idsse.py
```
Downloads the 7 IDSSE matches into `sequence_model/data/idsse/` (one `.npz` each). The data must never be uploaded to cloud services.

## 2. Evaluate (leave-one-match-out)

```
sequence_model/.venv/Scripts/python.exe sequence_model/scripts/run_lomo.py          # all models, seeds 2027 2028
```
Compares `frame`, `tcn`, `tcn_change` and `baseline50`. Runtime about 45-60 min on 16 cores (about 3 min per TCN fold; per-phase timings on stderr). The table goes to stdout and to `sequence_model/runs/lomo_<models>_<seeds>.md`.
- The criterion needs at least 2 seeds and 7 paired matches per seed.
- `frame` and `tcn` choose only `w` on the training matches; `tcn_change` chooses `w` and `beta`. The selection is in-sample on the training matches; the held-out match is never used.
- `p_change` is the probability of a direct H<->A change within +-`change_k` frames of the frame; the decoder raises the H<->A rate by `1 + beta * p_change`.
- Caveat: absolute LOMO scores are optimistic (the degradation hides the ball on the held-out match's true dead-ball runs). Only relative comparisons are fair.

## 3. Train (the weights are already in the repo)

`sequence_model/weights/tcn_idsse.npz` and `tcn_idsse_decode.json` are versioned; retrain only if needed:
```
sequence_model/.venv/Scripts/python.exe sequence_model/scripts/train.py --data sequence_model/data/idsse --seed 2027 --out sequence_model/weights/tcn_idsse.npz
```
Trains on all IDSSE matches (degraded as in the evaluation) and writes the weights plus the sidecar `tcn_idsse_decode.json`: transitions counted on the controlled labels of the training matches, `prior` null (uniform: the TCN is trained with inverse-frequency class weights, already prior-corrected), `beta` 0, `w` 1, plus seed, matches, git commit and config hash. This is the measured configuration.

## 4. Predict (`scripts/predict.py`)

IDSSE match:
```
sequence_model/.venv/Scripts/python.exe sequence_model/scripts/predict.py --weights sequence_model/weights/tcn_idsse.npz --match sequence_model/data/idsse/J03WMX.npz --out <DIR>
```
BAR-ATH (run where the data is; the data must never be uploaded to cloud services; the `.txt` with the same stem must sit next to the `.xml`, kloppy is needed only for this):
```
sequence_model/.venv/Scripts/python.exe sequence_model/scripts/predict.py --weights sequence_model/weights/tcn_idsse.npz --match data/tactical/BAR-ATH_tactical_atd.xml --out runs/barath
sequence_model/.venv/Scripts/python.exe evaluation/validate_submission.py runs/barath/labels.csv
```
Decoding settings (trans, prior, beta, w) are read from the sidecar `<weights>_decode.json` next to the weights; `--beta` / `--w` override it. If the sidecar is missing, a warning is printed on stderr and the defaults are used (default transitions, beta 0, w 1), which differ from the measured configuration.

Outputs in `<DIR>`:
- `probs.npz`: `p_h`, `p_a`, `p_d`, `p_change`, float32, shape `(T,)`, one row per video frame (ATD `frame_id - 1`). `p_h + p_a + p_d = 1` (to 3 decimals). `p_change` is kept but unused by the shipped configuration. Rows outside the graded halves should be ignored (use `labels.csv`).
- `labels.csv`: `frame,label`, `H`/`A`/`D` on graded frames, `X` elsewhere.

Two runs give byte-identical files (check SHA-256).

Notes: `train_tcn` calls `torch.use_deterministic_algorithms(True)` and `set_num_threads`, which are process-global (they stay set for the rest of the Python process). The TCN is trained with inverse-frequency class weights, so its output is already prior-corrected and the decoder uses a uniform prior for it (`prior_corrected` attribute on the predictor).
