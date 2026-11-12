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

## Data -> evaluate -> (train) -> predict

1. Environment (inside `sequence_model/`, CPU only):
   ```
   cd sequence_model
   python -m venv .venv
   .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   .venv/Scripts/python -m pip install -r requirements.txt
   ```
   (On Linux/macOS use `.venv/bin/python`.)
2. Data: `python scripts/fetch_idsse.py` downloads the 7 IDSSE matches and fills `sequence_model/data/idsse/` with one `.npz` per match. The data must never be uploaded to cloud services.
3. Evaluate (leave-one-match-out, `frame` vs `tcn` vs `tcn_change`, plus `baseline50`):
   ```
   python sequence_model/scripts/run_lomo.py          # defaults: all models, seeds 2027 2028
   ```
   Runtime: measured on 16 cores: about 3 min per TCN fold (fit 40 s, ~7-10 s per match prediction, decoding grid ~1 min), so roughly 45-60 min for 7 folds x 2 seeds of the tcn group, plus a few minutes for frame; fewer cores will be slower; the per-phase timings on stderr tell where the time goes. Progress lines on stderr: per training epoch `[train_tcn] epoch e/N train_loss .. val_loss .. Ns`, per fold `[fold <match>] fit / test-match predict / test-match decode(s)`, `[select_decoder] inner predictions / grid`, and after each fold `[<group> seed <s>] held-out <match>: Ns`. The table goes to stdout and to `runs/lomo_<models>_<seeds>.md`.
   - The criterion needs at least 2 seeds and at least 7 paired matches per seed; with fewer seeds the report prints `criterion: needs at least 2 seeds` instead of a verdict.
   - `frame` and `tcn` choose only `w` on the training matches; `tcn_change` chooses `w` and `beta`, so the change branch is its only extra freedom.
   - `p_change` is the probability of a direct H<->A possession change within +-`change_k` frames (`config.yaml`, 12 frames = 0.5 s) of the frame; the decoder raises the H<->A transition rate by `1 + beta * p_change`.
   - Caveat: absolute LOMO scores are optimistic, because the degradation hides the ball exactly on the held-out match's true dead-ball runs. Relative comparisons between models stay fair. Do not quote the absolute numbers as BAR-ATH expectations.
4. Train: there is no training script yet. It is to be added only if the criterion passes; `predict.py` needs weights `W.npz` written by `tcn.export_weights` from such a script.
5. Predict: see the section above.

Notes: `train_tcn` calls `torch.use_deterministic_algorithms(True)` and `set_num_threads`, which are process-global (they stay set for the rest of the Python process). The TCN is trained with inverse-frequency class weights, so its output is already prior-corrected and the decoder uses a uniform prior for it (`prior_corrected` attribute on the predictor).
