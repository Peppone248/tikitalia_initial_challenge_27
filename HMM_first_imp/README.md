# HMM_first_imp — first CSV for the Initial Challenge (v0, HMM engine)

```text
dati grezzi ─► Match ─► features ─► events ─► regole ─► MOTORE ─────────────► CSV
               misure    misure      fatti      opinioni  NN light + Bakis HMM  decisione
               pulite    derivate    nel tempo  pesate    (Viterbi per tempo)   unica
```

Everything that decides the output is in `config/*.yaml`. Write-ups:
[REPORT_HMM_v0.md](REPORT_HMM_v0.md) (the pipeline) and [REPORT_HMM_v1.md](REPORT_HMM_v1.md)
(v1: rule-gated team changes, TURNOVER rules, D-WHISTLE anchor, machine-independent NN).
`config_v0/` keeps the v0 settings: `--config-dir HMM_first_imp/config_v0`.

**Approach C (v2), the best on the IDSSE validation:** `--config-dir HMM_first_imp/config_c`
(segment first, then assign the team: [REPORT_C_v2.md](REPORT_C_v2.md)). Validation on other
games: [REPORT_VALIDATION_v0_v1.md](REPORT_VALIDATION_v0_v1.md), `scripts/run_validation.py`.
**Approach C v3** (Smart Tagging dead ball, goalkeeper, simpler evidence): `--config-dir HMM_first_imp/config_c3`,
[REPORT_C_v3.md](REPORT_C_v3.md).
**Approach B (boosted baseline), the best on IDSSE so far** (0.354 degraded / 0.500 clean vs
c_v3 0.319 / 0.479): `python HMM_first_imp/scripts/make_submission_b.py` -> `outputs/run_b/`,
[REPORT_B.md](REPORT_B.md). Same report: why we have too few possessions, D-SLOW for fouls and
offsides, and the broadcast recording aligned to the tactical timeline (`scripts/align_broadcast.py`).
Public-window evaluation of every version side by side: [EVAL_TRACKING.md](EVAL_TRACKING.md)
(`scripts/track_public_eval.py`, versions listed in `config/eval_tracking.yaml`).

## Run

From the repository root (the folder that contains `games.py`, `data/` and `HMM_first_imp/`):

```powershell
# Windows / PowerShell, with the team venv
.\venv\Scripts\python.exe -m pip install -r HMM_first_imp\requirements.txt
.\venv\Scripts\python.exe HMM_first_imp\scripts\make_submission.py
```

```bash
# macOS / Linux
python -m pip install -r HMM_first_imp/requirements.txt
python HMM_first_imp/scripts/make_submission.py
```

About 80 s and ~1.2 GB of RAM. The script forces one BLAS thread; the CSV is the same on every machine (REPORT_HMM_v1 §1). Output in `HMM_first_imp/outputs/run/` (gitignored):

| file | what |
|---|---|
| `tikitalia.csv` | the submission, `frame,label`, 199,319 rows, already checked with `evaluation/validate_submission.py` |
| `run_summary.json` | noise per feature, rules (coverage, strength, weight, agreement), pseudo-labels, NN, HMM, output diagnostics — no ground truth involved |
| `run_config_snapshot.json` | every parameter value used by this run |
| `episodes/*.csv` | episodes per rule, with frame numbers (local debugging only) |

Options:

```bash
python HMM_first_imp/scripts/make_submission.py --set hmm.noise.mode=std        # one value
python HMM_first_imp/scripts/make_submission.py --set hmm.hmm.min_duration_s.D=1.0
python HMM_first_imp/scripts/make_submission.py --config-dir my_configs/        # whole files
python HMM_first_imp/scripts/make_submission.py --out-dir outputs/run_b
```

`--set` takes `hmm.*`, `rules.*`, `pipeline.*` (e.g. `rules.P-SHAPE.enabled=false`). Thresholds
are edited in `config/params_hmm.yaml` (same schema as the team `params.yaml`: value, unit,
origin, why).

## Other commands

```bash
python HMM_first_imp/scripts/eval_sigma_synthetic.py            # MAD vs std for the noise (synthetic)
python -m unittest discover -s HMM_first_imp/tests -v           # 19 synthetic tests
python -m unittest discover -s tests -v                         # team tests, untouched (39)
```

Checking the CSV on the public 5 minutes (pipeline correctness only — never to choose between
versions, RULES.md 2):

```bash
mkdir -p HMM_first_imp/outputs/public_check/results
cp HMM_first_imp/outputs/run/tikitalia.csv HMM_first_imp/outputs/public_check/results/hmm_v0.csv
python evaluation/run_evaluation.py --results HMM_first_imp/outputs/public_check/results --out HMM_first_imp/outputs/public_check/reports
```

## Layout

```text
HMM_first_imp/
├─ config/
│  ├─ params_hmm.yaml     thresholds added here (merged on top of src/tikitalia/config/params.yaml)
│  ├─ rules_hmm.yaml      rule registry: class, kind, scope, stage, prior_q, params, options
│  ├─ hmm.yaml            noise mode, votes, pseudo-labels, NN, HMM (states, Bakis, transitions, ω)
│  └─ pipeline.yaml       game, data/cache paths, output folder, team name
├─ hmm_v0/
│  ├─ run.py              the pipeline, one function per layer
│  ├─ noise.py            σ per feature: std | MAD | MAD+tolerance; strength z/σ
│  ├─ events.py           control_episodes, pass_chains, restart_kicks, ACC_KICK
│  ├─ rules/              P-DRIB P-FREE P-PASS P-SHAPE S-PHASE · D-KO D-HOLD D-WHISTLE D-GAP
│  │                      D-SP D-STILL · R-TOUCH R-KO · X-NOBALL  (+ team D-OUT, D-GOAL)
│  ├─ votes.py            w = q·s → log-odds evidence per class
│  ├─ pseudo_labels.py    NN targets from the rules only
│  ├─ frame_features.py   52 per-frame inputs × 5-frame window
│  ├─ nn.py               light MLP in numpy (no framework), H/A swap augmentation
│  ├─ hmm.py              Bakis-expanded HMM, time-varying transitions, Viterbi
│  ├─ engine.py           MOTORE: observation model ω, emissions, decoding per half
│  ├─ diagnostics.py      possessions, durations, splits, agreement — without ground truth
│  └─ submission.py       CSV writer + official validator
├─ scripts/  make_submission.py · eval_sigma_synthetic.py
└─ tests/    test_hmm_v0.py
```

It imports the team package `src/tikitalia` (Match, features, kick-offs, D-OUT, D-GOAL) and does
not modify it.
