# HMM_first_imp — report v0 (30/10/2026)

First complete workflow from the raw data to an assessable CSV `frame,label`, with the MOTORE
implemented as a **light NN + Bakis Hidden Markov Model** over the rule votes. Nothing committed.

```text
dati grezzi ─► Match ─► features ─► events ─► regole ─► MOTORE ─────────────► CSV
               misure    misure      fatti      opinioni  NN light + Bakis HMM  decisione
               pulite    derivate    nel tempo  pesate    (Viterbi per tempo)   unica
```

**Command** (repository root): `python HMM_first_imp/scripts/make_submission.py` →
`HMM_first_imp/outputs/run/tikitalia.csv`. Valid (official validator), 25 s, reproducible bit for
bit (two runs → same SHA-256).

| | |
|---|---|
| What exists now | events (`control_episode`, pass chains, `restart_kick`), 16 rules (12 new + team D-OUT/D-GOAL reused), MAD noise in the vote weight, NN, Bakis HMM, CSV writer, validation, tests |
| What it is not | a tuned system. Every number is a priori or estimated from the tracking without labels. Chapter 7 of STATO_LAVORI (repo, document) is out of scope as requested |
| Read first | §2 (rules compliance), §5 (the MAD decision), §8 (how the HMM works), §10 (what to expect) |

---

## 1. What was asked → where it is

| Request | Where | Status |
|---|---|---|
| First CSV for the Initial Challenge | `outputs/run/tikitalia.csv` | ✅ 199,319 rows, valid |
| Rules for H/A (and X) | `hmm_v0/rules/possession.py`, `noball.py`, `restart.py` | ✅ P-DRIB, P-FREE, P-PASS, P-SHAPE, S-PHASE, R-TOUCH, R-KO, X-NOBALL |
| Finish the features / missing events | `hmm_v0/events.py`, `frame_features.py` | ✅ `control_episode`, pass chains, `restart_kick`, `ACC_KICK`; 52 NN inputs |
| Missing D rules (STATO 4.2) | `hmm_v0/rules/dead_extra.py` | ✅ D-KO, D-HOLD, D-WHISTLE (A), D-GAP, D-SP, D-STILL |
| MOTORE = HMM on tracking / Smart Tagging | `hmm_v0/hmm.py`, `engine.py`, `nn.py` | ✅ |
| spellF1 first: unbroken possession sequences | Bakis minimum durations, XH/XA states, P-PASS | ✅ §8, §9 |
| MAD for the noise in w, evaluated on synthetic data vs the previous definition | `hmm_v0/noise.py`, `scripts/eval_sigma_synthetic.py` | ✅ MAD wins → adopted (§5) |
| Everything configurable | `config/*.yaml`, `--set key=value` | ✅ |
| No commit; report | this file | ✅ |

## 2. Rules of the challenge — checklist

| Rule (RULES.md) | How this pipeline respects it |
|---|---|
| 1. No hand labels on the provided game, for any purpose | The NN is trained on **pseudo-labels produced by the rules** (`pseudo_labels.py`): an algorithm, no person decides any label. No human label anywhere |
| 1. No rules/constants tied to specific frames | No frame number in the code (checked with a grep for integer literals). Kick-off, half starts, restarts are **detected**; `half_video_blocks` comes from `games.py` as in the team code |
| 2. No private GT, no human data about the game | Nothing in `hmm_v0/` reads `evaluation/gt/` (grep). Smart Tagging is automatic (FAQ: allowed) |
| 2. Public 5 minutes only to check the pipeline | Scored **once at the end** (§11), as a correctness check. No parameter was chosen on it (§11 documents the one change made after seeing it, and why it is not tuning) |
| 3. The repository regenerates the same file | Deterministic: fixed seed, numpy only, no GPU/framework. Two runs → identical SHA-256. Pin versions (`requirements.txt` note) |
| Data not redistributed | All outputs under `HMM_first_imp/outputs/` (gitignored by the root `outputs/` pattern); the ATD cache is the team one in `outputs/cache/` |
| Organisers can inspect and run it | Plain Python + numpy + PyYAML; this report + docstrings explain every layer |

## 3. Layout and how the team code is reused

`HMM_first_imp/` imports `src/tikitalia` (Match, `features.py`, kick-offs, attack direction, D-OUT,
D-GOAL) and **does not modify it**: the team tests still pass (39/39). Configuration is merged:
`src/tikitalia/config/params.yaml` + `config/params_hmm.yaml` (same schema, validated by the team
loader: value, unit, origin, why). No team parameter is overridden.

```text
config/params_hmm.yaml   new thresholds (R_CONTROL, T_CONTROL, V_PASS, ACC_KICK estimated, …)
config/rules_hmm.yaml    registry: class, kind, scope, stage, prior_q, params, options
config/hmm.yaml          noise mode, votes, pseudo-labels, NN, HMM topology/transitions/ω
config/pipeline.yaml     game, paths, cache, output folder, team name
```

## 4. Events (`events.py`) — STATO 3 rows done

| Event | Definition | BAR-ATH |
|---|---|---|
| `control_episode` | nearest player of the team ≤ `R_CONTROL` (1.5 m), nearer than every opponent, relative speed ≤ `V_REL_CONTROL` (2 m/s, NaN = not blocking), ball inside the pitch; gaps ≤ 0.2 s bridged; kept if ≥ `T_TOUCH` (0.2 s) | 792 (296 ≥ `T_CONTROL` = 1 s) |
| pass chain | consecutive control episodes of one team, each received by another player (other slot or ≥ `D_PASS_MIN` = 5 m away) within `T_PASS_MAX` (3 s); an opponent episode breaks it | 148 chains |
| `restart_kick` | ball still ≥ `T_STILL_RESTART` (1 s), then moving; strength = acceleration / `ACC_KICK` | 42 (+ 6 kick-offs) |
| `ACC_KICK` | origin *distribution*: 0.9 quantile of the ball acceleration on visible graded frames | 18.3 m/s² |

## 5. Noise ("rumore") with MAD in the weight w — evaluation and decision

### 5.1 What changed

NOMENCLATURA §2 defined σ as the **standard deviation** of the residual (raw − moving average),
and the team code composed the σ of derived features (distances, speeds) from the position σ,
which sits on the 0.10 m floor. Now (`noise.py`) σ is measured **on each feature the conditions
read**, as a robust MAD:

```text
residual = x − moving_average(x, 5)          on every finite stretch of the feature
σ_i      = 1.4826 · MAD(residual)            mode "mad" (adopted)
z_i      = m_i / σ_i ;  s = clip(min_i z_i / Z_SAT, 0, 1) · d ;  w = q · s
```

Three modes are available in `hmm.yaml → noise.mode`: `std` (previous definition), `mad`,
`mad_tol` (= max(MAD, semantic tolerance), the STATO 2.1 proposal).

### 5.2 Synthetic evaluation (`scripts/eval_sigma_synthetic.py`)

Invented data, known truth. The control condition "nearest player within R_CONTROL", observed
distance = true distance + white jitter (smoothed upstream, as the provided tracking) + slow
semantic offset (centroid vs feet) + outliers (nearest-player switches / tracker jumps of 1–6 m
for 1–10 frames). Activations as in a rule (condition held ≥ 1 s). 7 scenarios × 3 seeds.

| scenario | mode | σ error vs sensor noise | σ with outliers / σ clean | Brier of Φ(z) |
|---|---|---|---|---|
| Gaussian, clean | std / **mad** | 1.89 / **0.68** | 1.0 / 1.0 | 0.0015 / **0.0013** |
| Gaussian, 2% outliers | std / **mad** | 22.1 / **0.99** | 7.96 / **1.18** | 0.0187 / **0.0030** |
| Gaussian, 5% outliers | std / **mad** | 34.7 / **1.63** | 12.3 / **1.56** | 0.0358 / **0.0058** |
| smoothed + semantic, 2% outliers | std / **mad** | 116 / **23** | 4.86 / **1.17** | 0.0337 / **0.0292** |
| smoothed + semantic, 5% outliers | std / **mad** | 181 / **31** | 7.55 / **1.54** | 0.0466 / **0.0325** |

Full table (with `mad_tol`, log-loss, saturation): `outputs/sigma_eval/sigma_eval.md`.

**MAD beats the previous definition in 7/7 scenarios** on the σ error, 7/7 on stability under
outliers, 7/7 on the Brier score (mean 0.0146 vs 0.0244); on log-loss 4/7 (std wins only where the
outliers inflate it into accidentally covering the semantic offset — an effect that changes with
the outlier rate, i.e. not something to rely on). `mad_tol` is better calibrated only in log-loss
on the semantic scenarios and worse in Brier on all of them. **Decision: `noise.mode: mad`.**

### 5.3 On BAR-ATH

| feature | std of residual | MAD σ | std / MAD |
|---|---|---|---|
| nearest_dist | 0.232 m | 0.0145 m | 16 |
| rel_speed | 0.351 m/s | 0.037 m/s | 10 |
| ball_speed | 0.286 m/s | 0.032 m/s | 9 |
| pressure_dist | 0.565 m | 0.004 m | 157 |
| dist_outside | 0.049 m | 0.009 m | 6 |

The residuals are **heavy-tailed** (std/MAD 6–157, Gaussian would be 1): exactly the
contaminated regime where std is wrong. **Open point, unchanged by this choice:** the MAD σ is
tiny (the tracking is pre-smoothed), so z still saturates often (P-DRIB s = 1 on 57% of its
frames, D-OUT 82%). The strength discriminates through the margin only near the threshold and
through the duration factor. If the call on STATO 2.1 decides for semantic tolerances, it is one
line: `--set hmm.noise.mode=mad_tol` (tolerances already in `hmm.yaml`).

## 6. Rules (`config/rules_hmm.yaml`)

Strength from the MAD z-scores (§5); w = q · s; q = `prior_q` (prior, REGOLE 4.1).

| rule | kind | what (summary) | coverage of graded frames | mean w | output agrees |
|---|---|---|---|---|---|
| D-OUT (team) | ANCHOR | ball beyond a line | 2.1% | 0.86 | 100% |
| D-GOAL (team) | ANCHOR | kick-off back to the goal | 4.4% | 0.48 | 100% |
| **D-KO** | ANCHOR | kick-off formation; from half start to the first kick-off; strength tapers in the last 1 s (the detected kick can be late) | 4.1% | 0.94 | 100% |
| **D-HOLD** | ANCHOR, stage 2 | after D-OUT/D-GOAL until restart kick, or a control ≥ 1 s **with live tempo**, or 90 s; strength decays (τ = 30 s) | 20.2% | 0.37 | 99% |
| **D-SP** | ANCHOR | ST SET PIECES + ball still ≥ 1 s → D until the restart kick | 0.9% | 0.90 | 98% |
| **D-WHISTLE** (A) | SOFT | both teams ≤ 0.5 × their tempo ≥ 1 s and ball still or unseen | 1.4% | 0.50 | **8%** |
| **D-GAP** | SOFT | ball missing ≥ 10 s and collective slowdown | 0.6% | 0.43 | – |
| **D-STILL** | SOFT | ball at rest ≥ 2 s while play is slower than usual (tip "speed ≈ 0 for several frames") | 1.2% | 0.50 | 100% |
| **P-DRIB** | ANCHOR | control episode ≥ `T_CONTROL` | 9.7% | 0.73 | 95% |
| **P-FREE** | ANCHOR | own player ≤ 1.5 m, opponents ≥ 8 m, ball ≤ 5 m/s, ≥ 1 s | 4.4% | 0.74 | 91% |
| **P-PASS** | ANCHOR | ≥ 2 receptions in a row: labels the **whole chain, passes in flight included** | 10.7% | 0.85 | 98% |
| **P-SHAPE** | SOFT | players in the opponent half + forward speed, team vs team | 93% | 0.38 | – |
| **S-PHASE** | SOFT | ST phase codes with team (possession → team, defending → other) | 59% | 0.50 | 69% |
| **R-TOUCH** | RESTART | nearest team in the 0.5 s before a restart kick | 0.6% | 0.10 | – |
| **R-KO** | RESTART | team at the centre spot at the kick-off | 0.1% | 0.95 | 100% |
| **X-NOBALL** | ANCHOR | ball not tracked ≥ 0.5 s → observation "X" (a fact, see §8) | 41% | 0.79 | – |

"Output agrees" = share of the frames where the rule votes with w ≥ 0.5 and the final label is the
same (no ground truth). Also fixed: the `emit()` overlap item of STATO 4.1 — `SafeOutput` keeps the
stronger label on overlaps and counts them, instead of silently mixing them.

## 7. From votes to evidence (`votes.py`)

```text
w_r(f) = q_r · s_r(f)
p_r(f) = 1/4 + 3/4 · w_r(f)                   w = 0 → chance (4 classes), w = 1 → certain (cap 0.99)
e_r(f) = ln(3 p / (1 − p)) · kind_scale       optimal log-odds weight (NOMENCLATURA 4)
E[f,c] = Σ_r e_r(f) over the rules voting c
```

- **kind_scale** SOFT = 0.25: SOFT rules vote continuously (P-SHAPE on 93% of frames); summed
  frame by frame they outvoted every anchor in the first run. Scaled, they fill gaps and break
  ties (REGOLE 6: "a change sustained only by SOFT votes is not accepted").
- **DEAD context**: where any D vote has w ≥ 0.5, OPEN_PLAY rules (P-*, S-PHASE) are muted —
  e.g. a player holding the ball for a throw-in is not P-FREE.

## 8. The MOTORE: NN + Bakis HMM

### 8.1 Light NN (`nn.py`, `frame_features.py`, `pseudo_labels.py`)

- **Inputs:** 52 measurements per frame (ball visibility, seconds missing, speed, `ball_fast` =
  speed > 10 m/s (human max), `ball_static`, acceleration, distance from the lines, collective
  tempo, nearest/pressure/relative speed/closing/chase per team, shape per team, ST set piece and
  ST phase team, kick-off formation). **No rule output** enters the input.
- **Context window:** frames t−2 … t+2 concatenated → 260 inputs (tip 1). Original 25 fps (tip 4).
- **Targets = pseudo-labels from the rules:** H/A/D where the rule evidence is clear (best ≥
  1 nat, margin ≥ 1.5 nats); **X only where the ball is not seen and nothing says D or a team**.
  So the NN's X means *live play, ball unseen*, and occluded stoppages become D targets (via
  D-HOLD, D-GOAL…), letting the NN learn what an unseen stoppage looks like from the players.
  BAR-ATH: H 18k, A 8k, D 22k, X 47k frames, 64k unlabelled.
- **Network:** 260 → 48 → 24 → 4, ReLU, softmax, class-balanced cross-entropy, Adam, 6 epochs, 80k
  stratified frames, seed 27, plain numpy. **H/A swap augmentation** (every frame also shown with
  the teams exchanged) and a symmetric prediction: the NN cannot learn "home is more likely".
  It reproduces its pseudo-labels on 98.7% of training frames; its value is on the frames no rule
  covers.

### 8.2 Hidden states and the meaning of X (tip 6)

`H, A, D, XH, XA`. XH/XA = *the team still has the ball but the tracker cannot see it*. A single X
state would forget who had the ball, and `H … X … H` would risk becoming `H … A`. With XH/XA the
wildcard keeps the memory, and the output maps XH → H, XA → A: **never X on a graded frame**
(IMPROVEMENTS A1), X on all non-graded frames.

### 8.3 Bakis topology (tip 2) — where it applies

A strict left-to-right model over a whole half is impossible: possession goes H → A → H. Bakis is
applied **inside each state**, as in a word model: state m is a chain of
`n_m = min_duration_s[m] · 25` sub-states, `i → i` or `i → i+1`, no backward jump, exit only from
the last one. Every visit therefore lasts at least `min_duration_s`:

| state | min duration | why (a priori) |
|---|---|---|
| H, A | 0.8 s (20 sub-states) | no 1-frame possessions (spellF1: flicker = spurious possessions) |
| D | 2.0 s (50) | no restart takes less; a shorter D between two spells of the same team only cuts one possession in two (`H D H`) |
| XH, XA | 1 frame | occlusions start and end anywhere |

### 8.4 Transitions (tip 5) — `hmm.yaml mean_time_s`

Written as mean times (p = 1/(t·25) per frame), missing pairs are forbidden:

| from \ to | H | A | D | XH | XA |
|---|---|---|---|---|---|
| H | – | 20 s | 30 s | *auto_visible* (22.9 s) | 200 s |
| XH | *auto_missing* (15.7 s) | 20 s | 30 s | – | 60 s |
| D | 60 s ×50 at restarts | 60 s ×50 | – | 120 s ×50 | 120 s ×50 |

- H → A is allowed directly (clean interception) but it costs a jump; H → XA (change while
  unseen) is rarer.
- **D → H/A requires a restart**: at detected restart kicks and kick-offs (± 0.5 s) the exits
  from D are multiplied by 50 (time-varying transitions). D persists otherwise.
- `auto_visible` / `auto_missing`: mean length of the stretches with / without the ball on the
  graded frames, estimated from the tracking at run time (origin: distribution, no labels).

### 8.5 Emissions — observation model ω

Each hidden state emits one of the 4 observation classes with probability ω (rows sum to 1):

| state | ω rules | ω NN |
|---|---|---|
| H | H .95, X .05 | H .95, X .05 |
| D | **D .2, X .8** | D .95, X .05 |
| XH | X .8, H .2 | X .8, H .2 |

```text
log b_m(f) = 0.2 · log Σ_c ω_nn[m,c] P_nn(c|f)/prior(c)  +  0.2 · log Σ_c ω_rules[m,c] e^{E(c,f)}
```

- For the rules, X = "ball not tracked": **uninformative between XH, XA and D** (same 0.8), so
  losing the ball votes neither for nor against a stoppage; D vs team evidence and the transitions
  decide. For the NN, X = "live, unseen" and points to XH/XA.
- 0.2 = 1/(2k+1): consecutive frames are not independent evidence (one independent observation
  per 5-frame window).

### 8.6 Decoding

Viterbi once per half on the graded frames (memory never crosses half-time), 92 expanded states,
~3 s per half. Changes are dated by the evidence itself (a control votes from its first frame), not
by when the decoder becomes sure (A2): in the synthetic test a change lands within one frame of
the true one (one frame is free for spellF1).

## 9. How the result was reached — iterations driven by checks without ground truth

The diagnostics in `run_summary.json` (no GT) drove four design corrections, not parameter tuning:

| run | symptom (no GT) | cause | fix |
|---|---|---|---|
| 1 | D 12%, median stoppage 2.8 s, D-HOLD agrees 25% | X competed with D; SOFT shape/phase votes accumulated over seconds | observation model ω (X compatible with D); SOFT ×0.25; DEAD context from any confident D vote |
| 2 | X pseudo-labels = 0 | X-NOBALL was SOFT, scaled below the target threshold → NN learnt "unseen ⇒ D" | X-NOBALL is ANCHOR (it is a fact) |
| 3 | D 9%, D-HOLD 17%, D-GOAL 31% | X more likely under XH (.8) than D (.5): the unseen ball voted against stoppages | same weight on X for XH/XA/D |
| 4 | XH/XA 1.3%: almost every occlusion = D | "XH → H in 1 s" prior; real gaps last ~16 s | `auto_missing` / `auto_visible` from the tracking; X targets only without D/team evidence; two ω (rules vs NN) |
| 5 | 34 `H D(<2 s) H` splits; D-HOLD ending on a player carrying the ball | D min 0.4 s; any control ended the hold | D min 2 s; hold ends only on a control with live tempo |

## 10. The CSV — what it looks like and what to expect

Final run (`outputs/run/run_summary.json`, no ground truth involved):

| | |
|---|---|
| graded frames | 158,857 — H 45.9%, A 31.4%, D 22.7%, X 0 |
| possessions | 321 (158 + 163), median 9.1 s, mean 15.3 s, 12 shorter than 1 s |
| dead spells | 151, median 5.4 s, mean 9.5 s |
| H ↔ A changes without a D | 171 |
| same-team possessions split by a D < 2 s | **0** |
| hidden-state share | H 28.8%, A 16.5%, D 22.7%, XH 17.1%, XA 14.9% |

**Expectations, honestly:**
- It should beat the published floor (baseline_smooth50, spellF1 0.24 on the full match) because
  it no longer flickers (minimum durations) and keeps possessions through occlusions and passes;
  the size of the gain is unknown until a validation set exists (IMPROVEMENTS B3).
- **D is probably under-estimated.** Ball-in-play in top football is ~55 of ~100 minutes, so a
  priori D could be ~40%+ of graded frames, against our 23%; stoppages also look short (median
  5.4 s). Causes: few restart kicks detected (42 — the ball is often unseen at restarts),
  D-WHISTLE overruled (8% agreement), no D-SHOT.
- H/A changes come only from controls of the new team (P-DRIB/P-FREE/P-PASS) and from the NN: no
  TURNOVER rule yet, so a change during a long occlusion is placed when the ball reappears.

## 11. Public 5 minutes — pipeline check only

Scored once with `evaluation/run_evaluation.py` (RULES 2: correctness, never selection):

| file | macroF1 | spellF1 | possessions | mean duration |
|---|---|---|---|---|
| hmm_v0 (final) | 0.720 | 0.369 | 22 (GT 17) | 11.9 s (GT 14.2) |

Conclusion: frames aligned, H/A not inverted, D in the right places, possession count of the
right order. **Not a ranking signal:** 17 possessions, one possession moves spellF1 by ~0.06
(EVALUATION.md). Transparency: an earlier run with the NN trained on 120k frames read 0.411 on
this window; the training subsample was then lowered to 80k **for memory on the team laptops**,
which reads 0.369 — the lower one was kept, and the gap is one possession, i.e. noise.

## 12. Limitations and next steps

Ordered by expected effect on spellF1:

1. **Validation set on other games (IMPROVEMENTS B3).** Without it no version can be compared
   with another; it also gives real `q` per rule (M1) and a calibration of z → p (NOMENCLATURA 4).
2. **D detection:** restart kicks with the ball unseen (reappearance after a D, player outside the
   line for throw-ins = R-THROW), D-WHISTLE variant B, D-SHOT, short D-OUT plausibility.
   D-WHISTLE agreement 8%: check what it fires on.
3. **TURNOVER rules** (T-REV, T-INTERCEPT, T-TEAMFLIP) as time-varying H↔A transition boosts,
   exactly like the restart boost for D.
4. **Strength saturation** with MAD σ (§5.3): decide STATO 2.1 (`mad_tol` is ready) or calibrate
   p(z) on other games.
5. **HSMM / forward-backward:** explicit duration distributions instead of minimum durations,
   per-frame posteriors for `X` on truly ambiguous transitions and for the report.
6. **Label model** (Dawid–Skene) for `q` from agreements instead of priors.
7. Known data issues untouched: referee/duplicate slots in `nearest_*` (24 + 22 slots), speed NaN
   on ~5% of frames, the 1st-half kick-off detected ~0.7 s after the first graded frame.

### Proposed STATO_LAVORI updates (not applied: outside HMM_first_imp)

§3 `control_episode` ✅ · `restart_kick` 🟡 (still ball only) · `pass_event` 🟡 (chains) — §4.1
`emit()` ✅ (SafeOutput) — §4.2 D-KO ✅, D-HOLD ✅, D-WHISTLE A 🟡, D-GAP 🟡, D-SP ✅ — §4.3 P-DRIB ✅,
P-FREE ✅, P-PASS ✅, P-SHAPE ✅ — §4.5 R-TOUCH 🟡, R-KO ✅ — §4.6 S-PHASE ✅ — §5 votes ✅, inertia →
HMM with minimum durations ✅, retro-dating ✅ (by construction), no X on graded ✅, engine params in
YAML ✅, tests ✅, HSMM/Viterbi (v1) → Viterbi done, HSMM ⏸ — §6 writer ✅, single command ✅,
validator ✅, public check ✅ — §2.1 noise: MAD adopted on synthetic evidence, tolerances pending.

## 13. Reproduction

```bash
python -m pip install -r HMM_first_imp/requirements.txt     # numpy, PyYAML, pandas
python HMM_first_imp/scripts/make_submission.py               # → HMM_first_imp/outputs/run/tikitalia.csv
python -m unittest discover -s HMM_first_imp/tests -v         # 14 tests, synthetic
python HMM_first_imp/scripts/eval_sigma_synthetic.py          # §5 table
```

- Data: `data/tactical/BAR-ATH_tactical_{atd.txt,atd.xml,smart_tagging.xml}` (or `MTAH_DATA_DIR`);
  the parsed ATD is cached in `outputs/cache/atd_arrays.npz` (team cache, reused).
- Deterministic on one machine (verified). Across machines, BLAS/numpy versions can change the NN
  in the last bits: pin the versions you used (`pip freeze`) before submitting, and regenerate the
  submitted file with them.
- Runtime ~25 s, peak RAM ~1.2 GB (mostly the per-slot feature arrays of the team code).
