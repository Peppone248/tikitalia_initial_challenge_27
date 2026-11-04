# Approach C (v2) — segment first, then assign the team (30/10/2026)

```powershell
.\venv\Scripts\python.exe HMM_first_imp\scripts\make_submission.py --config-dir HMM_first_imp\config_c   # -> outputs\run_c\tikitalia.csv
```

Code: `hmm_v0/segment_c.py` (+ `engine: segment` in `config_c/pipeline.yaml`). Parameters:
`config_c/segment.yaml`, each with its a-priori reason. Tests: `tests/test_hmm_v0.py::TestSegmentC`.

## 1. Idea

spellF1 judges **possessions as segments**, not frames. v0/v1 decide frame by frame (an HMM with
minimum durations); C decides by segment:

```text
1. DEAD BALL     D mask (from the D rules, or from the v1 HMM), cleaned: gaps < 1 s closed,
                 D < 2 s dropped, alive islands < 1 s absorbed
2. BOUNDARIES    in every alive stretch, possession may change ONLY at an observable event:
                 start of a contact (any team, >= T_TOUCH), of a turnover (T-INTERCEPT, T-REV),
                 of a restart (R-TOUCH, R-KO), of an uncontested control (P-FREE), first frame the
                 ball is seen again after a gap >= 0.5 s
3. ASSIGNMENT    each elementary segment gets its summed team evidence S (seconds of
                 full-strength evidence, > 0 = home): contacts x strength + signed rule votes
                 (turnovers, restarts, pass chains, Smart Tagging, shape) + nearest player
                 weighted by how uncontested. A 2-label dynamic programme chooses H/A for all the
                 segments of the stretch: minimise  sum -S_i sign(L_i) + lambda x #changes
```

`lambda` (`change_penalty_s`) is the one knob of the number of possessions: a new team must show
`lambda` more seconds of clear evidence than the old one to be credited with a possession.

## 2. Technical choices

| choice | why |
|---|---|
| boundaries only at events | a change is always dated by something observable (a contact, a restart): no drift, A2 by construction |
| evidence summed over the whole segment | robust to the ball being unseen for seconds: the visible frames and the slow signals carry the segment |
| exact optimum (DP), one penalty | deterministic, no training, no randomness: the same CSV everywhere |
| no NN in the pure version | 8 s instead of ~70 s per run; reproducibility trivial |
| D from the v1 HMM as an option (`dead.source: hmm`) | the D of the rules alone is weak (F1 0.64); the v1 D is the best measured (F1 0.82) |
| touches (any contact) count, but weighted by length and strength | "a touch without control is not a change" (CHALLENGE.md) is the penalty `lambda`, not a hard rule |

## 3. Differences from v0 and v1

| | v0 | v1 | **C (v2)** |
|---|---|---|---|
| decision unit | frame (HMM) | frame (HMM) | **segment between events** |
| where possession may change | anywhere (min. 0.8 s) | anywhere, cheap only near a rule event (team gate) | **only at events** |
| who decides the team | NN + rules | rules (NN team-blind) | **rule evidence summed over the segment** |
| dead ball | NN + rules, HMM | NN + rules + D-WHISTLE anchor + D-HOLD, HMM | rules (pure) **or the v1 D** |
| NN | MLP, minibatch | random-feature net, convex | none (pure) / v1's (hmmD) |
| knobs on the count of possessions | min. durations, transition times | + closed_factor | **one: `lambda`** |
| runtime on BAR-ATH | ~25 s | ~70 s | **8 s** (pure) / ~70 s (hmmD) |
| reproducible across machines | no (1.3% frames differ) | yes | yes |

## 4. Validation on IDSSE (7 matches, degraded tracking)

Same degraded tracking for every version; official scorer; pooled over 7 matches.

### Round 1

Labels **controlled** (DFL possession, touches < 1 s removed):

| version | spellF1 [90% CI] | macroF1 | D F1 | team acc | possessions pred / true |
|---|---|---|---|---|---|
| baseline50 | 0.210 [0.199–0.221] | 0.462 | 0.389 | 0.551 | 2782 / 2705 |
| v0 | 0.289 [0.274–0.304] | 0.642 | 0.646 | 0.743 | 1651 / 2705 |
| v1 | 0.256 [0.241–0.271] | 0.716 | 0.822 | 0.725 | 1117 / 2705 |
| C, D from rules, lambda 1 | 0.223 [0.209–0.238] | 0.618 | 0.640 | 0.670 | 987 / 2705 |
| C, D from rules, lambda 0.5 | 0.232 [0.218–0.247] | 0.618 | 0.640 | 0.671 | 1063 / 2705 |
| C, D from rules, lambda 2 | 0.196 [0.182–0.210] | 0.613 | 0.640 | 0.662 | 876 / 2705 |
| **C, D from v1 (hmmD), lambda 1** | **0.304** [0.287–0.319] | **0.733** | **0.821** | **0.752** | 1205 / 2705 |

Paired over the 7 matches (same tracking): **C hmmD − v0 = +0.015, better on 6/7**;
**C hmmD − v1 = +0.048, better on 7/7**; C hmmD − baseline50 = +0.094, 7/7. With the raw DFL
labels: +0.007 vs v0 (5/7), +0.043 vs v1 (7/7).

Reading:
1. **The C team assignment is better than both HMM engines** on spellF1, macroF1 and team
   accuracy, *when it gets a good dead ball*.
2. **The pure C (D from the rules) is weak** because its D is weak (F1 0.64, like v0). The D is
   the bottleneck of the pure version, not the segmentation.
3. **A smaller lambda helps**; all versions still predict too few possessions. With the DFL labels
   finer than ours, the lambda must not be chosen too low (it would chase DFL's contact-level
   changes): pick the most conservative value within the noise.

### Round 2 — lambda sweep with the v1 D

| labels | lambda | spellF1 [90% CI] | macroF1 | team acc | possessions pred / true | paired vs v0 |
|---|---|---|---|---|---|---|
| controlled | 1.0 | 0.304 [0.287–0.319] | 0.733 | 0.752 | 1205 / 2705 | +0.015 (6/7) |
| controlled | **0.5** | **0.316** [0.300–0.332] | 0.734 | 0.753 | 1302 / 2705 | **+0.027 (7/7)** |
| controlled | 0.25 | 0.323 [0.307–0.339] | 0.733 | 0.752 | 1405 / 2705 | +0.034 (7/7) |
| dfl | 1.0 | 0.272 [0.257–0.286] | 0.732 | 0.751 | 1205 / 3061 | +0.007 (5/7) |
| dfl | **0.5** | **0.285** [0.270–0.299] | 0.733 | 0.751 | 1302 / 3061 | +0.020 (5/7) |
| dfl | 0.25 | 0.293 [0.278–0.306] | 0.732 | 0.751 | 1405 / 3061 | +0.028 (7/7) |

Paired step by step: 0.5 − 1.0 = +0.012 (6/7 controlled, 7/7 dfl); 0.25 − 0.5 = +0.007 (5/7
controlled, 6/7 dfl). **Chosen: lambda = 0.5.** 0.25 is better by less than the step before and on
fewer matches, and a lower lambda moves the system towards DFL's contact-level possessions, which
are finer than the challenge's: the rule fixed in advance was to take the most conservative value
within the noise.

**C (v2) default = D from the v1 HMM + lambda 0.5.** Against the same data:

| (degraded, controlled labels) | spellF1 | macroF1 | team acc |
|---|---|---|---|
| baseline50 | 0.210 | 0.462 | 0.551 |
| v0 | 0.289 | 0.642 | 0.743 |
| v1 | 0.256 | 0.716 | 0.725 |
| **C (v2)** | **0.316** | **0.734** | **0.753** |

C (v2) − baseline50 ≈ **+0.106 spellF1 (+50%)**; − v0 = +0.027 (7/7); − v1 ≈ +0.06.

## 5. BAR-ATH

```text
CSV         outputs/run_c/tikitalia.csv   SHA-256 23a6ad7c1f9cf49f59ed88941e810c66e2814b5928f3605b863305cf772092a7
runtime     ~65 s (the v1 NN + HMM for the dead ball; the segmentation itself takes < 1 s)
graded      H 45.6% · A 27.9% · D 26.5%
possessions 244 (v1: 205, v0: 321), median 12.7 s · dead spells 122, median 8.2 s
```

Public 5 minutes, **structure check only** (the choice of lambda was made on IDSSE before this
run; the window's noise band is ±0.36 spellF1):

| file | macroF1 | spellF1 | possessions (truth 17) | mean length (truth 14.2 s) |
|---|---|---|---|---|
| C (v2) | 0.802 | 0.459 | **17** | 15.0 s |
| v0 | 0.773 | 0.432 | 21 | 12.2 s |
| baseline_smooth125 | 0.581 | 0.424 | 16 | 18.5 s |
| v1 | 0.741 | 0.345 | 12 | 21.2 s |
| baseline_smooth50 | 0.621 | 0.326 | 26 | 10.8 s |

Consistent with IDSSE: the possession count is now of the right order (v0 too many, v1 too few).

## 6. Limits and next steps

1. **The dead ball is now the bottleneck** (D share 35% vs a true 45% on IDSSE; D F1 0.82). The pure
   C (D from rules) loses 0.08 spellF1 to the hmm D: a better rule-based D would remove the NN
   entirely (faster, simpler). Restarts with the ball unseen and short stoppages are the gap.
2. Too few possessions still (1302 vs 2705 on IDSSE): partly DFL's finer definition; partly
   changes during ball gaps, with no candidate boundary inside a gap. A T-TEAMFLIP candidate
   (both teams reverse direction) would add boundaries where the ball is unseen.
3. Feature reduction (12 robust features, REPORT_VALIDATION addendum) applies to the NN of the D
   part, the only NN left.
4. The weights of `segment.yaml` are a priori; a few of them can be checked on IDSSE the same way
   as lambda (one at a time, conservative choice within the noise).

## 7. Clean tracking: does C also work on good data?

Same 7 IDSSE matches with the tracking **as provided** (ball always seen, stable players), labels
controlled. `outputs/validation_clean/results.md`, `config/validation.yaml` (version `c_v2`).

| version | spellF1 degraded | spellF1 **clean** | ratio | macroF1 clean | D F1 clean | D share clean (true 0.45) | team acc clean | possessions clean (true 2705) |
|---|---|---|---|---|---|---|---|---|
| baseline50 | 0.210 | 0.326 | 0.64 | 0.397 | 0.000 | 0.00 | 0.838 | 2644 |
| v0 | 0.289 | 0.450 | 0.64 | 0.799 | 0.833 | 0.42 | 0.803 | 2224 |
| v1 | 0.256 | 0.417 | 0.61 | 0.863 | 0.961 | 0.45 | 0.811 | 1406 |
| **C (v2)** | **0.316** | **0.481** | **0.66** | **0.881** | **0.960** | **0.46** | **0.836** | 1554 |

Paired on clean: C − baseline50 = +0.155 (7/7); C − v0 ≈ +0.031 (6/7).

Reading:
1. **No sign of a system fitted to dirty data.** Every version gains ~0.16 spellF1 from clean
   tracking, C the most; the ranking is the same on both (C > v0 > v1 > baseline) and C keeps the
   largest share of its clean score on degraded data (0.66). Nothing was tuned on clean data.
2. **With a clean ball the dead ball is solved** (D F1 0.96, share 0.46 vs 0.45): on BAR-ATH the D
   deficit is a tracking problem (ball unseen), not a rule problem.
3. **Team: on clean data the nearest player is already as good as us** (baseline 0.838 vs C 0.836).
   Our gain over the baseline comes from the dead ball and the segment structure, not from a
   better "who has the ball". On degraded data the rules do add a lot (0.75 vs 0.55).
4. **Too few possessions even with perfect tracking** (1554 vs 2705, 14.1 s vs 8.2 s): the
   under-segmentation is structural, not caused by ball gaps. Partly DFL's finer definition — our
   14.1 s is close to the 14.2 s of the challenge's public window — partly lambda, which was chosen
   on degraded data and is not re-tuned per data quality.
