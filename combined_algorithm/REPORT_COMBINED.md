# Combined algorithm: what works, measured on external matches (04/11/2026)

All numbers: official scorer (`evaluation/possession_metrics.py`), spellF1 and macroF1, mean per match, leave one match out unless stated.
In brackets: matches better / worse. **The pipeline has not been run on the provided match yet**: that run belongs on the PC.

## 1. In short

| question | answer |
|---|---|
| Can we measure without a leaderboard? | Yes, approximately: 10 external matches with simulated tracker errors reproduce the pattern of the published baselines (section 2) |
| Does our own tracking help? | A lot for simple models (rules +0.11 / +0.16), a little for learned ones (+0.016 / +0.009) |
| What moved the score most? | Replacing the logistic regressions with gradient-boosted trees and new inputs: +0.076 (7/0) IDSSE, +0.047 (3/0) Metrica |
| Where are the points left? | Mostly in the team (changes of possession), then the dead ball |
| Video detector, whistle, broadcast cut? | No gain on this benchmark: off |

| | IDSSE (7) spellF1 / macroF1 | Metrica (3) spellF1 / macroF1 | rough reading for the provided match |
|---|---|---|---|
| official baseline_smooth50 | 0.282 / 0.699 | 0.293 / 0.726 | 0.243 (published) |
| rules, raw ATD | 0.290 / 0.599 | 0.308 / 0.621 | |
| rules, our tracking | 0.400 / 0.711 | 0.473 / 0.737 | |
| previous HMM (logistic), our tracking | 0.583 / 0.879 | 0.639 / 0.873 | 0.50-0.53 |
| **chosen: trees + new inputs + HMM, our tracking** | **0.659 / 0.904** | **0.686 / 0.886** | **about 0.57** |
| ceiling: clean tracking, trained on clean (trees) | 0.794 / 0.954 | 0.836 / 0.929 | |

The reading applies one ratio, published baseline / benchmark baseline, to the model. It assumes the model loses as much as the baseline
does on the real tracker, which nothing proves: on the 5 public minutes the previous learned dead ball did worse than on external matches
(REPORT_B 17, 23). Treat 0.57 as the top of a range of about 0.45-0.60, a personal estimate.

## 2. The benchmark

**Matches.** 7 IDSSE (Bundesliga, DFL labels) and 3 Metrica public sample matches: the organisers' own sample data, with labels built by code
from their event data (`bench/fetch_metrica.py`). Both are other games.

**What Metrica showed.** In these corrected files the ball is missing almost exactly while it is dead: dead spans start and end on a ball gap
(median offset 0 s), and a long gap with no on-ball event inside is a stoppage the events do not name. CHALLENGE.md says the ground truth comes
from corrected tracking, so this is a good picture of what "dead" means there: it starts when the ball leaves play and ends on the restart.
They also make 3.0-3.5 possessions per minute, against 3.4 in the public window and about 4 on IDSSE, so their idea of a possession looks closer.

**Scenarios** (`ca/scenarios.py`, a priori round values). `br7+err`: the ball lost mostly when dead (IDSSE: hidden on 70% of dead spans, 80% of live
gaps given back; Metrica: its own gaps, plus 20% of simulated live gaps), players broken into fragments, and tracker errors: position noise, keeper
swaps, frozen ghosts, duplicated players, fragments on the wrong team, false balls.

**Check against the four published baselines** (`bench/check_realism.py`, macroF1 / spellF1, seed 2027):

| scenario | ball seen (dead / alive) | raw | smooth25 | smooth50 | smooth125 |
|---|---|---|---|---|---|
| random gaps (the old `degraded`), IDSSE | 0.61 / 0.52 | 0.401 / 0.060 | 0.441 / 0.185 | 0.447 / 0.206 | 0.428 / 0.183 |
| sd5, IDSSE | 0.31 / 0.60 | 0.552 / 0.076 | 0.600 / 0.232 | 0.604 / 0.259 | 0.576 / 0.204 |
| sd8, IDSSE | 0.10 / 0.71 | 0.680 / 0.086 | 0.732 / 0.282 | 0.729 / 0.309 | 0.681 / 0.234 |
| br7+err, IDSSE | 0.19 / 0.79 | 0.666 / 0.062 | 0.708 / 0.257 | 0.699 / 0.282 | 0.637 / 0.210 |
| br7+err, Metrica | 0.15 / 0.92 | 0.737 / 0.051 | 0.740 / 0.255 | 0.726 / 0.293 | 0.668 / 0.256 |
| **published, provided match** | overall 0.59 | **0.653 / 0.049** | **0.675 / 0.200** | **0.672 / 0.243** | **0.636 / 0.224** |

- The old `degraded` is clearly not the provided match: macroF1 0.45 against 0.67. This settles the open question of REPORT_B 19 and 26 on how the ball is lost.
- `br7+err` has the same order as the published row: smooth25 best on macroF1, smooth50 best on spellF1. The provided match stays harder: spellF1 about 0.85 of the benchmark.
- Nothing was fitted to the published row. It is still a number derived from the private ground truth, used as a sanity check: to be confirmed with the organisers.

**Limits.** 10 matches, simulated tracker errors, labels that are not the organisers' mechanical ones. One degradation seed moves a score by about 0.02 (REPORT_B 19.5), and EVALUATION.md puts spellF1 noise near 0.04 on one match, so gains below about 0.005 are not decided here.

## 3. Our own tracking (`ca/own_tracking.py`)

Against the truth of the external matches, `br7+err` (`bench/track_quality.py`, IDSSE):

| | raw | ours |
|---|---|---|
| keeper-labelled points in the opposite box, live ball | 1.94% | **0.03%** |
| frames with a team over 11 players | 28.5% | 18.5% |
| frames with a team under 10 players | 53.1% | 35.9% |
| points on the wrong team | 5.2% | 4.4% |
| true players covered within 2 m | 89.0% | 90.9% |
| player position error, median | 0.18 m | 0.14 m |
| reported ball within 3 m of the truth | 98.4% | 98.8% |
| live ball seen | 78.5% | 79.4% |

Effect on spellF1, ours minus raw, seed 2027:

| model | IDSSE | Metrica |
|---|---|---|
| rules | +0.110 (7/0) | +0.164 (3/0) |
| logistic + HMM | +0.012 (6/1) | -0.002 (2/1) |
| trees + HMM | +0.016 (6/1) | +0.009 (3/0) |

So the idea that clean data makes simple models strong is right: the rules go from 0.29 to 0.40. But learned models had already learned to live with these errors. Cleaning stays on, since it costs nothing, and it is the part that is safest if the real errors are worse than the simulated ones.

The ceiling says there is more if the data were truly clean: 0.79 / 0.84. Most of that distance is information our cleaning cannot create: the ball it does not see, exact positions at the moment of a touch.

Long-gap interpolation is not used for control votes: a straight line is within 1.5 m only 19% of the time for gaps of 3-6 s (REPORT_B 15). Only gaps under 1 s are filled. For the learned team a line across any gap is an input, with its age beside it.

## 4. Models (`ca/trees.py`, `ca/features.py`)

- **Team:** gradient-boosted trees (500 rounds, 31 leaves), each frame also shown with the teams swapped and the output made exactly antisymmetric, so that "home has the ball more" cannot be learned.
- **New inputs:** team distance to an estimated ball at several scales, players within 5 / 10 / 20 m of it, closing speed on it, players ahead of it, role swaps over 5-10 s, age of the last ball and vote.
- **Dead ball:** trees on the rule layers, plus players tracked, how the running stops, fastest player.
- **Reading:** plain numpy from JSON, no BLAS and no threads, inputs rounded: LightGBM is needed only to train.
- **Trained on:** the three scenarios together (`br7+err`, `sd8+err`, `random+err`), since which one is nearest cannot be proved.

| comparison (spellF1) | IDSSE | Metrica |
|---|---|---|
| trees (200) vs logistic, seed 2027 | +0.066 (7/0) | +0.037 (3/0) |
| the same, **seed 2028, not used to choose** | +0.053 (7/0) | +0.032 (3/0) |
| 500 rounds vs 200, seed 2027 | +0.010 (6/1) | +0.016 (3/0) |
| 500 rounds vs 200, seed 2028 | +0.016 (6/1) | +0.022 (3/0) |
| trained on three scenarios vs one, tested on `br7+err` | +0.001 (5/2) | -0.006 (1/2) |
| three scenarios vs one, 200 rounds, tested on `sd8+err` | -0.004 (1/6) | -0.001 (2/1) |
| three scenarios vs one, 200 rounds, tested on `random+err`, the worst case | +0.015 (7/0) | -0.001 (2/1) |

**Transfer between data providers,** the closest thing to the real risk. Trained on IDSSE only, tested on Metrica: trees 0.671, logistic 0.637. Trained on Metrica only (3 matches), tested on IDSSE, before the Metrica labels were corrected: trees 0.505, logistic 0.472. The gain survives a change of provider.

**Decoding.** Emission weight 0.35-1.5 and minimum durations: within noise, opposite signs on the two groups (`results/decode_*`). Nothing changed: weight 1, minimum 0.8 s possession, 2 s stoppage.

## 5. Where the points are

With the model of section 4 (200 rounds), replacing one part with the truth, before the Metrica labels were corrected (so the Metrica row is indicative):

| | IDSSE | Metrica |
|---|---|---|
| model | 0.640 | 0.680 |
| perfect dead ball | 0.696 | 0.767 |
| perfect team | 0.920 | 0.883 |

The team is still the biggest margin; accuracy is 0.91 where the ball is seen and 0.86 where it is not (IDSSE).

## 6. Latency (`models/refine.json`, off)

Moving each change to its most likely frame lowers the latency but does not raise spellF1:

| | spellF1 IDSSE | spellF1 Metrica | mean abs. latency, frames |
|---|---|---|---|
| seed 2027 (mix, 500) | -0.005 (3/4) | +0.011 (3/0) | 7.1 -> 5.8 / 6.2 -> 3.4 |
| seed 2028 (br7, 500) | -0.001 (3/4) | -0.006 (1/2) | 6.6 -> 5.5 / 6.3 -> 3.3 |

Off, since the ranking is on spellF1. One line turns it on: `--set models.refine=models/refine.json`. The trees alone already lowered the latency against the logistic model on IDSSE: 8.7 -> 7.1 frames.

## 7. Video and audio

**Ball detector (`ca/ball_fusion.py`).** Chains of candidates that a ball could have made, instead of single detections. On a simulated detector they lift precision to 0.67 for a weak one and to 0.96 for a good one (mean of 7 matches; before the chains 0.34 and 0.80 on the one match checked). But on `br7+err`, with models not retrained on the filled ball (IDSSE):

| spellF1 vs no detector | used everywhere | used only for the team |
|---|---|---|
| weak detector (near the generic RF-DETR seen so far) | -0.149 (0/7) | -0.025 (0/7) |
| good detector | -0.125 (0/7) | -0.005 (2/5) |

Why: a learned dead ball reads "no ball" as dead, and a ball found inside a stoppage looks like play. And when the ball is lost mostly while dead, the live ball is already seen on about 80% of the frames, so there is less to win than the +0.04 per 10 points of REPORT_B 24, measured with random gaps. **Off.** Not measured: models retrained on a filled ball. It would have to win there before spending time on training a detector.

**Motion blobs (`ca/motion_ball.py`).** No weights, a priori constants, found the ball on a synthetic video only. Never run on a real one.

**Whistle and broadcast cut (`ca/side.py`).** Simulated channels, on the 200-round model and before the Metrica labels were corrected, give at most +0.0005 on IDSSE and +0.0004 on Metrica, because in these scenarios the dead ball is already found. This may undervalue them: on the provided match the dead ball looks harder. Off. `scripts/whistle_external.py` measures a real whistle on other matches, but needs audio and event times: **not run**. SoccerNet has both, under its own NDA.

## 8. Rules

- **RULES.md:** corrections come out of code, the same on every match, no frame or slot typed; no labels on the provided game; external data is public.
- **Mail of 03/11:** nothing is trained or tuned on the provided match, its video or its ground truth, and nothing of it left the PC for this work.
- **Open, and this work depends on them:**
  - question 4: the quantiles of ball gaps and the Smart Tagging simulation in `config/validation.yaml` come from the provided tracking. If the answer is no, they must be replaced by a priori values and the models retrained.
  - a new question: may the four published baseline scores be used as a sanity check of a simulation (section 2)?
  - question 2 (audio) and question 3 (detector): both stay off anyway.
- **Metrica sample data** has no formal licence file: its README asks for responsible use and attribution. Worth a line to the organisers, who publish it.

## 9. Limits and next steps

1. Run it on the provided match, then the public window **once**, as a structure check. If the D share looks far from the public window, the learned dead ball is again the suspect: the fallback is the rules' mask inside the HMM.
2. Most of the margin is in the changes: signals for a change with the ball unseen, then a second-stage model on the first-stage probabilities (not tried).
3. Tracker errors are simulated, so their real shape is unknown. Automatic checks on the provided tracking, run on the PC, can compare error counts with the simulation (team counts, jumps, frozen points) without using any label.
4. Models retrained on a filled ball, before any further work on a detector.

Sources: Bassek et al., Scientific Data 12, 195 (2025); Vidal-Codina et al., arXiv 2202.00804; github.com/metrica-sports/sample-data.
