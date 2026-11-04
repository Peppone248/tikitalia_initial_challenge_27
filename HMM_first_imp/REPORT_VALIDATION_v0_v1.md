# Validation of v0 and v1 on other games — IDSSE (30/10/2026)

**Dataset.** IDSSE: 7 full Bundesliga 2022/23 matches, DFL / Sportec optical tracking at 25 fps,
CC-BY 4.0 (Bassek et al., Scientific Data 12, 195, 2025; mirror huggingface.co/datasets/pysport/idsse-data).
**Labels** come from the DFL tracking itself, per frame: `BallStatus` 0 → D, `BallPossession` 1/2 →
H/A. No person labelled anything. This is **another game** (RULES §1 allows labelled data from
other games, documented). Nothing of BAR-ATH's ground truth is used.

```powershell
.\venv\Scripts\python.exe HMM_first_imp\validation\fetch_idsse.py      # download + convert, once
.\venv\Scripts\python.exe HMM_first_imp\scripts\run_validation.py      # ~30 min; --reuse rescoring only
```

## Setup

| | |
|---|---|
| tracking **degraded** | IDSSE made to look like the BAR-ATH automatic tracking, using BAR-ATH *tracking* statistics only: ball visible/missing stretches resampled from BAR-ATH's (→ 42–50% missing), ball higher than 2.5 m always missing, player tracks broken like the BAR-ATH slots (~7% missing). Same degraded tracking for every version (paired comparison) |
| tracking **clean** | IDSSE as provided: the ceiling of the logic |
| labels **dfl** | as provided: very fine-grained, ~437 possessions per match, mean 7.2 s |
| labels **controlled** | CHALLENGE.md applied mechanically: possession flips shorter than 1 s (= `T_CONTROL`) between two spells of the other team are touches, not changes. Mean 8.2 s |
| scorer | `evaluation/possession_metrics.evaluate`, the official one; pooled over 7 matches (~640 minutes, ~2,700–3,000 possessions) with a 1000-draw bootstrap |

**Caveat.** DFL possession is more fine-grained than the challenge ground truth (public window:
mean 14.2 s against 7–8 s here). On IDSSE, spellF1 rewards systems that cut possessions more often
than BAR-ATH probably does. Dead ball (BallStatus) is a much closer match to the challenge's
"dead ball", so D share and D F1 are the most trustworthy numbers.

## Results (pooled, 7 matches)

| tracking · labels | version | **spellF1** [90% CI] | macroF1 | D F1 | alive/dead F1 | team acc | possessions pred / true | D share pred / true |
|---|---|---|---|---|---|---|---|---|
| degraded · controlled | v0 | **0.289** [0.274–0.304] | 0.642 | 0.646 | 0.801 | 0.743 | 1651 / 2705 | 0.27 / 0.45 |
| degraded · controlled | v1 | 0.256 [0.241–0.271] | **0.716** | **0.822** | **0.882** | 0.725 | 1117 / 2705 | 0.35 / 0.45 |
| degraded · dfl | v0 | **0.266** [0.252–0.279] | 0.641 | 0.646 | 0.801 | 0.741 | 1651 / 3061 | 0.27 / 0.45 |
| degraded · dfl | v1 | 0.230 [0.216–0.243] | **0.716** | **0.822** | **0.882** | 0.723 | 1117 / 3061 | 0.35 / 0.45 |
| clean · controlled | v0 | **0.450** [0.434–0.465] | 0.799 | 0.833 | 0.872 | 0.803 | 2224 / 2705 | 0.42 / 0.45 |
| clean · controlled | v1 | 0.417 [0.399–0.435] | **0.863** | **0.961** | **0.968** | 0.811 | 1406 / 2705 | 0.45 / 0.45 |

Paired spellF1 difference v1 − v0 on the same tracking: **−0.033 (degraded), v0 better on 7/7
matches**; −0.032 clean (6/7). Per-match tables: `outputs/validation/results.md`.

## What it says

1. **Dead ball: v1 is clearly better.** D F1 rises from 0.65 to 0.82 (degraded) and from 0.83 to
   0.96 (clean), and the D share from 27% to 35% of a true 45%. Steps 5 (D-WHISTLE as anchor +
   D-HOLD) and the new NN did what they were meant to. D is still under-detected on degraded
   tracking (35% vs 45%).
2. **Possessions: v1 merges too much.** It predicts 1,117 possessions against 2,705 true (v0: 1,651).
   That costs about 0.03 spellF1, consistently: v0 wins on 7/7 matches. Same reading as the
   public window (12 vs 17). The team gate (step 1) blocks too many real changes. Both
   versions under-segment; v1 more.
3. **The tracking noise is the biggest cost:** clean → degraded loses ~0.16 spellF1 and ~0.15
   macroF1 for both versions. Improving the data (ball gaps) is worth as much as all the logic.
4. **v1's dead-ball part should be kept and the team gate loosened.** This is exactly the kind of
   choice the validation set exists for: it can be tuned **on IDSSE** (other games: allowed), not on
   the 5 public minutes. Primary metric: spellF1 on degraded · controlled, with the D numbers
   watched alongside. Because DFL possession is finer than ours, prefer the more conservative of
   two settings that score within the noise.

## Next experiment (one command per setting)

```powershell
.\venv\Scripts\python.exe HMM_first_imp\scripts\run_validation.py --variants degraded --versions v1 --set v1:hmm.hmm.team_change.closed_factor=5 --out-dir outputs\validation_cf5
```
Candidates: `closed_factor` 1 (gate off), 3, 5, 10; `window_post_s` 1 → 2 s.

---

## Addendum (30/10 evening): the baseline on the same data, closed_factor 5, decisions

### The official baseline scored on the same degraded IDSSE tracking

`baseline50` = `starter/baseline.py` re-implemented on the arrays (nearest player + D where the ball
is not seen + majority vote ±50 frames). On BAR-ATH it is the published floor (spellF1 0.2433
full match). The comparison has to be made on the **same** data: our IDSSE numbers cannot be put
next to 0.2433 directly.

| degraded · labels | version | spellF1 [90% CI] | macroF1 | D F1 | team acc | possessions pred / true |
|---|---|---|---|---|---|---|
| controlled | baseline50 | 0.210 [0.199–0.221] | 0.462 | 0.389 | 0.551 | 2782 / 2705 |
| controlled | **v0** | **0.289** [0.274–0.304] | 0.642 | 0.646 | 0.743 | 1651 / 2705 |
| controlled | v1 | 0.256 [0.241–0.271] | **0.716** | **0.822** | 0.725 | 1117 / 2705 |
| dfl | baseline50 | 0.199 [0.188–0.209] | 0.461 | 0.389 | 0.550 | 2782 / 3061 |
| dfl | **v0** | **0.266** [0.252–0.279] | 0.641 | 0.646 | 0.741 | 1651 / 3061 |
| dfl | v1 | 0.230 [0.216–0.243] | **0.716** | **0.822** | 0.723 | 1117 / 3061 |

Paired over the 7 matches: **v0 − baseline50 = +0.079 spellF1 (controlled), better on 7/7**;
v1 − baseline50 ≈ +0.046. Relative to the floor: v0 is about +38%, v1 about +22%. macroF1 is +0.18
(v0) and +0.25 (v1). If the ratio carries over to BAR-ATH, v0 would sit around 0.30–0.33 against the
0.2433 floor: an estimate, not a measurement.

### closed_factor 5 (run on Beppe's PC)

v1 with `team_change.closed_factor` 5 instead of 20: spellF1 0.261 (controlled) vs 0.256, 1,143
possessions vs 1,117. **Almost no effect**: the team gate is not the main cause of v1's merging. The
other v1 changes reduce the changes too: the NN no longer sees the team (omega_nn), SOFT votes are
×0.25, and the turnover rules confirm few changes. Next: ablation one switch at a time on IDSSE.

### Decisions / direction agreed (to be discussed with the team)

1. **Baseline to beat = baseline50 on the same data** (IDSSE degraded, controlled labels):
   0.210. On BAR-ATH it remains 0.2433 on the full match, measured only by the organisers.
2. **Supervised learning on IDSSE (approach A): on hold.** Doubt: the BAR-ATH tracking is
   "dirtier" than any synthetic degradation, and DFL possession is finer than ours, so a model
   trained on IDSSE risks learning the wrong thing. It stays a later option for the D/alive part,
   where the labels match well.
3. **Fewer, more robust features** instead of 52 (see below).
4. **Next approach: C, segment first then assign the team**, with **B (baseline + our dead ball)** as
   a quick check: both evaluated on IDSSE against baseline50, v0 and v1.

### Feature set to reduce to (proposal)

Criteria: (1) defined even when the ball is not seen, or with an explicit fallback; (2) not
dependent on track identity (fragmented slots); (3) robust to outliers (medians, not means;
counts in radii, not exact distances); (4) relative to the attack direction (H/A symmetric).

| # | feature | why it is robust |
|---|---|---|
| 1 | ball seen / seconds since last seen | the fact itself, never NaN |
| 2 | ball out-of-pitch distance (clipped) | geometry, the strongest D signal |
| 3 | ball still for (s) | D before restarts |
| 4 | collective slowdown, all players (median / reference) | does not need the ball |
| 5 | slowdown of the faster team | "both teams stopped" = whistle |
| 6 | team of the nearest player, smoothed over 1 s (−1/0/+1) | the baseline signal, de-flickered |
| 7 | distance gap nearest H − nearest A (clipped ±10 m) | how contested |
| 8 | players within 3 m of the ball, H − A | counts survive slot swaps |
| 9 | players in the opponent half, H − A | team shape, works without the ball |
| 10 | forward speed of the teams, H − A (median) | who attacks, works without the ball |
| 11 | ST phase team (−1/0/+1) | automatic tag, available without tracking |
| 12 | kick-off formation flag | restart anchor |

Twelve inputs (×5 frames of context = 60 for the NN) instead of 52 (260). The pairs (H, A) become
differences, so H/A symmetry is built in. To be validated on IDSSE: same pipeline, only the
feature set switched.

### Update (30/10 night): approach C implemented

C (v2) = D from the v1 HMM + segment-then-assign, lambda 0.5: **spellF1 0.316** (degraded ·
controlled), +0.027 vs v0 (7/7), +0.106 vs baseline50. Details: [REPORT_C_v2.md](REPORT_C_v2.md).
