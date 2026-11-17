# Results

## Current shipped model (kick-off rule + goalkeeper fix, 19 features) — 17/11/2026

Full leave-one-match-out table on the 7 IDSSE matches, seeds 2027 and 2028, code at commit fab8b43
(weights in `weights/tcn_idsse.npz`, trained with seed 2027 on all 7 matches).

- **Criterion: SODDISFATTO** (satisfied). tcn vs frame: +0.117 (seed 2027) and +0.089 (seed 2028),
  better on 7/7 matches in both seeds; tcn_change vs frame: +0.117 / +0.093, 7/7.
- Against the model before the two fixes (tcn 0.555 / 0.552): now 0.567 / 0.543, i.e. +0.012 and
  -0.009 — opposite signs, mean 0.555 vs 0.554: no systematic change. The degradation now also hides
  the ball in the keeper's hands, so the two tables are not exactly paired.
- The kick-off rule never fires on IDSSE (halves start at the kick-off), so this table measures the
  goalkeeper fix only; see "Kick-off rule" below for its evidence on the provided match.
- Absolute numbers are optimistic (the degradation hides the ball on true dead-ball runs and on true
  keeper holds); only relative comparisons count.

## Per model and seed (mean over matches)

| model | seed | spellF1 | macroF1 | n_pred_poss | n_true_poss | transF1_25 |
|---|---|---|---|---|---|---|
| baseline50 | 2027 | 0.244 | 0.578 | 354.1 | 387.9 | 0.300 |
| baseline50 | 2028 | 0.246 | 0.589 | 348.3 | 387.9 | 0.281 |
| frame | 2027 | 0.450 | 0.835 | 258.3 | 387.9 | 0.439 |
| frame | 2028 | 0.454 | 0.833 | 263.3 | 387.9 | 0.429 |
| tcn | 2027 | 0.567 | 0.870 | 297.0 | 387.9 | 0.535 |
| tcn_change | 2027 | 0.568 | 0.871 | 313.6 | 387.9 | 0.540 |
| tcn | 2028 | 0.543 | 0.862 | 301.3 | 387.9 | 0.506 |
| tcn_change | 2028 | 0.546 | 0.864 | 308.7 | 387.9 | 0.516 |

## Chosen (beta, w) per fold

| model | seed | match | beta | w |
|---|---|---|---|---|
| frame | 2027 | J03WMX | 0 | 0.5 |
| frame | 2027 | J03WN1 | 0 | 0.5 |
| frame | 2027 | J03WOH | 0 | 0.5 |
| frame | 2027 | J03WOY | 0 | 0.5 |
| frame | 2027 | J03WPY | 0 | 0.5 |
| frame | 2027 | J03WQQ | 0 | 0.5 |
| frame | 2027 | J03WR9 | 0 | 0.5 |
| frame | 2028 | J03WMX | 0 | 0.5 |
| frame | 2028 | J03WN1 | 0 | 0.5 |
| frame | 2028 | J03WOH | 0 | 0.5 |
| frame | 2028 | J03WOY | 0 | 0.5 |
| frame | 2028 | J03WPY | 0 | 0.5 |
| frame | 2028 | J03WQQ | 0 | 0.5 |
| frame | 2028 | J03WR9 | 0 | 0.5 |
| tcn | 2027 | J03WMX | 0 | 1 |
| tcn | 2027 | J03WN1 | 0 | 1 |
| tcn | 2027 | J03WOH | 0 | 1 |
| tcn | 2027 | J03WOY | 0 | 0.5 |
| tcn | 2027 | J03WPY | 0 | 1 |
| tcn | 2027 | J03WQQ | 0 | 1 |
| tcn | 2027 | J03WR9 | 0 | 1 |
| tcn_change | 2027 | J03WMX | 20 | 0.5 |
| tcn_change | 2027 | J03WN1 | 20 | 0.5 |
| tcn_change | 2027 | J03WOH | 0 | 1 |
| tcn_change | 2027 | J03WOY | 20 | 0.5 |
| tcn_change | 2027 | J03WPY | 20 | 0.5 |
| tcn_change | 2027 | J03WQQ | 20 | 0.5 |
| tcn_change | 2027 | J03WR9 | 20 | 0.5 |
| tcn | 2028 | J03WMX | 0 | 1 |
| tcn | 2028 | J03WN1 | 0 | 1 |
| tcn | 2028 | J03WOH | 0 | 1 |
| tcn | 2028 | J03WOY | 0 | 1 |
| tcn | 2028 | J03WPY | 0 | 1 |
| tcn | 2028 | J03WQQ | 0 | 1 |
| tcn | 2028 | J03WR9 | 0 | 1 |
| tcn_change | 2028 | J03WMX | 5 | 0.5 |
| tcn_change | 2028 | J03WN1 | 20 | 0.5 |
| tcn_change | 2028 | J03WOH | 20 | 0.5 |
| tcn_change | 2028 | J03WOY | 20 | 0.5 |
| tcn_change | 2028 | J03WPY | 20 | 0.5 |
| tcn_change | 2028 | J03WQQ | 20 | 0.5 |
| tcn_change | 2028 | J03WR9 | 20 | 0.5 |

## Paired spellF1 delta vs frame (per match)

### baseline50, seed 2027

| match | delta |
|---|---|
| J03WMX | -0.141 |
| J03WN1 | -0.149 |
| J03WOH | -0.246 |
| J03WOY | -0.244 |
| J03WPY | -0.273 |
| J03WQQ | -0.192 |
| J03WR9 | -0.196 |
| **mean** | -0.206 |

better 0/7, worse 7/7

### baseline50, seed 2028

| match | delta |
|---|---|
| J03WMX | -0.188 |
| J03WN1 | -0.146 |
| J03WOH | -0.210 |
| J03WOY | -0.215 |
| J03WPY | -0.231 |
| J03WQQ | -0.237 |
| J03WR9 | -0.227 |
| **mean** | -0.208 |

better 0/7, worse 7/7

### tcn, seed 2027

| match | delta |
|---|---|
| J03WMX | +0.150 |
| J03WN1 | +0.125 |
| J03WOH | +0.078 |
| J03WOY | +0.084 |
| J03WPY | +0.088 |
| J03WQQ | +0.137 |
| J03WR9 | +0.154 |
| **mean** | +0.117 |

better 7/7, worse 0/7

### tcn, seed 2028

| match | delta |
|---|---|
| J03WMX | +0.085 |
| J03WN1 | +0.099 |
| J03WOH | +0.083 |
| J03WOY | +0.105 |
| J03WPY | +0.048 |
| J03WQQ | +0.089 |
| J03WR9 | +0.115 |
| **mean** | +0.089 |

better 7/7, worse 0/7

### tcn_change, seed 2027

| match | delta |
|---|---|
| J03WMX | +0.142 |
| J03WN1 | +0.130 |
| J03WOH | +0.078 |
| J03WOY | +0.114 |
| J03WPY | +0.070 |
| J03WQQ | +0.135 |
| J03WR9 | +0.154 |
| **mean** | +0.117 |

better 7/7, worse 0/7

### tcn_change, seed 2028

| match | delta |
|---|---|
| J03WMX | +0.086 |
| J03WN1 | +0.100 |
| J03WOH | +0.075 |
| J03WOY | +0.111 |
| J03WPY | +0.061 |
| J03WQQ | +0.089 |
| J03WR9 | +0.129 |
| **mean** | +0.093 |

better 7/7, worse 0/7

Note: for frame and tcn the w selection, and for tcn_change the (beta, w) selection, use the 6 training matches of each fold predicted by the network trained on them (in-sample, no inner retraining, to bound CPU time); the held-out match is never used.

criterion: SODDISFATTO

---

## Previous model (before the kick-off and goalkeeper fixes)


- Criterion: SODDISFATTO. The plain TCN gains +0.103 spellF1 over the frame baseline, better on 7/7 matches in both seeds.
- The change head adds nothing: tcn_change is about equal to tcn (+0.101 / +0.103), so p_change is kept in probs.npz but unused.
- Absolute numbers are optimistic: the degradation hides the ball exactly on the true dead-ball runs. Only relative comparisons count; do not quote them as BAR-ATH expectations.
- Shipped configuration: plain TCN, beta 0, w 1, uniform prior (prior_corrected), transitions = count_transitions over the controlled labels of all 7 training matches (`weights/tcn_idsse_decode.json`).
- Source of the table below: `runs/full_run_stdout.md` (copied verbatim).

## Per model and seed (mean over matches)

| model | seed | spellF1 | macroF1 | n_pred_poss | n_true_poss | transF1_25 |
|---|---|---|---|---|---|---|
| baseline50 | 2027 | 0.250 | 0.579 | 352.4 | 387.9 | 0.305 |
| baseline50 | 2028 | 0.247 | 0.591 | 346.6 | 387.9 | 0.280 |
| frame | 2027 | 0.452 | 0.834 | 258.9 | 387.9 | 0.439 |
| frame | 2028 | 0.449 | 0.832 | 268.4 | 387.9 | 0.425 |
| tcn | 2027 | 0.555 | 0.866 | 297.1 | 387.9 | 0.523 |
| tcn_change | 2027 | 0.552 | 0.867 | 307.9 | 387.9 | 0.526 |
| tcn | 2028 | 0.552 | 0.868 | 304.7 | 387.9 | 0.535 |
| tcn_change | 2028 | 0.552 | 0.869 | 309.3 | 387.9 | 0.538 |

## Chosen (beta, w) per fold

| model | seed | match | beta | w |
|---|---|---|---|---|
| frame | 2027 | J03WMX | 0 | 0.5 |
| frame | 2027 | J03WN1 | 0 | 0.5 |
| frame | 2027 | J03WOH | 0 | 0.5 |
| frame | 2027 | J03WOY | 0 | 0.5 |
| frame | 2027 | J03WPY | 0 | 0.5 |
| frame | 2027 | J03WQQ | 0 | 0.5 |
| frame | 2027 | J03WR9 | 0 | 0.5 |
| frame | 2028 | J03WMX | 0 | 0.5 |
| frame | 2028 | J03WN1 | 0 | 0.5 |
| frame | 2028 | J03WOH | 0 | 1 |
| frame | 2028 | J03WOY | 0 | 0.5 |
| frame | 2028 | J03WPY | 0 | 0.5 |
| frame | 2028 | J03WQQ | 0 | 0.5 |
| frame | 2028 | J03WR9 | 0 | 0.5 |
| tcn | 2027 | J03WMX | 0 | 1 |
| tcn | 2027 | J03WN1 | 0 | 1 |
| tcn | 2027 | J03WOH | 0 | 1 |
| tcn | 2027 | J03WOY | 0 | 1 |
| tcn | 2027 | J03WPY | 0 | 1 |
| tcn | 2027 | J03WQQ | 0 | 1 |
| tcn | 2027 | J03WR9 | 0 | 0.5 |
| tcn_change | 2027 | J03WMX | 20 | 0.5 |
| tcn_change | 2027 | J03WN1 | 5 | 0.5 |
| tcn_change | 2027 | J03WOH | 0 | 1 |
| tcn_change | 2027 | J03WOY | 5 | 0.5 |
| tcn_change | 2027 | J03WPY | 20 | 1 |
| tcn_change | 2027 | J03WQQ | 5 | 1 |
| tcn_change | 2027 | J03WR9 | 20 | 0.5 |
| tcn | 2028 | J03WMX | 0 | 1 |
| tcn | 2028 | J03WN1 | 0 | 1 |
| tcn | 2028 | J03WOH | 0 | 1 |
| tcn | 2028 | J03WOY | 0 | 1 |
| tcn | 2028 | J03WPY | 0 | 1 |
| tcn | 2028 | J03WQQ | 0 | 1 |
| tcn | 2028 | J03WR9 | 0 | 1 |
| tcn_change | 2028 | J03WMX | 5 | 0.5 |
| tcn_change | 2028 | J03WN1 | 20 | 0.5 |
| tcn_change | 2028 | J03WOH | 20 | 0.5 |
| tcn_change | 2028 | J03WOY | 5 | 0.5 |
| tcn_change | 2028 | J03WPY | 20 | 0.5 |
| tcn_change | 2028 | J03WQQ | 0 | 1 |
| tcn_change | 2028 | J03WR9 | 20 | 0.5 |

## Paired spellF1 delta vs frame (per match)

### baseline50, seed 2027

| match | delta |
|---|---|
| J03WMX | -0.149 |
| J03WN1 | -0.134 |
| J03WOH | -0.237 |
| J03WOY | -0.239 |
| J03WPY | -0.269 |
| J03WQQ | -0.191 |
| J03WR9 | -0.194 |
| **mean** | -0.202 |

better 0/7, worse 7/7

### baseline50, seed 2028

| match | delta |
|---|---|
| J03WMX | -0.191 |
| J03WN1 | -0.141 |
| J03WOH | -0.193 |
| J03WOY | -0.205 |
| J03WPY | -0.228 |
| J03WQQ | -0.236 |
| J03WR9 | -0.217 |
| **mean** | -0.202 |

better 0/7, worse 7/7

### tcn, seed 2027

| match | delta |
|---|---|
| J03WMX | +0.130 |
| J03WN1 | +0.055 |
| J03WOH | +0.091 |
| J03WOY | +0.106 |
| J03WPY | +0.053 |
| J03WQQ | +0.139 |
| J03WR9 | +0.148 |
| **mean** | +0.103 |

better 7/7, worse 0/7

### tcn, seed 2028

| match | delta |
|---|---|
| J03WMX | +0.150 |
| J03WN1 | +0.093 |
| J03WOH | +0.102 |
| J03WOY | +0.096 |
| J03WPY | +0.061 |
| J03WQQ | +0.091 |
| J03WR9 | +0.130 |
| **mean** | +0.103 |

better 7/7, worse 0/7

### tcn_change, seed 2027

| match | delta |
|---|---|
| J03WMX | +0.131 |
| J03WN1 | +0.061 |
| J03WOH | +0.091 |
| J03WOY | +0.107 |
| J03WPY | +0.042 |
| J03WQQ | +0.130 |
| J03WR9 | +0.141 |
| **mean** | +0.101 |

better 7/7, worse 0/7

### tcn_change, seed 2028

| match | delta |
|---|---|
| J03WMX | +0.151 |
| J03WN1 | +0.095 |
| J03WOH | +0.104 |
| J03WOY | +0.092 |
| J03WPY | +0.062 |
| J03WQQ | +0.091 |
| J03WR9 | +0.128 |
| **mean** | +0.103 |

better 7/7, worse 0/7

Note: for tcn_change the (beta, w) selection uses the 6 training matches of each fold predicted by the network trained on them (in-sample, no inner retraining, to bound CPU time); the held-out match is never used.

criterion: SODDISFATTO


## Kick-off rule (seqmodel/kickoff.py, config `kickoff`)

Frames of a graded period before its detected kick-off are set to D after decoding (predict.py and the evaluation path; switch `kickoff.enabled`). The kick-off is recognised from the players (Law 8: teams in their own halves, a kicker at the centre spot), the ball being only supporting evidence; only a kick-off within 300 s of the period start is accepted.

- Leave-one-match-out cannot exercise the rule: IDSSE halves start at the kick-off, 14/14 halves give no detection and 0 frames change. The only evidence is the provided match against the FAQ kick-off frames: period 1 detected +17 frames after the first graded frame, period 2 detected 12 frames after 113,742 (12 frames of live play labelled D). The rule changed 2,037 labels (1,315 H and 722 A -> D).
- Tolerances, not derived from the Laws: `max_intruders` = 2 was chosen after observing that the provided second-half kick-off was missed with 1 (two players of one team inside the circle during the set-up); no labels used. Ablation on the provided match: 1 -> period 1 detected, period 2 None; 2 -> both detected.
- `run_speed` 2.0 m/s and `max_live_share` 0.2 (guard against taking a goal restart for the start) are heuristic, set between two observed cases: an external IDSSE goal restart (live share 0.45) and the provided second half (0.05). Known gap: a calm, walking passage before a restart-like formation is still accepted.

## Goalkeeper fix (keeper holds, `keeper_ball` feature, keeper identity cleaning)

Measurements made while implementing the fix (commit fab8b43). The full two-seed table with the fix
is reported separately once it has run.

**Hold detection on clean IDSSE** (ball > 0.5 m, within 1 m of the team's deepest player, inside its
own penalty area, ≥ 1 s): 77 holds, 4,550 frames over the 7 matches (6/11/14/16/11/10/9 per match);
every hold frame is live (never D) and 99.6% belong to the keeper's team.

**One held-out fold** (J03WOY, the match with most holds; seed 2027; trained on the other 6 degraded
matches; decoding as shipped; kick-off rule on). "Before" = no keeper hiding in training and no
`keeper_ball` (18 features); "after" = defaults (19 features). Metric on the 895 true hold frames:

| test degradation | model | D share on holds | own-team share on holds | match spellF1 | macroF1 |
|---|---|---|---|---|---|
| holds hidden | before | 0.109 | 0.887 | 0.5635 | 0.8589 |
| holds hidden | after | **0.000** | **1.000** | 0.5588 | 0.8659 |
| holds seen | before | 0.125 | 0.873 | 0.5685 | 0.8591 |
| holds seen | after | **0.000** | **1.000** | 0.5563 | 0.8638 |

The whole-match spellF1 change (−0.005 / −0.012) is from a single fold and seed, within its noise.

**Keeper identity cleaning:** 0 corrections on the 7 IDSSE matches (clean and degraded); on the
provided match 14 keeper stretches moved (11,734 frames) and 5,826 frozen frames dropped.

**Provided match, no ground truth used:** graded frames predicted D in "keeper situations" (a team's
deepest player inside its own penalty area with the last-seen ball within 3 m): 2,684 with the
original model, 1,108 with the fix (−58%).
