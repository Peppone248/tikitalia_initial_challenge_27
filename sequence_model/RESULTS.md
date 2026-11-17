# Results (leave-one-match-out on the 7 IDSSE matches, seeds 2027 and 2028)

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
