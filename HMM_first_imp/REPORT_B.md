# Approach B — boosted baseline, too few possessions, D-SLOW, broadcast aligned (30/10/2026)

```powershell
.\venv\Scripts\python.exe HMM_first_imp\scripts\make_submission_b.py              # -> HMM_first_imp\outputs\run_b\tikitalia.csv
.\venv\Scripts\python.exe HMM_first_imp\scripts\eval_baseline_b.py --bootstrap 1000   # IDSSE table below (needs the c_v3 labels, see §6)
.\venv\Scripts\python.exe HMM_first_imp\scripts\align_broadcast.py                 # broadcast -> tactical, checks in outputs\broadcast\alignment.json
```

Code: `hmm_v0/baseline_b.py` (approach B, D-SLOW), `hmm_v0/broadcast.py` (alignment, D-BC),
`scripts/make_submission_b.py`, `scripts/eval_baseline_b.py`, `scripts/diag_possessions.py`,
`scripts/align_broadcast.py`. Config: `config_b/baseline_b.yaml`. Tests:
`tests/test_baseline_b.py` (6/6; `test_hmm_v0.py` still 25/25).

Every choice below was made on IDSSE (7 matches, controlled labels, simulated Smart Tagging,
official scorer). The public 5 minutes were scored once, after the choices, as a structure check.

## 1. Approach B

```text
nearest player to the ball (where seen) ─► majority vote ±25 frames ─► H/A
          ball not seen in the window ─► the more SPREAD-OUT team (std x + std y of its players)
D of c_v3 (v1 HMM + Smart Tagging anchor) ∪ D-SLOW  ─► written on top
possessions < 2 s inside a live stretch ─► given to the team before them
```

No HMM decision, no evidence weights: three arrays and two passes. Ablation on IDSSE, each line
adds one piece to the previous one:

| degraded tracking | spellF1 [90% CI] | macroF1 | D F1 | team acc | possessions (true 2705) | vs c_v3 |
|---|---|---|---|---|---|---|
| baseline50 (official) | 0.210 [0.199–0.221] | 0.462 | 0.389 | 0.551 | 2782 | −0.109 (0/7) |
| c_v3 | 0.319 [0.302–0.336] | 0.740 | 0.835 | 0.733 | 1264 | — |
| B0: baseline50 + our D | 0.284 [0.272–0.297] | 0.680 | 0.787 | 0.528 | 2504 | −0.035 (1/7) |
| B1: + last team when unseen, ±25, min 2 s | 0.328 [0.312–0.343] | 0.722 | 0.837 | 0.684 | 1608 | +0.009 (6/7) |
| B2: + spread-out team when unseen | 0.347 [0.332–0.362] | 0.750 | 0.835 | 0.750 | 1921 | +0.029 (7/7) |
| **B: + D-SLOW** | **0.354** [0.339–0.368] | **0.770** | **0.867** | 0.748 | 1912 | **+0.036 (7/7)** |

| clean tracking | spellF1 [90% CI] | macroF1 | D F1 | team acc | possessions | vs c_v3 |
|---|---|---|---|---|---|---|
| baseline50 (official) | 0.326 | 0.397 | 0.000 | 0.838 | 2644 | −0.153 (0/7) |
| c_v3 | 0.479 [0.461–0.496] | 0.876 | 0.957 | 0.827 | 1573 | — |
| **B** | **0.500** [0.483–0.515] | 0.864 | 0.957 | 0.809 | 1968 | **+0.022 (6/7)** |

B is better than c_v3 on both dirty and clean tracking, so it is not a system that is good only
on dirty data. On clean tracking B1 = B2 = B: the ball is almost always seen, so the spread and
D-SLOW never fire (by construction).

**Choices** (IDSSE degraded, pooled spellF1): window ±25 vs ±50 vs ±125 → 0.347 / 0.344 / 0.29;
minimum possession 1.5 / 2 / 2.5 / 3 s → 0.349 / 0.347 / 0.335 / 0.331: 2 s taken (1.5 s is
+0.002 and wins 6/7 instead of 7/7, within noise: the conservative value). Spread smoothing
1–10 s: flat (0.346–0.347), 3 s taken. Mixing the spread into the frames where the ball IS seen
(score = vote + β·spread) is worse for every β (0.344 at best): the spread is a fallback, not a vote.

**Why the spread works.** The team in possession opens up, the defending team closes down. On
IDSSE degraded, "the more spread-out team has the ball" alone names the right team on 74% of the
alive frames and on **71% of the frames with the ball unseen**, where c_v3 gets 60%. It needs no
ball, no identity and no event: only the positions of each team's players.

## 2. Point 2 — too few possessions: we miss changes, we do not invent them

Fewer possessions than the truth means fewer cuts, i.e. **too few changes of possession
attributed**, not too many. `scripts/diag_possessions.py` splits where possessions end
(IDSSE degraded, controlled labels):

| | c_v3 | B |
|---|---|---|
| possessions pred / true | 1264 / 2705 | 1912 / 2705 |
| dead spells pred / true | 722 / 711 | 746 / 711 |
| true direct changes H↔A | 1995 | 1995 |
| found within ±1 s · ±2 s | 12% · 18% | 17% · 31% |
| predicted changes | 540 (39% false at ±2 s) | 1164 (50% false) |
| true dead spells found: < 5 s · 5–15 s · ≥ 15 s | 67% · 87% · 96% | 67% · 89% · 99% |

- **The dead ball is not the cause.** We have as many dead spells as the truth and find 96% of
  the long ones. The missing possessions are the **direct changes H↔A**: 1995 in the truth (74%
  of all possession ends), 540 predicted in c_v3, only 18% of them timed right.
- B adds the changes that happen while the ball is unseen (the spread switches team), which is
  where the λ = 0.5 s rule of C cannot buy a change: +650 possessions, +0.036 spellF1, even though
  half of the predicted changes are not within 2 s of a true one.
- **BAR-ATH**: c_v3 2.15 possessions/min, B **3.48/min**. The public window has 3.4/min
  (17 in 5 min) and B finds **17/17** there.

## 3. Point 1 — dead ball for fouls and offsides: D-SLOW

On IDSSE the stoppages that end with a **free kick** (fouls, offsides) are **42% of the dead-ball
frames** and the worst found (frame recall 72%, start late by 1.4 s median; throw-ins 78%,
goal kicks 79%, corners 93%). **84% of the dead frames we miss have the ball unseen** (D recall
94% where the ball is seen, 55% where it is not).

The simplest witness of a stoppage needs no ball: **everybody walks**. Mean speed of the tracked
players, smoothed over 2 s, on IDSSE:

| mean player speed | < 0.75 | 0.75–1.0 | 1.0–1.25 | 1.25–1.5 | 1.5–2.0 | > 2.5 m/s |
|---|---|---|---|---|---|---|
| share of frames truly dead | 100% | 96% | 82% | 59% | 29% | ≤ 5% |
| our D on those frames (c_v3) | 91% | 79% | 64% | 45% | 24% | ≤ 8% |

**D-SLOW** = spans where the mean player speed stays under **1.1 m/s for ≥ 3 s** (gaps < 1 s
joined) **and the ball is unseen on ≥ 50% of the span**; added to the D of c_v3. On IDSSE it adds
2.6% of the frames as D with **94% precision**: D F1 0.835 → 0.867, spellF1 +0.007 (7/7). Without
the ball gate it costs −0.009 on clean tracking, so the gate stays (where the ball is seen, the D
rules already see the stoppage). Thresholds: 1.1 / 1.25 m/s are within 0.002, the lower one taken.

The speed distributions of BAR-ATH and IDSSE degraded are close (median 1.51 vs 1.64 m/s,
29.9% vs 26.2% of graded frames under 1.1 m/s), so the threshold transfers. On BAR-ATH D-SLOW
adds more (**9.7%** of graded frames, D 32.8% → 42.4%): BAR-ATH misses more dead ball than IDSSE
(c_v3 D share 32.8% vs 38.4%). The broadcast confirms these frames independently (§4.3):
**89% of them are frames where the TV was not showing live play**, against 8% of clearly live
frames.

Not done yet from the point-1 list: how the ball disappears and reappears (still, on a line,
impossible jump) and players beyond the lines. D-SLOW took the largest share first.

## 4. The broadcast recording, aligned to the tactical timeline

### 4.1 Method (`hmm_v0/broadcast.py`, no frame typed by hand)

DATA.md: the broadcast has its own timeline, cuts and replays, and no mapping is given. Both ATDs
are in pitch coordinates, so the two trackers can be matched:

1. **Coarse**: 20 s windows every 60 s; every offset t − b (2-frame step) scored by the mean ball
   distance (capped at 5 m); a window keeps its best offset only if it beats every offset > 10 s
   away by 0.8 m.
2. **Modes**: offsets voted by ≥ 10% of the windows. Inside a half the TV clock is continuous:
   replays are shown *instead of* live play, they do not shift the timeline.
3. **Switch** between modes: the frame minimising the summed player-matching cost of both sides.
4. **Fine**: ±12 frames around each mode, scored by the median distance from every broadcast
   player to the nearest same-team tactical player.
5. **Live / cut**: a broadcast frame shows live wide play if its homography is valid and its
   players match (median < 1.5 m, ≥ 4 players). Otherwise it is a **cut**: close-up, replay,
   crowd, graphic.

### 4.2 Checks: the alignment is right to 1–2 frames

| check | result |
|---|---|
| mapping found | broadcast 0–84,724 → tactical **+4,014**; 84,725–164,487 → **+26,920** (halftime removed) |
| coarse windows agreeing with the two modes | 59/60 (98%) |
| drift over the match (best offset per 5 min block) | within ±1 frame of the mode on every block with live play |
| player residual on live frames | **0.29 m** median (two independent trackers) |
| ball residual on live frames | 0.76 m median |
| kick-offs (Smart Tagging): first fast ball, tactical vs broadcast mapped | 11188 / 11187 · 49714 / 49713 · 163084 / 163082 (1–2 frames) |
| graded tactical frames covered by the broadcast | 98.3% (missing: the last ~75 s of the 1st half, the 2nd half before its kick-off) |
| on those frames | live 59.8%, cut 38.4% |
| synthetic test (two offsets + switch) | recovered exactly (`test_alignment_recovers_offsets_and_switch`) |

### 4.3 What it gives

**(a) The cut is a strong dead-ball signal, independent of the tactical tracking.** Share of
frames where the broadcast was on a cut:

| frames (BAR-ATH, no labels) | P(broadcast cut) |
|---|---|
| clearly live (ball seen > 5 m/s and players > 2 m/s) | 8% |
| alive in B | 20% |
| D of c_v3 | 58% |
| **added by D-SLOW** | **89%** |
| everybody under 0.8 m/s | 80% |

So the broadcast validates D-SLOW on the test match itself, without labels.

**(b) D-BC (option, OFF by default).** Inside a cut ≥ 3 s, the frames where nobody runs (mean
speed < 1.5 m/s) and no fast ball is seen are D. On BAR-ATH it adds **6.1%** of the frames on top
of B (D 42.4% → 48.5%, 359 possessions instead of 369).
Public 5 minutes, structure check only: D recall 76% → 96% at the same precision (0.87 → 0.86),
macroF1 0.835 → 0.878, spellF1 0.471 → 0.482 (noise ±0.36).
**Why off**: IDSSE has no broadcast, so D-BC cannot be validated on other games; the only
labelled frames it touches are 350 of the public window, which is not a tuning set (RULES §2).
48.5% D is also above the usual 35–45%. It is the team's call; the code is ready:
`make_submission_b.py --set broadcast_dead.enabled=true`.

**(c) The broadcast ball.** It fills 3.7% of the graded frames where the tactical ball is
missing (61% of them alive in B); where both are seen, the nearest team agrees 88% of the time.
Effect on B: +7 possessions, public window unchanged. Little gain: the broadcast tracker loses
the ball where the tactical one does. Off.

## 5. BAR-ATH

```text
CSV         outputs/run_b/tikitalia.csv     SHA-256 710380dbf4a4dd6415525f8d6015c2ef2c4c36643f6d88e2eb49e3ccb773b440
            regenerated from scratch (c_v3 included, its CSV again 6a7baedd…): identical
graded      H 34.0% · A 23.6% · D 42.4%            (c_v3: 43.5 / 23.7 / 32.8)
possessions 369, 3.48/min                           (c_v3: 228, 2.15/min)
dead spells 136
option D-BC outputs/run_b_bc/tikitalia.csv  SHA-256 1c239f0d30f4813fdb1849154967503bb7f8db50fdb3b64dfe212b89c98e9655
```

Public 5 minutes, official scorer, **structure check only** (choices made before, on IDSSE):

| file | spellF1 | macroF1 | possessions (true 17) | mean s (true 14.2) |
|---|---|---|---|---|
| c_v3 | 0.513 | 0.830 | 15 | 16.6 |
| B + D-BC (option) | 0.482 | 0.878 | 17 | 13.8 |
| **B** | 0.471 | 0.835 | **17** | 14.6 |
| baseline_smooth125 | 0.424 | 0.581 | 16 | 18.5 |
| baseline_smooth50 | 0.326 | 0.621 | 26 | 10.8 |

All within the window's noise (±0.36). Estimate on the full match, the same way as for v3: IDSSE
ratio B / baseline50 = 0.354 / 0.210 = ×1.69 → **about 0.41** on BAR-ATH (official floor 0.243).
It is an estimate; only the organisers can measure it.

## 6. Reproduce the IDSSE numbers

```powershell
.\venv\Scripts\python.exe HMM_first_imp\scripts\run_validation.py --st-sim --versions baseline50 c_v3 --variants degraded clean --out-dir outputs/validation_b
.\venv\Scripts\python.exe HMM_first_imp\scripts\eval_baseline_b.py --bootstrap 1000     # -> outputs/validation_b/results_b.md
.\venv\Scripts\python.exe HMM_first_imp\scripts\diag_possessions.py degraded
```

## 7. Limits and next steps

1. **Changes are still the bottleneck**: 31% of the true changes found within ±2 s, half of the
   predicted changes are not near a true one. The spread is a slow signal (team shape takes 1–3 s
   to change). Next: date a change at the first frame the spread flips sign *and* the forward
   velocities of the two teams reverse (66% accuracy alone on unseen frames).
2. **D-BC** waits for a team decision (§4.3b). A way to validate it on other games would be a
   broadcast-like feed for IDSSE, but its parameters would have to come from BAR-ATH without labels.
3. Point 1 continued: ball reappearing still or placed (free kick), players beyond the lines.
4. B still takes the D from c_v3, so it still runs the v1 NN + HMM for the dead ball (~65 s).
