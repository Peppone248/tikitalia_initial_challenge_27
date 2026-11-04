# Approach B — boosted baseline, too few possessions, D-SLOW, broadcast aligned (30/10/2026)

**Update 01/11/2026: the dead ball starts when the ball leaves the pitch** (section 11), after the
**tracking cleaning** (keeper identity, section 10), which runs before everything in B v3.

**Update 31/10/2026: B v3** (who has the ball: control votes, keep last team, position model) is
the current best: section 9. `make_submission_b.py` builds B v3 by default (`outputs\run_b3\`).
Earlier versions: `--config HMM_first_imp\config_b\baseline_b_v2.yaml --out-dir outputs/run_b2`
(B v2, section 8), `--config HMM_first_imp\config_b\baseline_b_v1.yaml --out-dir outputs/run_b` (B v1).

```powershell
.\venv\Scripts\python.exe HMM_first_imp\scripts\make_submission_b.py              # B v3 -> HMM_first_imp\outputs\run_b3\tikitalia.csv
.\venv\Scripts\python.exe HMM_first_imp\scripts\eval_baseline_b.py --bootstrap 1000   # IDSSE table below (needs the c_v3 labels, see §6)
.\venv\Scripts\python.exe HMM_first_imp\scripts\align_broadcast.py                 # broadcast -> tactical, checks in outputs\broadcast\alignment.json
```

Code: `hmm_v0/baseline_b.py` (approach B, D-SLOW), `hmm_v0/dead_b.py` (B v2 rules),
`hmm_v0/team_b.py` (B v3: control votes, position model), `scripts/train_position_model.py`,
`hmm_v0/broadcast.py` (alignment, D-BC, D-BCN),
`scripts/make_submission_b.py`, `scripts/eval_baseline_b.py`, `scripts/diag_possessions.py`,
`scripts/align_broadcast.py`. Config: `config_b/baseline_b.yaml`. Tests:
`tests/test_baseline_b.py` (22/22; `test_hmm_v0.py` still 25/25).

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
| kick-offs (Smart Tagging): first fast ball, tactical vs broadcast mapped | within 1–2 frames on all three |
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

## 8. B v2 (31/10/2026): goalkeepers, placed ball, set-piece formations

From the team's 31/10 brainstorming (points 1, 2, 3, 4, 10, 5). Code `hmm_v0/dead_b.py`, settings `config_b/baseline_b.yaml`, every rule measured
alone and together on IDSSE (`scripts/eval_baseline_b.py`), kept only if it helps on dirty
tracking and costs nothing on clean tracking.

### 8.1 The rules

| # | rule | what it does | kept |
|---|---|---|---|
| 1 | **goalkeeper fix** | a fragment of a slot that lives in one penalty area (≥ 90%) as the deepest of all players towards that goal (≥ 60%), for ≥ 10 s, belongs to the team defending that goal; if the slot says otherwise, the fragment moves to a new column with the right team | **yes** |
| 2 | **placed ball** | ball seen still (< 0.5 m/s) on the pitch for ≥ 2 s = dead, from where it was last seen moving (≤ 10 s back, through a gap) to the kick (> 8 m/s within 2 s); not when a player holds it inside his own box (keeper's hands: alive) | **yes** |
| 3 | earlier start | D-SLOW spans moved back over the collective deceleration (≤ 1.5 s) | no (±0.000) |
| 4 | **wall** | ≥ 3 players of one team within 1.5 m of each other, all standing (< 1 m/s): dead from the stoppage (back over the deceleration/walking, ≤ 8 s) to the kick | **yes** |
| 4 | corner | a standing player within 2.5 m of a corner flag | no (≈ 0) |
| 4 | throw-in | a player ≥ 1 m beyond a touchline (not near the corners) | no (team acc −0.009) |
| 10 | goal kick | a player in his own goal area, no opponent in the penalty area, the ball SEEN still in the goal area (with the ball unseen this is also the keeper holding after a save: precision 65%, −0.014) | no (≈ 0 strict) |
| 5 | D-BCN (broadcast) | broadcast on a cut AND fewer than 19 players in the tactical radar (1 s mean) | option, off |

Precision of the formations alone on IDSSE degraded (share of their frames truly dead): wall
**100%**, placed ball **99%**, corner 97%, throw-in 90%, goal kick 65% (loose version). Most of
these frames are already D in B: the gain is the part that is not (wall +0.5% of the frames,
99% dead; placed +1.3%, 93% dead).

### 8.2 Results on IDSSE (7 matches, controlled labels)

| | spellF1 degraded [90% CI] | macroF1 degraded | spellF1 clean | macroF1 clean |
|---|---|---|---|---|
| baseline50 (official) | 0.210 | 0.462 | 0.326 | 0.397 |
| c_v3 | 0.319 [0.302–0.336] | 0.740 | 0.479 | 0.876 |
| B (v1) | 0.354 [0.339–0.368] | 0.770 | 0.500 | 0.864 |
| B v2.1: + goalkeeper fix | 0.354 | 0.770 | 0.500 | 0.864 |
| B v2.2: + placed ball | 0.361 | 0.779 | 0.506 | 0.870 |
| **B v2: + wall** | **0.363** [0.348–0.378] | **0.783** | **0.507** | **0.870** |
| (off) + corner + throw-in + goal kick | 0.363 | 0.784 | 0.498 | 0.865 |
| (off) + earlier start | 0.362 | 0.783 | 0.507 | 0.870 |

**Paired, B v2 − B v1: spellF1 +0.009 degraded (7/7 matches), +0.006 clean (7/7); macroF1
+0.013 (7/7), +0.006 (7/7).** D F1 0.867 → 0.886 (degraded), 0.957 → 0.966 (clean).
B v2 − c_v3: +0.045 degraded (7/7), +0.028 clean (6/7).

The goalkeeper fix does nothing on IDSSE (by design: its player tracks are never swapped across
teams; it fires once in 14 match-variants). It is kept for BAR-ATH, where the swaps exist.

### 8.3 BAR-ATH

```text
CSV         outputs/run_b2/tikitalia.csv   SHA-256 c36b55ecd35797a1049f18015b29cc4a1553cb8c6b59d1207914835e59c81f36
            regenerated from scratch (c_v3 again 6a7baedd…, B v1 again 710380db… with baseline_b_v1.yaml)
graded      H 34.2% · A 22.8% · D 43.0%        (B v1: 34.0 / 23.6 / 42.4)
possessions 363 · dead spells 137              (B v1: 369 · 136)
```

- **Goalkeepers: 3 fixes, all on one slot** (Athletic in the metadata), which carries **Barcelona's
  goalkeeper**, the longest one over **the last 226 s of the match**.
  That is the "goalkeeper mirrored in the opposite box": in the 2nd half an
  Athletic slot sat in Barcelona's goal area on 8.7% of the frames, with no Barcelona player there.
  496 frames change team, 371 of them in that last stretch.
- **Placed ball + wall add 0.56% of the frames as D; 97% of them are frames where the broadcast
  was on a cut** (close-up / replay), an independent witness without labels.
- **Public 5 minutes**: B v2 = B v1 exactly (spellF1 0.471, macroF1 0.835, 17/17 possessions):
  no wall, placed ball or keeper swap in those 5 minutes. The window cannot see this change.

### 8.4 Point 5: broadcast framing + players in the radar (D-BCN, off)

Frames by broadcast state and tactical player count (BAR-ATH, B v2 labels; public GT where it exists):

| broadcast | players in the radar | share of graded | B v2 says D | public GT D |
|---|---|---|---|---|
| live | < 19 | 1.5% | 80–91% | – |
| live | ≥ 20.5 | 53.5% | 19% | 5% (5221 frames) |
| cut | < 16 | 5.1% | 91% | – |
| cut | 16–19 | 4.3% | 88% | – |
| cut | 19–20.5 | 5.5% | 81% | 15% (130) |
| cut | ≥ 20.5 | 23.6% | 61% | 63% (1730) |

Few players in the radar is already a stoppage for B v2 (88–91% D), with or without the cut, so
the combined rule adds little: **D-BCN adds 1.0% of the frames** (D 43.0% → 44.0%, 350
possessions). The large remaining block is "cut with all players visible" (23.6%, 61% D for us,
63% in the public GT), which is what D-BC (§4.3b, players slow inside a cut) targets: +5.6%.
Neither can be validated on other games and the public window does not contain the D-BCN cases,
so both stay off: `--set broadcast_count_dead.enabled=true` / `--set broadcast_dead.enabled=true`.

| option on top of B v2 | D share | possessions | public spellF1 / macroF1 | SHA-256 |
|---|---|---|---|---|
| none (B v2) | 43.0% | 363 | 0.471 / 0.835 | c36b55ec… |
| D-BCN | 44.0% | 350 | 0.471 / 0.835 | e0e6b881… |
| D-BC | 48.6% | 355 | 0.482 / 0.878 | bcba8dba… |
| D-BC + D-BCN | 48.9% | 348 | 0.482 / 0.878 | 75e943d0… |

## 9. B v3 (31/10/2026): who has the ball (points 7, 8, 9)

### 9.1 Why here

Oracle test on IDSSE degraded (each piece replaced by the truth, the rest of B v2 unchanged):

| | spellF1 |
|---|---|
| B v2 | 0.363 |
| ball always seen (a perfect video ball detector) | 0.392 |
| perfect dead ball | 0.442 |
| perfect team on the alive frames | **0.812** |

The points are in the team, not in the dead ball (≤ +0.08 left) nor in seeing the ball (+0.03:
with the ball always seen, "nearest player" still gives 0.392). Where B v2 is wrong
(19.7% of the alive frames): **71% of its errors are within 3 s of a true change**, 38% have the
ball seen (error rate 14%: the nearest player votes even when the ball is flying or loose), 44%
are in ball gaps ≥ 10 s (error rate 26%).

### 9.2 The rules (`hmm_v0/team_b.py`, settings `possession_v3` in `config_b/baseline_b.yaml`)

| point | rule |
|---|---|
| **8, 9** | **vote only on control**: a team votes only when the ball is seen, its nearest player is within **1.5 m** and the ball is slower than **8 m/s**. A pass or clearance in flight, a loose or contested ball votes for nobody (CHALLENGE.md: possession stays with the last team; an aerial duel is no change). Majority of the control votes over ±1 s. |
| **8, 9** | **keep the last team**: with no control vote, the last team that controlled keeps the ball for **1 s** |
| **7** | **position model**: after that, P(home in possession) from 8 collective features of the players, logistic regression trained on IDSSE (degraded like BAR-ATH, other games only, RULES §1) |
| | possessions shorter than **1.5 s** go to the team before them (was 2 s: with a sharper team signal the shorter real possessions can be kept, as the oracle said) |

**The 8 features** (home minus away, home always attacking +x, smoothed 3 s): spread (std x +
std y), length, width, forward velocity, centroid of all players (territory), height of the
defensive line, height of the most advanced player, mean speed. Standardised weights of the
final model: centroid **+5.9** and defensive line **−5.9** (together: how high the team keeps its
line compared with where the play is), spread +1.1, forward velocity +0.85, the others ≤ 0.6.
Alone, leave-one-match-out on IDSSE degraded, the model names the team of **83.5%** of the alive
frames and **81.5% with the ball unseen** (spread alone: 74.4% / 71.7%).

**Transfer to BAR-ATH, checked without labels**: the model agrees with the control votes of the
BAR-ATH tracking on **83.9%** of the controlled alive frames (on IDSSE: 87.6%), and the BAR-ATH
features sit in the training range (standardised means within ±0.4, standard deviations 0.9–1.2).
The weights live in `config_b/position_model.json` (deterministic training,
`scripts/train_position_model.py`); the prediction sums the products in numpy, not with a BLAS
matrix product, so the CSV is the same on every machine.

### 9.3 Results on IDSSE (7 matches, controlled labels; position model leave-one-match-out)

| | spellF1 degraded [90% CI] | macroF1 degraded | team acc | spellF1 clean [90% CI] | macroF1 clean |
|---|---|---|---|---|---|
| baseline50 (official) | 0.210 | 0.462 | 0.551 | 0.326 | 0.397 |
| c_v3 | 0.319 [0.302–0.336] | 0.740 | 0.733 | 0.479 [0.461–0.496] | 0.876 |
| B v2 | 0.363 [0.348–0.378] | 0.783 | 0.746 | 0.507 [0.489–0.522] | 0.870 |
| B v3.1: (7) no control vote → position model | 0.423 | 0.806 | 0.781 | 0.507 | 0.870 |
| B v3.2: + (8, 9) vote only on control | 0.464 | 0.821 | 0.804 | 0.578 | 0.897 |
| B v3.3: + minimum possession 1.5 s | 0.470 | 0.821 | 0.805 | 0.589 | 0.898 |
| **B v3: + keep last team 1 s** | **0.473** [0.457–0.488] | **0.822** | **0.806** | **0.601** [0.584–0.616] | **0.900** |

**B v3 − B v2: spellF1 +0.110 degraded, +0.094 clean; macroF1 +0.039 / +0.031; better on 7/7
matches in both.** The 90% intervals no longer overlap. B v3 − baseline50: +0.263 degraded.
On clean tracking the model is rarely used (the ball is almost always seen): there the gain is
all from voting only on control.

Choices (IDSSE degraded, then checked on clean; each knob swept around neighbouring values of the
others, so the numbers compare within a list, not across lists): control radius 1.0 / 1.5 / 2 / 3 m → 0.464 /
0.464 / 0.461 / 0.447 (1.5 m: the flat region, more votes); ball speed 8 / 12 / 16 m/s → 0.464 /
0.462 / 0.454; vote window ±12 / ±25 / ±50 → 0.457 / 0.456 / 0.433; minimum possession 1 / 1.5 /
2 / 3 s → 0.453 / 0.460 / 0.456 / 0.427; keep last 1 / 2 / 3 s → 0.472 / 0.473 / lower (1 s, the
shorter within noise). Blending the vote and the model in one score was worse (0.427–0.428):
the model is a fallback, as the spread was.

### 9.4 BAR-ATH

```text
CSV         outputs/run_b3/tikitalia.csv   SHA-256 f9f053a064f93d4d133512beda62b77686dc67978a4d126cf531088b0a6c5c58
            regenerated from scratch (c_v3 again 6a7baedd…; B v2 again c36b55ec… with baseline_b_v2.yaml)
graded      H 32.9% · A 24.1% · D 43.0%        (B v2: 34.2 / 22.8 / 43.0)
home share of the alive frames 57.7%           (B v2: 60.0%; public window truth: 57%)
possessions 351                                (B v2: 363)
```

Public 5 minutes, official scorer, **structure check only** (choices made before, on IDSSE):

| file | spellF1 | macroF1 | possessions (true 17) | transF1 | latency |
|---|---|---|---|---|---|
| c_v3 | 0.513 | 0.830 | 15 | 0.270 | +0.0 frames |
| **B v3** | **0.506** | **0.836** | **17** | **0.308** | +11.7 |
| B v2 | 0.471 | 0.835 | 17 | 0.256 | +8.0 |
| baseline_smooth125 | 0.424 | 0.581 | 16 | 0.167 | −7.7 |
| baseline_smooth50 | 0.326 | 0.621 | 26 | 0.082 | +4.0 |

B v3 finds one more true possession than B v2 at IoU 0.3–0.5 (11 / 11 / 10 against 10 / 10 / 9)
and the best transF1 of all. Within the window's noise (±0.36), but the same direction as IDSSE.

### 9.6 Keeper hold (added 31/10)

The case: Barcelona's keeper holds the ball in his box and throws it out; the tracker's
point for him drifts 2–3 m off the ball while he holds it up, an Athletic forward pressing him is
within 0.6–0.9 m, the ball is slow (he walks with it): the control vote went to Athletic for ~1 s.

Rule (`team_b.keeper_hold`, `possession_v3.keeper_hold`): a team that controls the ball for ≥ 1 s
inside its OWN penalty area keeps it until the release (ball faster than 8 m/s, at most 8 s); inside
that window every control vote is that team's. General, not tied to any frame; accepted because it
costs nothing on IDSSE: **+0.000 degraded, +0.001 clean** (3 better / 1 worse) -- on IDSSE the keepers
are tracked well, so an opponent never looks closer to the ball than the keeper. Keeping the ball in
the air after the release for the keeper's team (until the next control) measured −0.0004: off.

BAR-ATH: 46 holds; **125 frames change (0.08%), all A → H**. The case was found on a frame of the
public window, so the public-window change (macroF1 0.836 → 0.845, spellF1 unchanged) is NOT
evidence for the rule; IDSSE is.

```text
CSV         outputs/run_b3/tikitalia.csv   SHA-256 8ed4396690086a0ce0f9c7f9e711a924cdebdc8d3a55fc72b81e584575931a92
            (without the rule, --set possession_v3.keeper_hold.enabled=false: f9f053a0..., the B v3 of 9.4)
```

### 9.5 Limits and next steps

1. **Changes are dated late**: +11.7 frames (~0.5 s) on the public window. A control vote needs
   the ball within 1.5 m of the new team, so the change comes when the receiver has it, not at the
   interception. Next: date the change at the start of the run of control frames, not where the
   majority flips.
2. The position model is trained on Bundesliga (DFL). Its transfer to BAR-ATH is checked without
   labels (83.9% agreement with the control votes against 87.6% on IDSSE): good, not proven.
3. The dead ball (≤ +0.08 left in the oracle) is now the second lever again; the video ball
   detector (+0.03 with the old logic) may be worth more with control votes: to remeasure.

## 10. Tracking cleaning: obvious tracker errors fixed by code (01/11/2026)

Code `hmm_v0/clean_b.py`, settings `cleaning:` in `config_b/baseline_b.yaml`, applied first in
`scripts/make_submission_b.py`; every correction is listed in `run_summary.json` (`cleaning_fixes`).

### 10.1 Compliance with RULES.md

- **Allowed**: "Improving that data [...] is entirely legitimate and needs no justification [...] as
  long as they are produced by an algorithm" (*The provided data is a starting point*).
- **Respected limits**: corrections come out of code ("Corrections must come out of code"); no frame
  number, no slot number and no constant tied to a moment of the provided game -- the keeper slots
  are read from the ATD metadata (`position_type`), the rules are physical / football constraints
  applied to the whole match and run unchanged on the IDSSE matches; thresholds fixed a priori from
  physics, accepted on IDSSE, never tuned on BAR-ATH or on the public window; the input files are
  never modified (cleaned arrays in memory); no cleaned frame-level data is published.
- Each correction is a general rule, not labels or frame-specific corrections.

### 10.2 Rule 1: keeper identity

A goalkeeper slot of team T (from the metadata) inside the **opponent's** penalty area, on a
stretch of track that did **not run there from T's own half**, is the opponent's keeper: the tracker
gave his track the other keeper's identity. That stretch is moved to the other team. A track
stretch ends at a gap or at a jump > **2.5 m between two frames** (62 m/s: another person). Acts after
**0.5 s** in the opponent's box (box-edge jitter). A keeper who really goes up for a late corner runs
there inside one continuous stretch and is left alone. Inside a wrong stretch, frames where the
point does not move at all (identical coordinates ≥ **2 s**) are a **frozen ghost** and are removed,
not given to the other team (a global "frozen = ghost" rule was rejected: on IDSSE 0.26% of the
frames of real players are identical for 5 s).

**Why it was needed (BAR-ATH, tracking only)**: the Athletic keeper slot spends 254 s (1st half) +
384 s (2nd half) inside Barcelona's box, the Barcelona keeper slot 13 s + 47 s inside Athletic's;
with the ball within 1.5 m on ~950 frames. The old fix (section 8, "deepest player ≥ 10 s") misses
most of them, because a frozen point on the goal line is deeper.

### 10.3 Validation on IDSSE

| | spellF1 | macroF1 | team acc |
|---|---|---|---|
| IDSSE as it is: corrections made | **0** (degraded and clean) | | |
| degraded, no keeper swaps (B v3) | 0.473 | 0.822 | 0.806 |
| degraded + 86 simulated keeper swaps, B v3 without the rule | 0.462 (−0.011) | 0.816 | 0.797 |
| degraded + the same swaps, with the rule | 0.471 (−0.001) | 0.821 | 0.805 |

Simulated swaps: per half a Poisson number of swaps with the BAR-ATH rate (7.5 per half), lengths
resampled from the 15 BAR-ATH corrections (graded seconds), a keeper's track given to a column of the
other team with the keeper flag -- statistics of the BAR-ATH tracking only, as for the degradation and
the simulated Smart Tagging. The rule recovers ~87% of the loss and changes nothing where nothing is
wrong.

### 10.4 BAR-ATH

```text
CSV         outputs/run_b3/tikitalia.csv   SHA-256 bf6c3d45a5dc5567ed4431e2aa250f1b6e7a61fde59f4b14af2a7c0a397ceeb8
            without the cleaning (--set cleaning.keeper_identity.enabled=false): 8ed43966... (section 9.6)
corrections 15 keeper stretches (6 of the Barcelona keeper slot, 9 of the Athletic one)
changed     1324 frames (0.83%): 697 A -> H, 627 H -> A, dead ball unchanged; 338 possessions
```

Public 5 minutes, structure check only: macroF1 0.845 → **0.858**, transF1 0.308 → **0.359**,
latency +11.7 → +7.7 frames, team accuracy 0.875 → 0.895 (122 more right frames; both changed
stretches are now right frame by frame); spellF1 0.506 → 0.471 -- the same possessions cut a little
differently change which predicted spell matches which true one among 17 (within ±0.36). Not used to
decide: the decision is IDSSE's (10.3).

### 10.5 Next tracker errors to collect

Candidates seen in the data, each to be turned into a general rule and validated the same way:
duplicated players (two slots on the same person: > 11 players of a team on the pitch), a team's
outfield slot re-attached to an opponent (team-shape outliers), the ball reported far off the pitch
or jumping faster than a shot.

## 11. The dead ball starts when the ball leaves the pitch (01/11/2026)

### 11.1 The error

A dead span can start seconds before the ball goes out. The Smart Tagging anchor (c_v3) starts a
throw-in's D at the last action seen before the tag (`back_to_last_action_s`): when the last
action is a duel near the touchline, the ball is still in play, comes back into the pitch, changes
team and only then goes out. The label says D for all of it, while the tracking shows the ball inside
the lines and moving. Law 9: the ball is out of play only when it has wholly crossed a line.

### 11.2 The rule (`dead_b.ball_out_start`, `ball_out_start:` in `baseline_b.yaml`)

Inside every dead span in which the ball is later seen OUTSIDE the lines (by more than `out_m`),
the frames before the first crossing where the ball is seen INSIDE (at least `in_m` from every line)
and moving (faster than `min_speed`) are play: the span starts at the crossing. To give back only a
ball really on its way out:

- only the ball seen without a break up to the crossing counts, at most `approach_s` = 4 s before it;
- in that stretch the ball has been at least `min_approach_m` = 2 m inside the line it crosses (not
  only dribbled along it);
- a span with a Smart Tagging **free kick** (`skip_tag_types`) is left alone: the stoppage there is
  the whistle, and a ball going out afterwards is the ball kicked away after it.

A span in which the ball is never seen outside (fouls, offsides, injuries) is never touched. No frame,
slot or moment of the provided game appears anywhere: the rule is a Law of the Game applied to the
whole match, the same way to any game.

### 11.3 Validation on IDSSE (B v3, LOMO position model, paired per match)

```text
                          spellF1   macroF1   D-F1    vs B v3                    frames given back to play
degraded  B v3            0.473     0.822     0.886
          + ball out      0.472     0.822     0.887   -0.0008 (2/7 better)       479, of which truly play 98%
clean     B v3            0.602     0.901     0.966
          + ball out      0.605     0.901     0.967   +0.0032 (6/7 better)       1562, of which truly play 99%
```

Neutral on the degraded variant (the ball is unseen more often near the lines, so the rule rarely
applies), positive on the clean one; on both, almost every frame given back to play is play. The
free-kick guard changes nothing measurable on IDSSE (the simulated feed tags 16% of the free kicks):
it is there for a semantic reason (11.2), not for a score. A first version, which measured the
approach as "distance from the line at the start of the stretch minus at the crossing", missed the
throw-in case that motivated the rule (the duel was right on the line) and was replaced by the
"been 2 m inside" test above; the IDSSE numbers above are for the final version.

### 11.4 BAR-ATH

```text
CSV         outputs/run_b3/tikitalia.csv   SHA-256 05dcc7cec1ecdfb72eaf897e6c949ce5e4f18e9e7d67fe36b004360c7ef1f4c3 (until section 12)
            without it (--set ball_out_start.enabled=false): bf6c3d45... (section 10.4)
changed     839 frames given back to play (0.53% of the graded frames); 347 possessions (338 before)
```

Public 5 minutes, structure check only (not used to decide): macroF1 0.858 → **0.864**, spellF1
0.471 → 0.471, alive/dead F1 0.958 → 0.960, transF1@25 0.359 → **0.410**, team accuracy 0.895 →
0.901. Compliance note: the free-kick guard was added after looking at the one public stretch the
first version changed for the worse (a ball kicked away after a whistle). It has no number in it and
is a Law of the Game reading ("labels meaning what you think", RULES.md §2), it is neutral on IDSSE,
and it can be switched off with one line (`skip_tag_types: []`); without it, IDSSE is +0.0034 clean,
-0.0008 degraded.

## 12. Keeper catch: the ball lost in the goalkeeper's hands (01/11/2026)

### 12.1 The error

A Barcelona shot (15 m/s) is caught by the Athletic goalkeeper. The tracker loses the ball as it
reaches him and finds it again 11.9 s later at his feet, 9 m out of his goal: he has walked to the
edge of the box with the ball in his hands. B v3 labelled the gap: Athletic for 1 s (keep last
team), then **Barcelona for 8.7 s** (the position model: Barcelona is still in its attacking shape,
P(home) 0.6-0.9), then Athletic again when the ball reappears. A phantom Barcelona possession split
the Athletic one in two: the costliest kind of error for spellF1.

Decided before coding: a **rule**, not a tracking fix. Interpolating the ball across a 12 s gap
cannot follow the keeper (he walks 11 m out and back), so no control vote would fire; "attaching
the ball to the keeper" would already be a possession decision hidden inside the coordinates.
Changing bx/by would also move every dead-ball layer.

A second error in the same stretch is NOT addressed here: before the shot the ball runs through
four Athletic defenders at ~1 m and 5-8 m/s, each one a control vote, so the change to Athletic is
dated too early, before the catch (point 2 of 12.5).

### 12.2 The rule (`team_b.keeper_catch`, `possession_v3.keeper_catch` in `baseline_b.yaml`)

A gap in the ball tracking is a catch by team T when

- **before it** the ball was last seen moving faster than `min_speed` = 8 m/s (from its seen
  positions over `look_s` = 0.5 s, not from the smoothed speed, which is inflated at a gap's edge),
  towards T's goal, at most `near_goal_m` = 25 m from T's goal line (a shot or a cross, not a long
  ball);
- **after it**, within `reappear_s` = 1 s, the first control vote is T's, by **T's deepest player**
  (the goalkeeper's position: no role or identity is needed, so the rule runs unchanged on IDSSE),
  with the ball inside T's own penalty area;
- the gap lasts at most `max_gap_s` = 15 s.

The catch starts when the ball reaches the keeper (gap start + distance / last seen speed) and lasts
to that control vote; before it the shooting team keeps the ball (step 3, keep last). It overrides
only frames that no control vote decides. In the chain it is part of step 3 (`team_source` KEEP,
mask `keeper_catch` in `trace.npz`). All four numbers
were set a priori; no frame, slot or match constant appears anywhere.

### 12.3 Validation on IDSSE (B v3 + ball_out_start, LOMO position model, paired per match)

The IDSSE tracking keeps the ball in the keeper's hands, so two variants were added
(`validation.make_variant`, `degrade.keeper_hands` in `config/validation.yaml`): **`*_kh`** = the
ball hidden where the CLEAN ball is above 0.5 m, inside a penalty area, within 1 m of the defending
team's deepest player, for at least 1 s (75 holds in the 7 matches). The simulation reads `ball_z`;
the engine never does. c_v3 was re-run with `--st-sim` on each variant
(`outputs/validation_kc`). `scripts/eval_keeper_catch.py`:

```text
tracking      spellF1 off   on       paired    better / worse   frames changed   right before -> after
degraded      0.4720        0.4720   +0.0000   0 / 0            0                0 -> 0
clean         0.6052        0.6052   +0.0000   0 / 0            0                0 -> 0
degraded_kh   0.4708        0.4708   +0.0000   0 / 0            0                0 -> 0
clean_kh      0.6060        0.6062   +0.0002   1 / 0            116              0 -> 116
```

The full ablation table (`scripts/eval_baseline_b.py --labels-dir outputs/validation_kc --variants
degraded clean degraded_kh clean_kh`, `outputs/validation_kc/results_b.md`) now has the official rows
**B v3.5 + ball_out_start** (degraded 0.472, clean 0.605: the section 11 numbers, previously from a
scratch script) and **B v3.6 + keeper catch** (0.472 / 0.605 / 0.471 / 0.606 on degraded / clean /
degraded_kh / clean_kh), plus sensitivity rows: min_speed 5 or 12, max_gap 8 or 25 s change nothing
at three decimals.

Read with care: IDSSE barely exercises the rule. Of the 75 simulated holds, 45 follow a possession
of the keeper's own team and only 10 follow a fast ball; the B v3 chain already labels most of them
right (hiding the ball costs B v3 nothing: clean 0.602 vs clean_kh 0.604). The evidence is **no
harm on 28 match-variants, one correct firing**, not a measured gain. A first version without
`near_goal_m` fired on a long ball from 58 m into a 12 s gap during which play went on
(degraded_kh J03WQQ, 99 frames wrong, -0.0004): the distance condition removed it and changed
nothing else. With the smoothed speed instead of the seen positions it also fired on two slow balls
in BAR-ATH; replaced.

### 12.4 BAR-ATH

```text
on (default since 01/11)  outputs/run_b3/tikitalia.csv  SHA-256 8ae5118d156036be1ae14f959b4ebad7f8dc58b9f2d344c5e6b0487148ecd720 (until section 13)
off                       --set possession_v3.keeper_catch.enabled=false  SHA-256 05dcc7cec1ecdfb72eaf897e6c949ce5e4f18e9e7d67fe36b004360c7ef1f4c3 (section 11.4)
changed                   217 consecutive frames (8.7 s) Barcelona -> Athletic; 345 possessions (347)
shares (graded)           H 0.3314, A 0.2439, D 0.4247; deterministic (two runs, same hash; Linux / numpy 2.2.6)
```

Two catches are detected in the match (26685-26945, 29979-30222); the second was already labelled
right. Public 5 minutes (structure check only): identical scores, the rule touches no frame of the
window (11185-18684).

### 12.5 Limits and next steps

1. **Switched on** (team decision, 01/11): IDSSE neutral with one correct firing and no loss on 28
   match-variants; on BAR-ATH it fixes one error. The evaluation scripts
   switch it off explicitly for the rows before "B v3.6", so the ablation stays readable.
2. **Ball passing through defenders**: a control vote from a player next to a ball that keeps its
   speed and direction is not control. It touches many frames: measure it alone on
   IDSSE.
3. Catches where the keeper kicks or throws the ball straight from his hands reappear fast and far,
   so the rule does not fire; the hold there is short and step 3 / the model usually cover it.

## 13. Second batch of rules: restarts, fouls, passing balls, unseen exits, team swaps (01/11/2026)

Five errors, one general rule each, each switched off by
default, measured **alone** on IDSSE against the current B v3 (section 12 included; paired per match,
`scripts/eval_rules_b.py`, `outputs/validation_kc/results_rules_b_*.md`), then together. No frame,
slot or match constant is in the code, and every threshold was set
before the IDSSE run (the sensitivity rows were not used to pick a value).

### 13.1 The five cases and the rules

| | case | cause | rule |
|---|---|---|---|
| R1 | a corner already kicked and still D for about 4 s; a goal kick: D ends 1 s **before** it, H for 1 s, D again for 2.2 s **after** it | the c_v3 dead span is placed on the Smart Tagging window, not on the kick | `dead_b.restart_kick`: inside a restart tag (corner, goal kick, throw-in, free kick, kick-off) the first **kick from rest** ends the dead span. Kick = the ball seen leaving > 8 m/s (over 0.2 s, from the seen positions) after being seen < 2 m/s or unseen within 1 s, dated to the end of the rest (the tracker spreads a kick over ~0.8 s; it often finds the ball already kicked). After it: play until a new stoppage shows (ball seen outside, unseen 2 s, still 2 s; at most 15 s; frames D-SLOW / placed / wall keep their D). A span ending at most 3 s before the kick, ball unseen or still in between, is extended to it. |
| R2 | Barcelona's pass runs past an Athletic defender at 0.6-1.3 m, 4 votes, A for 2.2 s (the +/- 1 s majority dates it before he is even close); a Barcelona ball at 7 m/s through four defenders | a control vote only asks "nearest player within 1.5 m, ball < 8 m/s" | R2 `max_rel_speed`: a vote only when \|v_ball - v_player\| <= 3 m/s. R2b `pass_by`: a team's run of votes during which the ball never slows (starts > 3 m/s, never 1.5 m/s below its running maximum) is removed. |
| R3 | ball last seen 3 m from the goal line at 10 m/s, unseen 31 s; D only from the corner tag; the players set up the corner at 1.2-1.6 m/s, so D-SLOW does not fire | ball_out_start needs the ball SEEN outside | `dead_b.ball_leaves_unseen`: a gap >= 3 s whose last seen ball was heading out (> 3 m/s, <= 5 m from the line, crossing within 1 s) is dead from the estimated crossing to the ball's reappearance. R3b: only if the players then slow down (median mean speed over 5 s <= 2 m/s). |
| R4 | a foul for Athletic: ball lost, D-SLOW only about 4.6 s later | D-SLOW needs everybody walking, which comes seconds after the whistle | `dead_b.slow_from_ball_loss`: a D-SLOW span starts where the ball was last seen, if it has been unseen without a break from at most 6 s before. |
| R5 | 12 Barcelona / 10 Athletic during a corner (two Athletic slots end, new "Barcelona" fragments start) | the tracker starts a fragment on a slot of the other team | `clean_b.team_continuity`: a fragment of team T starting where an OPPONENT's track ended (closest ended track within 1 s, <= 2 m + 8 m/s x gap) while T has >= 2 more players tracked (median over the fragment) is moved to a new column of the other team. On IDSSE validated on a new variant `degraded_ts` (5% of the fragments after a gap given to the other team, `validation.team_swaps`). |

A single entry point applies R4 -> R3 -> R1 after ball_out_start (`dead_b.extra_dead`), the same code in
`make_submission_b.py` and in the IDSSE evaluation; R2/R2b sit in `team_b.control_votes` /
`drop_pass_by`, R5 in `clean_b.clean_tracking`. The trace stores the new layers
(`dead_slow_back`, `dead_unseen_out`, `dead_restart_extend`, `dead_restart_play`).

### 13.2 Each rule alone on IDSSE (spellF1, paired vs B v3 with the batch off)

```text
                                        degraded            clean               degraded_ts         frames changed: right before -> after
                                        paired  (b/w)       paired  (b/w)       paired  (b/w)       (degraded)
R1 restart kick                         +0.0019 (6/1)       +0.0008 (4/3)       +0.0013 (4/3)       369 -> 952
R2 relative speed 3 m/s                 -0.0025 (3/4)       -0.0107 (2/5)                           4722 -> 3757
   (sens) 2 / 4 m/s                     -0.0075 / -0.0032   -0.0228 / -0.0031
R2b ball never slows                    -0.0074 (0/7)       -0.0132 (1/6)       -0.0093 (1/6)       5771 -> 4226
R3 ball leaves unseen                   -0.0023 (3/4)       +0.0000 (0/0)                           4280 -> 2309
R3b + players slow down                 -0.0014 (2/4)       +0.0000 (0/0)                           3204 -> 1806
R4 D-SLOW from the ball loss            +0.0014 (7/0)       +0.0000 (0/0)       +0.0014 (7/0)       223 -> 790
   (sens) back 3 s                      +0.0006 (5/0)
R5 team continuity                      +0.0000 (0/0)       +0.0000 (0/0)       +0.0067 (3/4)       (degraded_ts) 2899 -> 4244
```

- **R1 and R4 are kept**: positive on every variant where they act, most of the frames they change
  become right.
- **R2 and R2b are dropped**: worse at every threshold. In a dribble the ball is pushed ahead and
  accelerates away from the carrier, so "receiving means slowing it" and "moving with the player" both
  remove more good votes than bad ones. They fix two cases on BAR-ATH; that is not a reason
  (RULES.md: decided on IDSSE). Another idea is needed for the passing-ball votes.
- **R3 and R3b are dropped**: on IDSSE the ball gaps are random, so a ball lost near a line is seldom a
  ball going out. The BAR-ATH tracker may lose the ball at the line far more often than at random, but
  nothing available can show it without looking at BAR-ATH's labels; the rule stays in the code, off.
- **R5**: no change at all on IDSSE without swaps (no false correction); with simulated swaps it
  recovers 0.0062 of the 0.037 they cost, and the frames it changes become right far more often than
  wrong, but match by match it is 3 better / 4 worse.

### 13.3 Together (decided on IDSSE: R1 + R4 + R5 switched on)

```text
                 degraded            clean               degraded_ts         frames changed: right before -> after
                 paired  (b/w)       paired  (b/w)       paired  (b/w)       degraded / clean / degraded_ts
R1+R4            +0.0033 (7/0)       +0.0008 (4/3)       +0.0027 (7/0)       592->1742 / 475->1192 / 597->1671
R1+R4+R5         +0.0033 (7/0)       +0.0008 (4/3)       +0.0094 (4/3)       592->1742 / 475->1192 / 3492->5924
pooled spellF1   0.4720 -> 0.4752    0.6052 -> 0.6058    0.4351 -> 0.4439
D-F1             0.8867 -> 0.8884    0.9673 -> 0.9688    0.8888 -> 0.8900
```

R5 is switched on although its match count on degraded_ts is 4/3: it never acts on tracking without
swaps (zero frames on degraded and clean, i.e. no false corrections), its mean is positive, and of the
frames it changes about two thirds become right. It is the least certain of the three: off with
`--set cleaning.team_continuity.enabled=false`.

### 13.4 BAR-ATH

```text
CSV        outputs/run_b3/tikitalia.csv  SHA-256 f4958dca4e20522506d067edc5d7ff52f414ef1b6194e6771cd053489f98cc7f
           (two runs, same hash; before this section 8ae5118d... = section 12; on the branch of
           section 14: 4a4d80fb... rules, ba6667b4... supervised)
changed    3242 frames in 47 stretches; 344 possessions (345), 141 dead spells (144)
shares     H 0.3398, A 0.2422, D 0.4180 (D 0.4247 before)
R1         1870 dead frames given back to play after a restart kick, 22 added before one
R4         1594 frames added before D-SLOW spans
R5         55 fragments moved to the other team (20768 player-frames)
```

The cases of 13.1: corner -> play from the kick; goal kick -> D up to the kick, then play;
foul -> D starts about 4.5 s earlier. The R2 and R3 cases are NOT fixed: their rules
failed on IDSSE. The R5 swap is not caught by R5 either (the wrong fragment starts 13 s after the
Athletic tracks were lost, outside the 1 s window).

Public 5 minutes, structure check only (17 possessions, not used to decide): macroF1 0.8638 -> 0.8643,
spellF1 0.4706 -> 0.4706, alive/dead F1 0.9604 -> 0.9606, team accuracy 0.9008 -> 0.9013, transF1@25
0.410 -> 0.359 (one transition of the window moved beyond the +/-25-frame tolerance; mean latency
+6.4 -> +3.1 frames).

### 13.5 Limits and next steps

1. Passing-ball votes need another idea than relative speed or "receiving slows the
   ball": e.g. weighting a vote by how long the same player stays close, or a vote from the team that
   kicked the ball until an opponent stops it (a touch is not a change, CHALLENGE.md).
2. Unseen exits: only a better simulation of WHERE the BAR-ATH tracker loses the ball (near
   the lines, in the air) would let IDSSE judge R3 fairly.
3. R5 with a longer window (lost in a crowd for seconds) needs its own IDSSE run before use.

## 14. Branch supervised-test: third batch of rules and a team learned on the Bundesliga matches (01/11/2026)

Work meant for a separate branch, `supervised-test` (no commit made by Claude; see 14.5). Two parts:
three more rules, measured like section 13, and a first supervised step that
learns the team in play on IDSSE and keeps the rules' dead-ball mask.

### 14.1 Third batch of rules (alone on IDSSE, paired vs the section 13 configuration)

| | case | rule | degraded | clean | decision |
|---|---|---|---|---|---|
| R6 | inside a stoppage the players walk at 1.2-1.3 m/s, just above D-SLOW's 1.1; a 5-7 s island of play opens between two dead spans, the position model fills it | `dead_b.dead_islands`: a play stretch of at most 10 s between two dead spans, ball never seen, median mean player speed <= 2 m/s, is the same stoppage | **+0.0039 (7/0)**, right frames 371 -> 5521 | 0 | **on** |
| R3c | a throw-in and a goal-line case: the ball is lost heading out and found seconds later on the same line | R3 + the ball is seen again within 3 m of the line it left by, or beyond it | **+0.0004 (4/1)**, right frames 700 -> 1451 | 0 | **on** |
| R1b | a corner and a free kick: the taker's single vote is too short, the spell goes to the next team | `team_b.restart_possession`: the team of the player nearest to the ball at the R1 kick keeps it until the opponent's first control vote | -0.0017 (2/4) | +0.0020 (3/0) | off |

(sens) R3c within 5 m: -0.0000 (4/2) -- the 3 m set a priori is kept; the throw-in is found 4.4 m inside the
line and stays unfixed. R1b probably fails on degraded tracking because the tracker often finds the
ball already kicked, when the nearest player is the receiver, not the taker. R3c first used |distance|
and missed a ball found BEYOND the line (x = -6.9 m): a logic fix, made before the IDSSE run.

### 14.2 The supervised team (`supervised/team_model.py`, `scripts/eval_supervised.py`)

- **What is learned:** P(home has the ball) per frame, only on frames that the rules call alive. The
  dead mask, the cleaning and every rule output stay as they are; the rules also feed the model.
- **Features (31):** control votes now and summed over past / next 0.5, 1, 2, 4 s; the last and the
  next vote, decayed by their distance in time (0.5, 2, 8 s); which team is closer to the ball (three
  distance scales, and on a slow ball); the ball's position in home's attacking direction (now and
  last seen); the 8 collective features of the position model; the rules' keeper catch and restart
  team. Every feature changes sign when the teams are swapped and the logistic model has no intercept,
  so P(home | teams swapped) = 1 - P(home): it cannot learn "home is more likely".
- **Decoding:** per alive stretch, a 2-state Viterbi on log P with a cost per change of team. The cost
  is chosen from (0, 2, 5, 10, 20, 40, 80) on the other six matches; it was 5 for all seven.
- **Training:** IDSSE degraded, DFL controlled labels (RULES.md: labels of other games), every 3rd
  frame, L2 = 1, Newton steps (deterministic). Leave-one-match-out for every number below. The final
  model (all seven matches) is `supervised/team_model.json` (31 weights + scales): make_submission_b.py
  only reads it (`supervised_team` in baseline_b.yaml), IDSSE is not needed to regenerate the CSV.

### 14.3 Results on IDSSE (leave-one-match-out; rules = section 13 + R6 + R3c)

```text
              spellF1 rules -> supervised   paired (better/worse)   macroF1           team acc          possessions (true 2705)
degraded      0.4795 -> 0.4977              +0.0177 (6/1)           0.8283 -> 0.8374  0.8053 -> 0.8189  1988 -> 1929
clean         0.6058 -> 0.6324              +0.0262 (7/0)           0.9021 -> 0.9146  0.8532 -> 0.8714  1988 -> 2100
```

Per match and switch cost: `outputs/validation_kc/results_supervised.md`. The gain is 5-10 times that
of any single rule of sections 12-14 and holds on 13 of the 14 match-variants. With cost 0 (no
decoding) it is far worse than the rules: the decoding matters as much as the classifier. Against
this morning's B v3 (0.473 / 0.602) the branch is at 0.498 / 0.632.

The weights are dominated by the timing of the votes (last vote within 2 s: +1.95; within 0.5 s:
-1.37, i.e. a vote that has only just appeared counts less than one that holds) and by the collective
shape (territory +4.6 and defensive line -4.7, strongly correlated with each other).

### 14.4 BAR-ATH

```text
rules only (R1 R4 R5 + R6 R3c)    outputs/run_b3/tikitalia.csv             SHA-256 4a4d80fbaeba2550c8142267eecf9b5903dba72e3b280fd83d1eae7e7631563d
                                  317 possessions; H 0.3309  A 0.2329  D 0.4362; R6 1433 frames, R3c 5672 frames
rules + supervised team           outputs/run_b3_supervised/tikitalia.csv  SHA-256 ba6667b440e42440a62aa679e58041b01f63bcce8fc11a74ed6356dadc8b661e (superseded: section 15)
                                  (--set supervised_team.enabled=true)     286 possessions; H 0.3291  A 0.2348; 6505 alive frames change team
```

Both deterministic (two runs, same hash). The passing ball that R2/R2b could not fix without harm becomes
Barcelona (P 0.94); the ball through four defenders stays Athletic.

Public 5 minutes, structure check only (17 possessions, noise +/-0.36; not used to decide anything):

```text
                          macroF1   spellF1   alive/dead F1   team acc   transF1@25   possessions
section 13 (f4958dca)     0.8643    0.4706    0.9606          0.9013     0.359        17
+ R6 + R3c                0.8643    0.4706    0.9606          0.9013     0.359        17   (no frame of the window changes)
+ supervised team         0.8810    0.6000    0.9606          0.9264     0.400        13
```

### 14.5 Branch, limits, next steps

- Branch: Claude cannot run git on the PC and makes no commit. The files are written to the working
  tree; `supervised_team.enabled` stays **false** in baseline_b.yaml, so the default CSV is the rules
  one, and the supervised CSV lives in its own folder. Turning it on is one line.
- The model is trained on IDSSE tracking degraded by our simulation; BAR-ATH's tracker differs (where
  it loses the ball, team swaps). The public window agrees in direction, but it is 17 possessions.
- Next: (1) learn the dead ball too (D vs alive), with the rule layers as features; (2) richer models
  (interactions, a small network: hmm_v0/nn.py already has a deterministic one) once the logistic
  baseline is settled; (3) a more faithful degradation (ball lost near the lines / in the air / in the
  keeper's hands) so that training and validation look more like BAR-ATH.

## 15. Branch supervised-test: what moves the learned team -- transitions and contacts (01/11/2026)

Goal: better dead-ball limits, less latency, the right frame for each change. Four ideas were tried
and **kept only with a real spellF1 gain on IDSSE** (leave-one-match-out, mean of the 7 matches,
paired). `scripts/eval_supervised.py --sets ...`, `outputs/validation_kc/results_supervised_sets*.md`.

### 15.1 Interpolating the ball through short gaps

Measured first on IDSSE degraded, where the hidden ball is known (straight line between the last and
the next seen position, alive frames):

```text
gap        median error   within 1.5 m (control radius)
< 1 s      0.1 m          99%
1-3 s      0.8 m          68%
3-6 s      3.9 m          19%
6-15 s     9.4 m           9%
```

On BAR-ATH the ball is missing on 40.7% of the graded frames: gaps < 1 s hold 0.1% of them, 1-3 s
none, 3-6 s 3.8%, 6-15 s 21.2%, longer 74.8%. 42% of the possession changes on IDSSE happen with the
ball unseen, only 8% of those in gaps <= 6 s. So interpolating 5-6 s gaps would invent control votes
(19% within the radius) for a small share of the changes; only gaps < 1 s were tried
(`clean_b.interpolate_short_ball_gaps`, `cleaning.ball_short_gaps`): +0.0013..+0.0017 degraded (5/2),
-0.0002..-0.0016 clean -- no real gain, **off**.

### 15.2 New feature groups for the learned team (all odd under the team swap)

| group | features | degraded vs base model | clean vs base model | decision |
|---|---|---|---|---|
| trans | change of the collective shape across the frame: mean over the next 1 / 2 s minus the previous 1 / 2 s of spread, forward velocity, territory, defensive line (8) | **+0.0498 (7/0)** | **+0.0329 (7/0)** | **kept** |
| paper | Vidal-Codina et al. 2022: a contact = the ball within 1.5 m of a player changes direction (> 30 degrees) or speed (> 2 m/s) over 3 frames, signed by his team: now, sums over the past / next 1 and 2 s, last and next contact decayed; a duel (both teams within 1 m) times the last vote (10) | **+0.0243 (6/1)** | **+0.0532 (7/0)** | **kept** |
| edge | the ball continued at most 1 s past its last seen frame / before its next one, and who was closest to it there (4) | -0.0002 (3/4) | -0.0044 (3/4) | dropped |
| inf | sign of the last / next vote without decay (2) | +0.0001 (3/3) | +0.0014 (2/0) | dropped |

Why "trans" works: at a change of possession the two teams swap roles -- the side that defended
stretches and moves forward, the other drops -- and that is visible in the players even when the ball
is not, which is where 42% of the changes happen. Why "paper" works: it separates a ball touched (it
changes direction or speed) from a ball running past a player, the error R2 / R2b could not fix as
hard rules (section 13).

### 15.3 The final learned team: base + trans + paper (leave-one-match-out)

```text
                 rules     base model   + trans + paper    vs rules (b/w)     vs base model (b/w)
degraded         0.4808    0.4985       0.5521             +0.0713 (7/0)      +0.0536 (7/0)
clean            0.6064    0.6326       0.6934             +0.0870 (7/0)      +0.0608 (7/0)
degraded_ts      0.4529    0.4778       0.5296             +0.0767 (7/0)      +0.0518 (7/0)
transF1@25       0.386 / 0.571 (rules) -> 0.501 / 0.702 (degraded / clean)
latency          mean |latency| 9.0 -> 7.0 frames (degraded), 7.9 -> 5.0 (clean)
```

(Means over the 7 matches; the pooled spellF1 of 14.3 was 0.4795 / 0.6058 for the rules.) The
final model (`supervised/team_model.json`: 49 weights, groups trans + paper, switch cost 10 chosen on
the seven matches) replaces the base one of section 14. Against this morning's B v3 (0.473 / 0.602)
the branch is at 0.552 / 0.693 on IDSSE.

### 15.4 BAR-ATH

```text
outputs/run_b3_supervised/tikitalia.csv   SHA-256 1c89b8f25b83c0049875b653a6778f52a627f2bfd19557dac3d24cee5c21e0d8
                                          (two runs, same hash); 261 possessions (rules 317); H 0.3329 A 0.2310 D 0.4362
                                          7680 alive frames change team against the rules
outputs/run_b3/tikitalia.csv (rules)      4a4d80fbaeba2550c8142267eecf9b5903dba72e3b280fd83d1eae7e7631563d (unchanged)
```

Public 5 minutes, structure check only (17 possessions, noise +/-0.36, not used for any decision):

```text
                           macroF1   spellF1   transF1@25   latency   possessions
rules                      0.8643    0.4706    0.359        +3.1      17
base learned team          0.8810    0.6000    0.400        -1.0      13
+ trans + paper            0.8889    0.6815    0.438        +0.7      10
```

### 15.5 Limits and next steps

- The learned team makes fewer possessions (BAR-ATH 261 vs 317 with the rules; public window 10 vs
  17 predicted). IDSSE rewards it, but the count is worth watching: the true IDSSE count is 2705 for
  ~1990 predicted by the rules already.
- Everything is trained on IDSSE degraded by our simulation; BAR-ATH's tracker is different.
- Next: learn the dead ball too (alive vs dead, the rule layers as features) -- the third of the goals
  (dead-ball limits) is still decided by the rules; a richer model (interactions, the deterministic
  network of hmm_v0/nn.py) on the same features.

## 16. State of the best model, step by step, and its theoretical bases (03/11/2026)

The best model on spellF1 so far is the branch configuration of section 15: **the rules decide when
the ball is dead (D), a model learned on IDSSE decides who has the ball (H / A) on the alive frames.**
In the config it is `supervised_team.enabled: true` (default still `false`: the default CSV is the
rules-only one).

### 16.1 Numbers

```text
                                        spellF1 rules   spellF1 best   matches better
IDSSE degraded (7 matches, LOMO)        0.481           0.552          7 / 0
IDSSE clean                             0.606           0.693          7 / 0
IDSSE degraded_ts (simulated swaps)     0.453           0.530          7 / 0
Public 5 min (structure check only)     0.471           0.682          macroF1 0.864 -> 0.889
```

Every IDSSE number is from a model trained on the other six matches (leave one match out). The public
window holds 17 possessions (noise about +/-0.36) and was looked at once, after the choices.
BAR-ATH: `outputs/run_b3_supervised/tikitalia.csv`, SHA-256 `1c89b8f2...`, 261 possessions.

### 16.2 The steps, in order

**0. Input.** ATD tracking (22 players + ball; the ball is missing on 40.7% of the graded frames),
Smart Tagging (throw-ins, corners, goal kicks, free kicks, kick-offs), ATD metadata (position_type,
for the goalkeepers).

**1. Tracking cleaning** (`hmm_v0/clean_b.py`, section 10, 13).
- *Keeper identity*: a keeper slot inside the opponent's box on a stretch of track that did not run
  there is the other keeper (the tracker swaps slots; a jump > 62 m/s is not a person).
- *Team continuity (R5)*: a new fragment continuing an opponent's just-ended track, while its own team
  already has 2 players more, is the opponent's.
- *Goalkeeper fix*: a fragment >= 10 s inside one box and mostly the deepest player is the keeper of
  the team defending that goal.
Basis: physical limits (human top speed) and football constraints (11 per team, keepers in their box).

**2. Dead ball, the c_v3 HMM** (`config_c3`, REPORT_C_v3).
- Rule votes: tens of rules (ball out, ball still, controls, passes, restarts...) vote H / A / D / X;
  a vote of weight w is evidence ln((K-1)p / (1-p)) nats for its class, summed like a naive Bayes.
- Pseudo-labels where the rule evidence is clear (margin >= 1.5 nats; no human decision) train a
  random-feature network (convex training, integer arithmetic: the same CSV on every machine, RULES 3)
  on +/-2 frame windows: P(alive / dead / unseen).
- A Bakis (left-to-right) HMM with 5 states H, A, D, XH, XA (XH / XA: the team still has the ball but
  the tracker does not see it). Minimum durations 0.8 s (possession) and 2 s (D); transitions set
  a priori (about one change every 20 s of play, one stoppage every 30 s); the D exits multiplied by
  50 at detected restarts. Emissions: log-linear combination of the network and the rule evidence,
  weight 0.2 each (consecutive frames are not independent evidence). Only its D is used.
- Smart Tagging restarts anchor the stoppages.
Basis: HMM with Bakis topology (from speech recognition), log-linear evidence pooling,
pseudo-labelling.

**3. Dead ball, the rules on top** (`hmm_v0/dead_b.py`, sections 3, 8, 11, 13, 14).
- D-SLOW: everybody walking (mean speed < 1.1 m/s) with the ball mostly unseen; R4 starts it where the
  ball was last seen.
- Placed ball: ball seen still > 2 s on the pitch, not in a keeper's hands -> D up to the kick.
- Wall: >= 3 standing players of one team close together.
- Ball-out start (Law 9): a ball seen leaving is dead only from the crossing; before it, play.
- R3c: ball lost on its way to a line and found again near (or beyond) the same line -> it went out.
- R1: a Smart Tagging restart ends the stoppage exactly at the kick (ball leaving fast after rest).
- R6: an island of play < 10 s between two stoppages, ball never seen, players slow -> still dead.
Basis: the Laws of the Game (Law 9, restarts), ball physics, the tracker's statistics. Each rule was
switched on only if it improved IDSSE.

**4. Who has the ball: the learned team** (`supervised/team_model.py`, sections 14, 15).
Per alive frame, 49 inputs, all *odd under the team swap* (they change sign if H and A are swapped):
- control votes (nearest player <= 1.5 m, ball < 8 m/s): the frame's vote, the vote sums over
  0.5 / 1 / 2 / 4 s before and after, the last / next vote decayed with time (0.5, 2, 8 s) so that a
  vote still counts while the ball is unseen;
- closeness to the ball at three scales (1, 3, 8 m), the ball's x in H's attacking direction (and its
  last seen value);
- 8 collective features, H minus A: spread, length, width, forward velocity, centroid, defensive line,
  attacking line, mean speed (the team in possession stretches and moves up, the other compacts);
- transitions: "after minus before" (1 s, 2 s) of forward velocity, centroid, defensive and attacking
  lines -- at a change of possession the two teams swap roles, visible without the ball;
- contacts and duels (Vidal-Codina et al. 2022): a contact = the ball near a player turns > 30 deg or
  changes speed > 2 m/s; the last / next contact, contacts in the window, a duel (two opponents
  within 1 m of the ball) times the last vote;
- two rule signals: keeper catch, team taking the restart.

*Symmetric logistic regression*: no intercept, so P(H | teams swapped) = 1 - P(H) exactly; it cannot
learn "home has the ball more", only the evidence. Newton steps with L2 penalty (= MAP with a Gaussian
prior on the weights), 184k frames of the 7 IDSSE degraded matches, labels = DFL BallPossession. The
largest standardized weights: centroid and defensive line, then the transitions, then the last / next
control vote -- the teams' shape matters most, the touches correct it.

*2-state Viterbi per alive stretch*: the most likely H / A sequence with a cost of 10 nats per change
(an HMM with a transition probability of about e^-10 per frame). A change needs ~7 frames (0.3 s) of
P = 0.8 against, or ~1 s of P = 0.6: a single frame never flips the team. The cost was chosen on the
held-out matches.

Basis: logistic regression / MAP, equivariance by construction, HMM / Viterbi, football tactics
(transition phases), the contact definition of the paper.

**5. Validation.** Leave one match out on IDSSE; IDSSE degraded with the BAR-ATH tracker's statistics
(ball gaps resampled from its run lengths, tracks broken like its slots, Smart Tagging simulated with
its recall) -- never its ground truth; paired per-match comparisons with the official scorer; the
public window looked at once at the end.

### 16.3 What is left

- The dead-ball HMM has hand-set transitions, and the team is decided by a separate 2-state HMM that
  never sees the transitions to D: the possession and the stoppages are decided one after the other.
- Next: one H / A / D model with transitions and durations counted on IDSSE (MAP), the rules and the
  learned P(H) as emissions, forward-backward smoothing (section 17, when done).
- The learned team is off by default; it makes fewer possessions (261 vs 317 on BAR-ATH).

## 17. One H / A / D HMM with forward-backward smoothing and minimum durations (03/11/2026)

Branch `supervised_model_v0`, evaluation only (`scripts/eval_hmm3.py`, `supervised/hmm3.py`); the CSV
is not changed yet.

### 17.1 What it is

Section 16 decides the stoppages first (rules) and the team afterwards (2-state Viterbi): the two
never see each other. Here one HMM decides H / A / D together:

- **states** H, A, D, each a Bakis chain of minimum duration (c3: 0.8 s possession, 2 s stoppage,
  the a priori values of the c_v3 HMM; q2: the 2% quantiles of the IDSSE spells, 0.5 s / 2.7 s;
  none);
- **transitions** counted on the six training matches (rate of each change after the minimum,
  Dirichlet pseudo-count 1 = MAP), not set by hand;
- **emissions** w * (log P(class | frame) - log prior), with P(H), P(A), P(D) from two classifiers:
  the learned team of section 15 (P(H | alive)) and a new **learned dead ball**: a logistic
  regression (intercept, L2, Newton) on 40 inputs -- the final rule mask, the c_v3 D, every rule
  layer (D-SLOW, placed ball, wall, R4, R3c, R1 play / extend, R6), the rule mask over +/-1, 3, 8 s,
  the ball (seen, speed, time unseen before / after, last seen outside the lines, distance to the
  line), the players' mean speed and its +/-2, 5 s means, the Smart Tagging windows, the time
  to / since a tagged kick and any kick from rest; labels = DFL BallStatus;
- **decoding** forward-backward posteriors frame by frame (mpm) or Viterbi.

The alternative "rules' mask as P(D)" (eps 0.02 / 0.1 error rate) is scored too.

### 17.2 Results (leave one match out, setting for each match chosen on the other six)

```text
spellF1                         rules    rules D + learned team    H / A / D HMM          
IDSSE degraded                  0.4808   0.5521                    0.5895  +0.0374 (7/0)
IDSSE clean                     0.6064   0.6934                    0.7137  +0.0202 (6/1)
IDSSE degraded_ts               0.4529   0.5296                    0.5639  +0.0343 (7/0)
transF1@25 degraded / clean     0.386 / 0.571   0.501 / 0.702      0.546 / 0.723
mean |latency| degraded / clean 9.0 / 7.9       7.0 / 5.0          7.2 / 5.1 frames
```

The chosen setting is the same for all seven matches: learned dead ball, c3 durations, w = 1.0,
Viterbi. Around it the result is flat (degraded: w 0.75-1.0 Viterbi or 0.5 mpm all within 0.002; w
1.5-3 lower), so it is not a lucky point.

Where the gain comes from:
- the learned dead ball: per-frame alive/dead accuracy 0.942 vs 0.914 of the rules on degraded
  (7/7 matches), 0.981 vs 0.971 on clean; with the rules' mask as P(D) the HMM reaches only 0.563
  degraded / 0.696 clean (+0.011 / +0.003 over section 16);
- the joint decision with counted transitions: the rest.
- forward-backward (mpm) and Viterbi are equivalent at their best weights (0.5 and 1.0).

### 17.3 Notes

- No rule is changed and nothing is set on BAR-ATH (RULES.md untouched): like section 14, it is
  trained on the IDSSE DFL labels.
- The loops are compiled with numba when installed (eval speed); without it the same code runs in
  plain Python (slower, same numbers).

### 17.4 BAR-ATH and the public window (one check, after the choice)

`supervised/hmm3_model.json` (dead-ball classifier and transitions trained on all seven degraded
matches; the team model is `supervised/team_model.json`), `hmm3.enabled: true` in
`config_b/baseline_b.yaml` (default false):

```text
outputs/run_b3_hmm3/tikitalia.csv   SHA-256 2224cd618fe16236d8ce2383b1dcc76f040ac113d9ed1cf7b5393cd975f4d312
                                    (two runs, same hash); 301 possessions; H 0.3485 A 0.2523 D 0.3992
                                    (rules 317 poss., D 0.4362; learned team 261 poss., D 0.4362)
```

Public 5 minutes (official scorer, 17 possessions, noise about +/-0.39):

```text
                           macroF1   spellF1   transF1@25   F1 of D   team acc   possessions
rules                      0.8643    0.4706    0.359        0.822     0.901      17
rules D + learned team     0.8889    0.6815    0.438        0.822     0.940      10
H / A / D HMM              0.8521    0.6194    0.368        0.750     0.939      14
```

The HMM is better than the rules but below the learned team of section 15 on the public window.
The difference is in the dead ball (F1 of D 0.75 vs 0.82; team accuracy unchanged): on the whole
BAR-ATH match the learned dead ball calls 3.7 points of frames fewer D than the rules (0.399 vs
0.436), while on IDSSE it was better than the rules on every match. The likely cause is the domain
gap: the dead-ball classifier learned the IDSSE degraded signals (simulated ball gaps, simulated
Smart Tagging), and BAR-ATH's real tracker and tagging differ. With 17 possessions the gap is within
the noise, so it is a warning, not a verdict; no setting was changed after looking.

- Next, still chosen on IDSSE only: make the degradation closer to BAR-ATH where the dead ball is
  concerned (ball lost near the lines and in the air, the real Smart Tagging lags), or use the
  rules' mask inside the HMM (+0.011 / +0.003 on IDSSE, the dead ball exactly as the rules say).

### 17.5 Ball interpolation before the HMM (03/11/2026)

Question: does filling short ball gaps by a straight line (section 15) before everything help the
dead ball and spellF1 once the HMM smooths the states? `eval_hmm3.py --interp-s 1 | 3`, same
protocol, paired per match against the HMM without interpolation:

```text
                       spellF1 HMM        rules D + learned team   alive/dead accuracy (learned)
gaps <= 1 s degraded   +0.0004 (4/3)      +0.0053 (5/2)            +0.0011
gaps <= 1 s clean      +0.0037 (6/1)      -0.0003 (3/4)            +0.0015
gaps <= 3 s degraded   +0.0036 (4/3)      +0.0042 (5/2)            +0.0023
gaps <= 3 s clean      +0.0047 (6/1)      +0.0006 (5/2)            +0.0029
```

At most +0.005, not consistent on degraded (4 better / 3 worse): no effective gain, left off. Why:
75% of the missing ball frames on BAR-ATH are in gaps longer than 15 s, which a straight line cannot
fill; the HMM already bridges the short gaps through its minimum durations and transitions; and a
straight line between two seen points inside the pitch can hide an exit (throw-in, ball out and
back), the case the dead ball needs most.

## 18. The referee's whistle from the match audio (03/11/2026)

The tactical video has an audio track (AAC 48 kHz stereo, the program feed: stadium + commentary).
Extracted once with ffmpeg (mono, 16 kHz, FLAC, in `data/`: derived data of the match, never in
git), read by `hmm_v0/whistle.py`, `scripts/detect_whistles.py` -> `outputs/whistles/` (ignored).

### 18.1 Detector (signal processing only, no labels, no training)

- Spectrogram per video frame (1024-point FFT, hop 640 samples = 1 frame, so audio frame i =
  video frame i; the audio is 2 frames shorter than the video, start at 0).
- The referee's whistle on this feed is two flat lines sounding together (two-chamber whistle). Per
  frame: the strongest pair of narrow peaks f1 in 3.3-4.7 kHz and f2 300-600 Hz above it, scored by
  the weaker of the two local SNRs (peak vs the bins 2-10 away).
- The pitch is calibrated on the match itself (mode of the strong two-tone frames, origin
  "distribution"): **f1 = 3753 Hz, f2 - f1 = 441 Hz**.
- A blast: >= 3 frames with both tones >= 12 dB above their neighbourhood at that pitch (+/-150 Hz,
  gap +/-70 Hz); blasts closer than 0.32 s merged. Every constant is a priori.

Commentary and crowd make tonal lines too, but they glide and come in harmonic stacks: the
two-tone, steady test rejects most of them. A visual look at 12 random detections (spectrograms,
a debugging check, not labels) showed 9 clear whistles, 3 voice / crowd lines.

### 18.2 What it finds on BAR-ATH (no ground truth)

```text
95 blasts, 80 on graded frames:
  inside a dead span of the rules (restart whistle, card, during the stoppage)   47
  ball alive by the rules, a stoppage of the rules starting within 3 s           12
  ball alive by the rules, a stoppage starting 3-10 s later                      11
  ball alive by the rules, no stoppage within 10 s                               10
lag whistle -> rules' stoppage: median 1.8 s (0.1 ... 9.7 s)
```

The referee does not whistle for most balls out of play (throw-ins, goal kicks, corners), so the
whistle covers fouls, offsides, goals, kick-offs and restarts of free kicks -- the stoppages with the
ball inside the pitch, where the rules are slowest (D-SLOW). The 11 stoppages starting 3-10 s after
a whistle are the latency the whistle can remove.

### 18.3 Rules, not a simulation

IDSSE has no audio. Simulating whistles there (from the DFL events) would mean inventing how they
sound, when they come and how many false alarms they have, then tuning on that invention -- the same
kind of simulation gap that hurt the learned dead ball on the public window (17.4). The team chose
rules from a fact of the Laws of the Game instead, with a priori constants
(`hmm_v0/dead_b.whistle_dead`, config `whistle`, applied after R6, before the team):

- **R7a** a blast while the ball is alive (rules' mask) starts a stoppage at the blast if within
  3 s the players really slow down (lowest 1 s mean speed <= 0.7 x the second before, or
  <= 1.1 m/s): dead from the blast to the rules' next stoppage if it starts within 15 s, else to the
  next kick from rest within 30 s, else nothing.
- **R7b** a blast inside a stoppage that has lasted >= 5 s announces the restart: the stoppage ends
  at the first kick from rest within 2 s of the blast's end, or 2 s after it.

### 18.4 Checks without labels on BAR-ATH: the broadcast and the players

The broadcast (aligned to 1-2 frames, section 4) shows a cut (replay, close-up, crowd) far more
often during stoppages; it does not depend on the audio nor on the rules:

```text
frames (graded, broadcast covered)            n       broadcast cut   player speed   ball seen
rules alive (reference)                   89168           0.17            2.19 m/s      0.83
rules dead  (reference)                   66939           0.68            1.09 m/s      0.28
R7a added to the dead ball                 3241           0.50            1.30 m/s      0.45
R7b given back to play                    13717           0.71            0.98 m/s      0.23
R7b with a kick required (6 cases)          415           0.53            0.78 m/s      0.66
```

R7a adds frames that look dead (24 stoppages, 9 blasts refused for no slowdown). R7b gives back
to play frames that look as dead as the rules' stoppages: in this feed the whistle inside a
stoppage is not reliably followed by the restart within 2 s (cards, substitutions, whistles while
the ball is placed), and with a kick required it acts only 6 times. Decision, made before looking
at any label: **R7a on, R7b off** (`whistle.restart: false`).

### 18.5 Public window (one check, after the choice)

```text
                                  macroF1   spellF1   transF1@25   F1 of D   D recall   latency   poss.
rules                             0.8643    0.4706    0.359        0.822     0.761      +3.1      17
rules + R7a                       0.8856    0.4571    0.439        0.869     0.848      -1.0      18
rules D + learned team            0.8889    0.6815    0.438        0.822     0.761      +0.7      10
rules D + learned team + R7a      0.9109    0.6963    0.545        0.869     0.848      -2.9      10
H / A / D HMM (+ R7a: identical)  0.8521    0.6194    0.368        0.750     0.632      +5.1      14
```

The whistle does what it was designed for: the dead ball starts on time (D recall 0.76 -> 0.85 at
the same precision, transF1@25 +0.11 on the learned team). spellF1 moves within the noise (17
possessions, +/-0.39); macroF1 0.889 -> 0.911 (bootstrap 5-95% 0.80-0.96).
`outputs/run_b3_whistle_sup/tikitalia.csv` SHA-256
`c677e47f333174108fc1a8a15806a475fd5d5896baa909f49b401597ab9c96fb` (two runs, same hash),
253 possessions, D 0.458.

Reproducing it needs the audio extracted with ffmpeg and `scripts/detect_whistles.py` (writes
`outputs/whistles/BAR-ATH_whistles.json`); run on the same FLAC the detection is the same.

## 19. Do results on IDSSE carry over to BAR-ATH? What a non-gain means (04/11/2026)

The numbers of sections 19-23 use the sorted BAR-ATH run lengths (`outputs/validation_kc/barath_run_lengths.npz`,
`validation/export_barath_stats.py`: lengths only, no frame). Sorting changes which lengths each seed draws,
so the c_v3 labels and every baseline were regenerated. Seed 2027 unless stated.

Compliance (RULES.md, mail of 03/11): nothing here is chosen on BAR-ATH or on the public ground truth. The run
lengths are question 4 to the organisers, still open; the state-loss scenarios are set a priori, in round values.

### 19.1 Answer

**No: a non-gain on IDSSE does not imply a non-gain on BAR-ATH or on the public ground truth.** It means
"IDSSE gives no evidence for it". How far it transfers depends on why there was no gain:

| why no gain on IDSSE | example | what it says about BAR-ATH |
|---|---|---|
| the error does not exist on IDSSE | keeper identity: 0 corrections on IDSSE, 1324 frames changed on BAR-ATH | nothing |
| the rule does not fire on degraded IDSSE | keeper catch: 0 frames; keeper hold: 0 frames on degraded, 78 per match on clean | nothing |
| it depends on how the ball is lost | R3: negative with random gaps, about 0 when the ball is lost in stoppages (19.3) | little |
| the gain is smaller than the noise of the degradation | fragment merging, R1b (19.5) | nothing either way |
| negative on every scenario and on clean tracking | R2, R2b (19.3) | it probably transfers |

The public window cannot settle it either (19.4).

### 19.2 The degradation is blind to the state of play

`validation.degrade` draws ball gaps without looking at the game, so on degraded IDSSE the ball is seen
**more** when it is dead than when it is alive (58% vs 49%: only live balls fly). On BAR-ATH (18.4) the ball
is seen on 83% of the frames the rules call alive and on 28% of those they call dead. That reading is partly
circular, because some rules use the unseen ball. The learned dead ball took the IDSSE pattern: `ball_seen`
has weight +0.72 towards D in `hmm3_model.json`.

New variants `degraded_sd5` / `degraded_sd8` (`validation.state_loss`), same `degrade()` realization, so paired:
each dead span is hidden with probability 0.5 / 0.8, each live gap is given back with probability 0.3 / 0.6.
Ball seen, alive / dead: 0.59 / 0.29 and 0.69 / 0.11.

`scripts/eval_transfer.py`, leave one match out, spellF1:

```text
                       rules    learned team                        H/A/D HMM
trained on:                     degraded  sd8      both             degraded  sd8      both
test degraded          0.4847   0.5568    0.5583   0.5587           0.5770    0.5537   0.5766
test degraded_sd5      0.4781   0.5479    0.5465   0.5496           0.5731    0.5820   0.5826
test degraded_sd8      0.4874   0.5520    0.5523   0.5538           0.5795    0.6132   0.5970
test clean             0.6061   0.6996    0.7034   0.7029           0.7183    0.6995   0.7199
```

- The hypothesis "the state-dependent loss makes the HMM lose to the learned team" is **rejected**: trained on
  plain degraded, the HMM stays ahead on every scenario.
- The HMM does write less D than the rules once the ball is lost in stoppages (0.412 vs 0.431 on sd8), the
  direction seen on BAR-ATH (0.399 vs 0.436).
- **Training on the wrong kind of loss has a real cost**: on sd8, HMM trained on degraded 0.5795, on sd8 0.6132;
  on degraded, trained on sd8 0.5537, on degraded 0.5770. Section 23 follows from this.

### 19.3 Rules across scenarios

`scripts/eval_rule_transfer.py`, rules-only labels, rule on minus off, spellF1 (better / worse matches). Placed
ball and ball-out start are not in the table: `eval_rules_b.labels` applies them without reading `enabled`.

```text
                  in model  degraded         sd5              sd8              clean
R6 dead islands   yes       +0.0050 (7/0)    +0.0050 (7/0)    +0.0071 (7/0)    0
R1 restart kick   yes       +0.0010 (4/3)    +0.0021 (4/3)    +0.0052 (6/1)    +0.0008 (4/3)
R3c unseen exit   yes       +0.0007 (3/0)    +0.0005 (3/1)    +0.0021 (5/1)    0
R4 slow from loss yes       +0.0000 (3/2)    +0.0008 (5/1)    +0.0006 (4/1)    0
keeper hold       yes       0                0                0                +0.0008 (3/2)
keeper catch      yes       0                0                0                0
R1b restart team  no        +0.0002 (3/2)    +0.0017 (5/2)    +0.0003 (5/2)    +0.0023 (3/0)
R3 unseen (loose) no        -0.0015 (2/5)    -0.0023 (3/4)    +0.0008 (3/4)    0
R3b + slow        no        +0.0005 (2/3)    -0.0011 (3/4)    +0.0010 (4/3)    0
R2 relative speed no        -0.0048 (1/6)    -0.0034 (4/3)    -0.0035 (4/3)    -0.0101 (2/5)
R2b pass-by       no        -0.0036 (3/4)    -0.0054 (1/6)    -0.0068 (1/6)    -0.0132 (1/6)
```

- **R3**: frames changed that are right, off -> on: 5449 -> 2161 on degraded, 9591 -> 13136 on sd8. The sign of the
  frames flips, the score stays about 0: not a gain, but its rejection in 13.2 owed much to the random gaps.
- **R2, R2b** are negative everywhere: the most reliable non-gains. They fix cases seen on BAR-ATH, but on every
  scenario they remove more good votes than bad ones.
- **R1, R3c, R6** grow as the ball is lost more in stoppages: their small IDSSE gains may understate BAR-ATH.
- **R1b**: with the quantile fallback of the first run it looked like a candidate (+0.0028 on sd5); with the real
  lengths it is +0.0002 to +0.0017. No decision: it stays off.

### 19.4 What a 5-minute window can see

`scripts/eval_windows.py`: 637 windows of 5 minutes of IDSSE stand in for the public ground truth. Possessions per
window: median 21, 5-95% 12-29, so 17 is ordinary.

```text
pair (degraded)         whole matches   window sd   window > 0   window < 0   window <= -0.06
learned team - rules    +0.072          0.083       79%          19%          5%
HMM - rules             +0.092          0.098       82%          18%          6%
HMM - learned team      +0.020          0.059       63%          34%          8%
```

- A true gain of 0.02 shows the wrong sign in a third of the windows. The public result for the HMM (17.4: -0.062
  against the learned team) happens in about 1 window in 12 even if the HMM is truly better by 0.02: a weak warning,
  not a verdict. The public +0.21 of the learned team over the rules is about 1.8 window sd above what IDSSE predicts.
- A rule worth 0.005 (R6) changes no frame in 66% of the windows; R4 and R3c change none in 96%. A flat public score
  says nothing about them, and neither does a moved one.

### 19.5 The noise of the degradation itself

The same pipeline, same real lengths, only the degradation seed changed (2027 -> 2028, `degraded`):

```text
                    seed 2027   seed 2028
rules               0.4847      0.4944
learned team        0.5568      0.5633
H/A/D HMM           0.5770      0.5987
merge, HMM          +0.0010 (5/1)   -0.0017 (2/5)
refine +/- 2 s, HMM +0.0016 (4/3)   +0.0036 (6/1)
```

The HMM moves by 0.022 with no change at all; a gain of about 0.002 can change sign. **So one seed cannot decide a
gain under about 0.005**, and the earlier single-seed rule decisions of that size (13, 14) are not settled by IDSSE.
Consequence for choosing: several scenarios and a seed not used for the choice.

## 20. Where the points are, and the latency of the changes (04/11/2026)

`scripts/eval_oracles.py`: each fix uses the truth, so it is an upper bound, not a method. Degraded, HMM, spellF1 0.577:

```text
every change on the right frame, within +/- 1 s    +0.023
every change on the right frame, within +/- 2 s    +0.051
fragments merged                                   +0.030
false changes removed                              +0.017
spells covering several true spells cut            +0.163
missing changes added                              +0.212
```

Of the true possessions, 53% are matched at IoU 0.5; 28% are lost or wrong, 13% merged with a neighbour, 5% fail on
the limits, 1% are split. The HMM predicts 303 possessions per match against 386 true. **The largest margin is in the
changes the model does not find, not in dating the ones it finds.**

`supervised/refine.py`: a conditional logit over the frames within +/- W of each predicted change, one model each for
H<->A, alive->D and D->alive, inputs signed towards the new team. Same spells and same order: only the frame moves.
Trained on the other six matches. A first version used plain Newton steps that diverged at +/- 2 s on D->alive
(-0.027); the steps are now damped.

On the HMM trained on both degradations (23), spellF1 against no refinement:

```text
                 seed 2027, W = 2 s    seed 2027, W = 1 s    seed 2028, W = 2 s
degraded         +0.0050 (6/1)         +0.0042 (5/1)         +0.0051 (6/1)
degraded_sd5     +0.0091 (7/0)         +0.0047 (4/3)         not run
degraded_sd8     +0.0104 (7/0)         +0.0031 (4/2)         +0.0083 (6/1)
clean            +0.0120 (7/0)         +0.0086 (6/1)         +0.0094 (7/0)
```

Degraded, seed 2027: mean absolute latency 7.8 -> 4.9 frames, transF1@12 0.397 -> 0.462, transF1@25 0.529 -> 0.543.
W = 2 s kept (`supervised/refine_model.json`, trained on all seven matches of both degradations).

## 21. Fragmented possessions (04/11/2026)

`scripts/eval_refine.py`: a spell shorter than `team_s`, or a D shorter than `dead_s`, between two spells of the same
team takes their label; thresholds for a match chosen on the other six.

```text
HMM trained on degraded    seed 2027        seed 2028
degraded                   +0.0010 (5/1)    -0.0017 (2/5)
HMM trained on both
degraded                   +0.0029 (5/2)    -0.0003 (4/3)
degraded_sd8               +0.0026 (3/4)    +0.0038 (6/1)
clean                      +0.0048 (6/1)    +0.0014 (5/2)
```

No reliable gain, the sign changes with the seed: **left out**. It fits section 20: the models make too few
possessions, and few true ones are split. Writing X on the same frames gives the same spellF1 and no better macroF1.

## 22. A public ball detector on the video: preparation (04/11/2026)

**Not run and not switched on. Question 3 to the organisers is still open.** Nothing was run on any video here, and the
code path through a real RF-DETR model is untested: the tests use a fake detector.

- `ball_detector/detector.py`: RF-DETR, overlapping 640-pixel tiles because a ball of a few pixels is lost if the frame
  is resized. The `rfdetr` package and its Nano to Large checkpoints are Apache 2.0; the Ultralytics weights are AGPL.
- `scripts/eval_ball_detector.py`: chooses the score threshold on external annotated video only (lowest threshold with
  precision >= 0.9, a priori) and writes `config_b/ball_detector.json`.
- `scripts/detect_ball_video.py`: run on the PC, keeps all candidates in `outputs/ball_detector/`, which is ignored by
  git and must not be shared; no threshold is chosen there.
- Datasets: SoccerNet-Tracking (ball annotated, broadcast, public via the SDK; its data licence still has to be read
  before use) and SoccerTrack v2 (CC BY 4.0, fixed panoramic camera, closest to the tactical view; its ball track is
  described as a reconstruction, so it is weaker as truth).
- Still missing: the fusion with the tracking (the homography maps pitch to pixels, so its inverse gives the pitch
  position, wrong for a ball in the air), and an IDSSE measure of what a better ball is worth.

## 23. The model chosen on IDSSE: HMM trained on both degradations, changes refined (04/11/2026)

Nothing else is known about how BAR-ATH loses the ball without using it to choose, so the models are trained on the two
kinds of loss together (`degraded` + `degraded_sd8`, the same weight) instead of betting on one.
`supervised/team_model_mix.json`, `supervised/hmm3_model_mix.json`, `supervised/refine_model.json`; the earlier models
stay, so their CSVs can still be regenerated. All defaults stay off.

Against the HMM trained on degraded alone, spellF1 (better / worse matches):

```text
training on both   seed 2027 (used to choose)   seed 2028 (not used)
degraded           -0.0004 (4/3)                -0.0077 (0/7)
degraded_sd5       +0.0095 (7/0)                not run
degraded_sd8       +0.0175 (7/0)                +0.0134 (7/0)
clean              +0.0015 (4/3)                -0.0005 (3/4)
```

**It is a trade, not a free gain**: the fresh seed shows a real cost of up to 0.008 if the ball is lost at random, for a
gain of 0.013-0.018 if it is lost in stoppages. The evidence available points to the second (19.2), though it is partly
circular.

The whole model against the previous candidates, seed 2027, same matches:

```text
                 HMM both + refine   vs rules D + learned team   vs HMM (17)
degraded         0.5816              +0.0249 (7/0)               +0.0047 (6/1)
degraded_sd5     0.5916              +0.0437 (7/0)               +0.0185 (7/0)
degraded_sd8     0.6074              +0.0554 (7/0)               +0.0280 (7/0)
clean            0.7319              +0.0322 (7/0)               +0.0135 (7/0)
```

Seed 2028: 0.5960 degraded, 0.6222 sd8, 0.7261 clean. The refined models also load and run on the IDSSE caches with the
same feature names; `make_submission_b.py` itself was not run here, because it needs BAR-ATH.

```powershell
.\venv\Scripts\python.exe HMM_first_imp\scripts\make_submission_b.py --set hmm3.enabled=true --set hmm3.model=supervised/hmm3_model_mix.json --set supervised_team.model=supervised/team_model_mix.json --set refine.enabled=true --out-dir outputs/run_b3_hmm3_mix_refine
```

### 23.1 Public window (one check, after the choice)

`outputs/run_b3_hmm3_mix_refine/tikitalia.csv`, SHA-256
`bd690f2bdf6bd1a4fcd7c470e3a97dfa37520d0234da0ba26f8bd96c261b173c`. Official scorer, 17 possessions, not used to decide.
At Beppe's request it was scored in the cloud session, on a temporary copy of the CSV and of the public ground truth,
deleted afterwards; only the scores were kept.

```text
                           macroF1   spellF1   transF1@25   F1 of D   D recall   team acc   latency   poss.
rules                      0.8643    0.4706    0.359        0.822     0.761      0.901      +3.1      17
rules D + learned team     0.8889    0.6815    0.438        0.822     0.761      0.940      +0.7      10
H / A / D HMM (17)         0.8521    0.6194    0.368        0.750     0.632      0.939      +5.1      14
HMM both + refine          0.8168    0.5576    0.333        0.668     0.553      0.932      -1.3      16
```

spellF1 bootstrap 5-95%: 0.36-0.74. macroF1: 0.63-0.90. D share: 0.128 predicted, 0.196 true. The pipeline looks correct: 16
possessions against 17, team accuracy as before, latency near zero.

**The result goes against IDSSE, and it cannot be waved away as noise.** The gap to the learned team is -0.124; on 637 IDSSE
windows, the HMM trained on both (without refinement) falls that far behind the learned team in 1-2% of the windows. Against
the earlier HMM (-0.062) it is 2-4%. The loss is in the dead ball: precision holds (0.845), recall falls, and the team is
unchanged. Each learned step has lowered D recall on this window, which is the direction already noted in 17.4. So training
on both losses did **not** fix what the scenario was meant to fix, and the argument for it in section 23 is weakened.

Nothing is changed because of this: the window may not choose between versions (RULES.md 2). What it does is point at the
learned dead ball as the part to study again, on IDSSE and without using these numbers as a target.

Limits and next steps:
1. The largest margin is the missing changes (+0.21 at most): the next step, on the signals that make a change visible.
2. The state-loss scenarios are round guesses. The question to the organisers on BAR-ATH tracking statistics decides
   whether they may be measured instead. If the answer is no, the run lengths and the fallback quantiles of the
   degradation must go too, and everything trained on them has to be redone.
3. The whistle (18) stays off, waiting for the answer on the audio.

## 24. How much a better ball is worth, and why more changes cannot simply be bought (04/11/2026)

IDSSE only. Models trained on both degradations, leave one match out, seed 2027.

**A better ball.** Variants `degraded_gb5` / `degraded_gb10` give back half / all of the gaps of the live ball, at the true
position, except where the ball is too high or too far out (`validation.state_loss`, `dead_hide` 0):

```text
                      ball seen   rules    learned team   H/A/D HMM
degraded              53%         0.4847   0.5587         0.5766
half of the gaps      63%         0.5201   0.6064         0.6197
all of the gaps       73%         0.5536   0.6495         0.6641
clean tracking        100%        0.6061   0.7029         0.7199
```

About **+0.04 of spellF1 for every 10 points of ball seen**. It is an upper bound: a detector adds noise and false balls, and
its gaps are not whole stretches. The oracle of 9.1 (+0.03, with the logic of that time) no longer holds: the learned
models use the ball far better. A ball detector moves up to first priority.

**Where the changes are lost** (`scripts/diag_errors.py`, degraded, HMM trained on both). Of 1995 true H<->A changes, 53% have a
predicted change to the same team within 2 s:

```text
ball seen around the change    none 42% of changes: 40% found    all 26%: 63% found
next possession lasts          < 1 s 7%: 22%   1-2 s 18%: 30%   2-3 s 14%: 48%   5 s or more 42%: 67-71%
ball speed, height, zone       no effect on the share found
```

Possessions found at IoU 0.5 by share of the possession with the ball seen: under 25%, 48% of the possessions: 38% found; over
75%: 63%. True stoppages: 89% covered, worst with the ball unseen (77%) and when short (under 5 s: 28-54%).

**Buying more changes does not work** (`scripts/eval_more_changes.py`): the H<->A jump rate of the HMM multiplied by 1.5 to 12 gives
more possessions (301 -> 325) and a lower spellF1 on degraded (-0.001 to -0.004, mostly worse matches), sd8 likewise, flat on
clean. The missing changes are a problem of signal, not of a threshold: they have to be seen.

## 25. Running the ball detector on the PC (04/11/2026)

`scripts/detect_ball_video.py` runs on the PC that holds the data, in its own environment (`venv_yolo`), so the versions behind
the CSV do not change. `--only-missing` keeps to the graded frames with no tracked ball (about 65,000). The video stays on the PC.

Check of the generic COCO model (`scripts/check_ball_candidates.py`, counts only, against the tracking's ball projected on the
video; no ground truth, no threshold chosen from it). One frame per second of the graded match, 6354 frames, 0.14 s per frame on an
RTX 3090: with a score of 0.5 or more it is near the tracked ball on 38% of the tracked frames and elsewhere on 45% (precision about
46%); at 0.7, 23% and 22%. Partly pessimistic: the tracked ball can be wrong, and a high ball projects badly. On SoccerNet-Tracking
(`scripts/eval_ball_detector.py`, 10 sequences of the train split, truth annotated): precision 82% / recall 60% at 0.5, 90% / 30% at
0.8. The two are not measured alike, but the wide tactical view is clearly harder. The generic model is not used as it is.

## 26. Fourth batch of rules, measured on IDSSE (04/11/2026)

Only general rules enter the code, with no frame or match constant, every number set a priori.
All are **off**. Rules-only labels, rule on minus off, spellF1 (better / worse matches), two degradation seeds.

```text
                                         seed 2027                               seed 2028
                                         degraded     sd8          clean         degraded     sd8          clean
R8  everybody running = play             -0.0031(0/7) -0.0019(1/6) -0.0057(0/7)  -0.0015(1/6) -0.0018(1/6) -0.0058(0/7)
R9  fast ball on the pitch = play        -0.0014(1/4) -0.0006(1/2) +0.0055(6/1)  +0.0004(3/2) -0.0005(2/5) +0.0055(6/1)
R10 ball unseen >= 15 s, slow = stoppage -0.0134(0/7) +0.0104(6/1)  0            -0.0064(1/6) +0.0081(6/1)  0
D-SLOW also with the ball seen           -0.0006(3/4) -0.0049(0/6) -0.0080(0/7)  -0.0001(3/4) -0.0029(1/6) -0.0079(0/7)
D-SLOW ball seen, below 0.9 m/s          -0.0030(1/6) -0.0084(0/7) -0.0016(1/6)  -0.0025(1/6) -0.0086(0/7) -0.0016(1/6)
```

- **R8, D-SLOW with the ball seen**: dropped. Negative on every scenario and on clean tracking, so this is the reliable kind of
  non-gain (19.1). R8 also makes fewer frames right on all scenarios: players do run during stoppages.
- **R9**: about 0 with a poor ball, +0.0055 (6/1 on both seeds) with a clean one, with many more frames right (59 -> 5560 on clean).
  It depends on seeing the ball: to measure again once the detector is in.
- **R10**: the clearest case of 19.1. Its sign is the sign of the assumption: -0.006 to -0.013 if the ball is lost at random, +0.008
  to +0.010 if it is lost in stoppages. IDSSE cannot decide it, because the two scenarios are our own. Off until there is a way
  to tell which one BAR-ATH is without choosing on it.
- Rules-only measure: a rule can behave differently inside the learned models; those that stay candidates must be measured there.
- Still to write: a stoppage that ends only with a kick while a keeper holds the ball, a ball slow in the box with the keeper not tracked,
  the save in two stages. Degraded IDSSE barely exercises keeper rules (0 frames for hold and catch, 19.3), so a variant is needed first.

## 27. How good must a ball detector be? (04/11/2026)

IDSSE only. Variants `degraded_y*` (`validation.ball_fill`): gaps cut into stretches of about 0.5 s, a share `fill` of them gets a
detection, right with probability `precision` (true position + 0.5 m noise), otherwise a false ball staying in one random place on
the pitch. Balls too high or too far out are never detected; the tracking's ball is never touched. Ball seen: 53% on degraded, 65% with
fill 0.3, 77-79% with fill 0.6. HMM, spellF1 against 0.5766 on degraded with no detector (better / worse matches):

```text
detector                    models not retrained      models retrained on this kind of input
fill 0.3, precision 0.9     0.5471  -0.0295 (1/6)     0.6084  +0.0318 (7/0)
fill 0.6, precision 0.9     0.5283  -0.0484 (1/6)     0.6265  +0.0499 (7/0)
fill 0.3, precision 0.7     0.5308  -0.0458 (1/6)     0.5904  +0.0138 (4/3)  (trained on fill 0.3, precision 0.9)
fill 0.6, precision 0.7     0.5120  -0.0646 (1/6)     0.6118  +0.0352 (7/0)  (trained on fill 0.6, precision 0.9)
fill 0.6, precision 0.5     0.4944  -0.0822 (0/7)     0.5951  +0.0185 (5/2)  (trained on fill 0.6, precision 0.7)
```

- **Feeding a detector to the current HMM makes it worse.** Scattered balls break what it learned (D share 0.30-0.38, 370-411
  possessions). Trained on such input it recovers (D share 0.42-0.43, 306-326 possessions).
- The learned team alone, not retrained, moves little: +0.028 at precision 0.9, about 0 at 0.7, -0.025 at 0.5.
- **The retrained models work only on that input**: an HMM trained on fill 0.6 / precision 0.7 scores 0.5422 with no detector.
- **Correction to section 24**: +0.04 per 10 points of ball seen was the whole-gap upper bound. With scattered detections it is about
  +0.02 to +0.03.
- **The false balls in that table are optimistic.** Sitting at random on the pitch they are seldom near a player, while real ones (heads,
  boots) can be, and create false control votes. So the worst case was simulated too: every false ball on a tracked player
  (`false_near_player`, variants `degraded_n*`), models retrained on the same kind of input, seed 2027:

```text
detector                     false balls at random     false balls on players
fill 0.6, precision 0.5      +0.0185 (5/2)             -0.0344 (0/7)
fill 0.6, precision 0.7      +0.0352 (7/0)             -0.0033 (2/5)
fill 0.3, precision 0.9      +0.0318 (7/0)             +0.0072 (3/4)
fill 0.6, precision 0.95     not run                   +0.0426 (7/0)
fill 0.3, precision 0.97     not run                   +0.0295 (7/0)
```

  Reality lies between the two columns. **Precision is what decides**: below about 0.9 the gain can vanish, at 0.95 or more it holds even
  in the worst case. The threshold rule of `eval_ball_detector.py` moves from 0.9 to **0.95**, a choice made on IDSSE and on no BAR-ATH
  number. The generic COCO model, near 50% on the wide view, is ruled out. One degradation seed: the gaps of 0.002 in this table mean nothing.

## 28. Training a ball detector on external data: preparation (04/11/2026)

Not run yet. `scripts/prepare_ball_dataset.py`: COCO tiles of 576 pixels from SoccerNet-Tracking, frames scaled by a random factor
in 0.4-1 so that the ball is small as in a wide view, 25% of the tiles with no ball, sequences never mixed between train and valid.
`scripts/train_ball_detector.py` trains RF-DETR Medium; `--weights` in `eval_ball_detector.py` and `detect_ball_video.py` uses the result.
The licence of the SoccerNet data still has to be read before anything built on it enters the submission.

## 29. Does the tactical camera say when the ball is out? No (04/11/2026)

`scripts/check_camera.py`, on the PC, aggregate numbers only, no ground truth, nothing tuned from it. From the provided homography:

- The camera pans along x and barely along y (centre y 41-46 m), median speed 0.1 m/s, under 0.5 m/s on 62% of the samples, 58% of
  the pitch in view.
- Against the tracked ball: correlation in x 0.83, in y 0.07. The players' centroid gives 0.82 and 0.80 and is closer to the ball (median
  20 m against 24 m). The camera follows about 1.4 s late. It adds nothing to what the players already say.
- After a ball lost near a line or mid-pitch, its speed (0.19 / 0.15 m/s) and its distance from the line (23 / 27 m) hardly differ; tagged
  restarts (0.06-0.24 m/s) look like untagged play (0.12 m/s), and only 5 kick-offs and free kicks are few.

**The camera is dropped as a signal for the stoppages.** The homography stays useful where it is exact: pixels to pitch for detector
candidates, the expected size of a ball, and the filters of a future fusion.

A side result on where the tracker loses the ball, from the same run: of the ball gaps of at least 3 s, 62 start with the ball last seen
within 5 m of a line and 28 further than 15 m from all lines, a ratio of 2.2. On IDSSE: 1.41 with `degraded`, 1.82 with `degraded_sd5`, 2.51 with
`degraded_sd8`; a live ball spends 17% of its time near a line and 44% that far. With 90 gaps the interval is wide (about 1.4-3.5), so it barely
separates the scenarios. It leans towards a tracker that loses the ball on its way out. Not used to set anything: question 4 to the organisers.

Sources: Vidal-Codina et al., arXiv 2202.00804; Bassek et al., Scientific Data 12, 195 (2025); Cioppa et al.,
SoccerNet-Tracking, arXiv 2204.06918; Scott et al., SoccerTrack v2, arXiv 2508.01802; rfdetr on PyPI.
