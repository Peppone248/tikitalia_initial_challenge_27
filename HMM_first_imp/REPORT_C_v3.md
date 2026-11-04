# Approach C v3 — possession as CHALLENGE.md words it, goalkeeper, dead ball from Smart Tagging (30/10/2026)

```powershell
.\venv\Scripts\python.exe HMM_first_imp\scripts\make_submission.py --config-dir HMM_first_imp\config_c3   # -> outputs\run_c3\tikitalia.csv
```

Code: `hmm_v0/segment_c.py` (`st_dead`, `goalkeeper_windows`, `confirmed_gains`), `hmm_v0/run.py`
(option `st_set_piece_feature`), `hmm_v0/validation.py` (`simulate_smart_tagging`). Config:
`config_c3/` (copy of `config_c` + the changes below, each commented). Tests:
`tests/test_hmm_v0.py::TestSegmentV3` (25/25 OK). v2 stays in `config_c` unchanged.

Three requests: (A) check that the change of possession follows CHALLENGE.md and simplify;
(B) goalkeeper with the ball in his hands keeps possession through the clearance until the next
touch; (C) use Smart Tagging to find stoppages, before trying any interpolation of ball gaps.

## A. Change of possession vs CHALLENGE.md

| CHALLENGE.md says | C (v2 and v3) does |
|---|---|
| a change happens when an opponent gains **controlled** possession | a new team needs **λ = 0.5 s of net evidence** (its contacts, x strength, plus restarts) to be credited: a segment of evidence shorter than that cannot buy a change |
| proxy: a player keeps the ball for a few seconds | a contact counts for its length: 1 s of control = 1 s of evidence > λ |
| proxy: the team completes passes | T-INTERCEPT (an opponent touch confirmed by control or by the next touch of a team-mate) adds evidence at the start of the gain |
| loose / contested ball: possession stays with the last team | segments with no evidence cost nothing to keep and λ to change: the last team keeps them |
| not a change: touch without control, deflection | a touch of 0.2–0.4 s gives < λ: no change (test `test_short_touch_does_not_buy_a_possession`) |
| not a change: aerial duel without control | high balls are not tracked: no contact, no evidence, no change |
| a restart has a clear possessor | R-TOUCH / R-KO give the restarting team the segment |
| change dated at the gain | possession may change only at the start of a contact / restart / ball reappearing (the candidates): the change is dated at the opponent's first touch |

**The literal version was implemented and measured** (`possession.mode: control`): a team gains
possession only at the start of a *confirmed* contact (≥ T_CONTROL, or followed within T_PASS_MAX
by a team-mate's contact) or at a restart, everything else changes nothing. On IDSSE (7 matches,
same dead ball): **−0.052 spellF1 degraded, −0.036 clean, worse on 0/7 matches**, and lower team
accuracy too (0.737 vs 0.753). It misses the changes where the confirmation is not visible (short
contacts, ball unseen). With T_CONTROL 0.5 s it is −0.028 / −0.003. Kept as an option, not the
default: the λ form is the same rule written softly ("needs λ seconds of control"), and it measures
better.

**Simplified** (v3): four of the nine evidence weights set to 0 — P-FREE, P-PASS, T-REV, P-SHAPE —
because the contacts already carry them. IDSSE: 0.000 degraded, +0.006 clean. What remains: contacts,
T-INTERCEPT, R-TOUCH, R-KO, Smart Tagging phase (only for ball-unseen stretches), nearest player.

## B. Goalkeeper

A contact of a team's **deepest player** (the goalkeeper; the tracking has no identities) **inside
his own penalty area**, held **≥ 1 s** (hands; a shorter touch is at his feet), opens a window:

- contacts of opponents pressing him during the hold are ignored;
- the window runs **until the first contact after he has released the ball**, so the clearance in
  flight belongs to his team (CHALLENGE.md: a loose ball stays with the last team);
- inside the window the evidence is his team's (2 x a contact per frame) and no change can happen.

Check on IDSSE (DFL possession, 4 matches): per window, **86–98% of the frames are the goalkeeper's
team** in the DFL labels; 15–47 windows per match, median 4–7 s. Cost on IDSSE: −0.004 spellF1
(degraded), −0.003 (clean): the windows end at slightly different moments than DFL's
possession changes. Kept, because it is the rule asked for and the hold-and-clearance case is
exactly CHALLENGE.md's loose ball. BAR-ATH: **22 windows**.

## C. Dead ball from Smart Tagging

### What the tagging contains (BAR-ATH)

| SET PIECES type | tags | IDSSE mean per match | share tagged |
|---|---|---|---|
| throw-in | 26 | 42.7 | ~61% |
| corner kick | 12 | 7.4 | ~100% |
| goal kick | 10 | 15.3 | ~65% |
| free kick | 5 | 31.6 | ~16% |
| kick-off | 6 (2 + 4 after goal) | 2.4 | 100% |

**There is no tag for fouls or offsides** and only 5 free kicks: the stoppages for fouls/offsides
(~30 per match) are almost never tagged. They remain the job of D-WHISTLE / D-HOLD (players
slowing down).

**Timing** (BAR-ATH tracking only): a set piece tag starts a few seconds **before** the restart:
first fast ball after the tag start +3…+10 s for corners, mostly within +10 s for throw-ins (the
thrower is already off the pitch about 3 s before the tag start). The tag lasts 35 s and covers
the play after the restart.

### The rule (segment_c.st_dead, `dead.smart_tagging` in segment.yaml)

For every throw-in, corner, goal kick and free kick tag:

- **end of the D** = first sign of play after the tag start, within 15 s: the ball seen faster than
  8 m/s, or the start of a contact; nothing seen → tag start + 5 s;
- **start of the D** = the last action seen before the tag (end of a contact or ball seen fast),
  at most 20 s earlier; nothing seen → tag start − 3 s.

It works when the ball is not seen, which is where the D was missing. The span is joined to the D of
the HMM, then cleaned as before.

### The old rule D-SP had a bug (also in v1/v2)

D-SP marked D from a still ball to the next restart kick *inside the tag*; with no restart kick
detected it marked D **to the end of the 35 s tag**, into live play. On IDSSE with simulated tagging
it dropped the D precision from 0.94 to 0.81 on one match (J03WPY: spellF1 0.332 → 0.281). **v3
turns it off** (replaced by the rule above) and also drops the NN input "inside a set piece tag"
(the 35 s mask is mostly live play: +0.004 on the 2 matches checked). On BAR-ATH v1 and v2 ran with
D-SP active.

### How it was validated without BAR-ATH labels

IDSSE has no Smart Tagging, so a **tagging-like feed is simulated** from the true restarts of the
other games (`simulate_smart_tagging`, settings in `config/validation.yaml`): each restart
(BallStatus 0 → 1) gets its type from where the ball is, is kept with the BAR-ATH share for that
type, and its tag starts a lead resampled from the BAR-ATH leads before it; kick-offs and goals
are tagged like on BAR-ATH. About 50 tags per match, as on BAR-ATH. Like the degraded tracking,
this uses only other games' labels and BAR-ATH *tracking/tagging* statistics.

## Results on IDSSE (7 matches, simulated tagging, official scorer, pooled)

`scripts/run_validation.py --st-sim --versions baseline50 c_v2 c_v3 --variants degraded clean
--out-dir outputs/validation_v3` → `outputs/validation_v3/results.md`.

| tracking · labels | version | spellF1 [90% CI] | macroF1 | D F1 | team acc | possessions (true) |
|---|---|---|---|---|---|---|
| degraded · controlled | baseline50 | 0.210 [0.199–0.221] | 0.462 | 0.389 | 0.551 | 2782 (2705) |
| degraded · controlled | c_v2 | 0.303 [0.287–0.319] | 0.716 | 0.798 | 0.720 | 1243 |
| degraded · controlled | **c_v3** | **0.319** [0.302–0.336] | **0.740** | **0.835** | **0.733** | 1264 |
| clean · controlled | baseline50 | 0.326 | 0.397 | 0.000 | 0.838 | 2644 |
| clean · controlled | c_v2 | 0.479 | 0.880 | 0.960 | 0.834 | 1551 |
| clean · controlled | **c_v3** | **0.479** | 0.876 | 0.957 | 0.827 | 1573 |

Paired, c_v3 − c_v2: **+0.016 spellF1 degraded (6/7 matches), D recall +0.042**; clean −0.000
(3/7). c_v3 − baseline50: +0.109 degraded, +0.153 clean (7/7). With the raw DFL labels: +0.013
degraded (6/7).

So v3 helps where the tracking is dirty and costs nothing where it is clean (the principle agreed:
no system that is good only on dirty data). Note that c_v2 scores 0.303 here and 0.316 without
tagging: that difference is the D-SP bug.

## BAR-ATH

```text
CSV        outputs/run_c3/tikitalia.csv   SHA-256 6a7baedd12453aef954c6710800ce93fd1eb33efc6e9f4be6aefd037b30e01f0
graded     H 43.5% · A 23.7% · D 32.8%          (v2: 45.6 / 27.9 / 26.5)
dead       124 stoppages, mean 16.8 s, median 11.6 s, p90 35.7 s   (v2: 122, 13.8 / 8.2 / 25.5)
possession 228 possessions, median 13.3 s          (v2: 244)
goalkeeper 22 windows
```

Public 5 minutes, **structure check only** (nothing was chosen on it; noise ±0.36):
spellF1 0.512, macroF1 0.830, D recall 76% (v2: 66%), D precision 87%, 15 possessions (true 17).
Full side-by-side in `EVAL_TRACKING.md`.

## Differences v2 → v3

| | v2 (`config_c`) | v3 (`config_c3`) |
|---|---|---|
| dead ball | HMM D (v1) + D-SP | HMM D (v1) **∪ Smart Tagging anchor**; D-SP off; NN without the set-piece mask |
| D when the ball is unseen | only what the HMM infers | the tagged restarts give D from the last action to the restart |
| team evidence | 9 weights | 5 (P-FREE, P-PASS, T-REV, P-SHAPE at 0) |
| goalkeeper | nothing special | window from his hold to the next touch after the clearance |
| possession rule | λ = 0.5 s of evidence | same (literal state machine measured: −0.05, kept as option) |

## Limits and next steps

1. **Fouls and offsides are not in the tagging**: about 30 stoppages per match still depend on the
   players slowing down (D-WHISTLE / D-HOLD). This is the next gap for the dead ball.
2. The **ball-gap reading** (how the ball disappears and reappears, players beyond the lines) was
   not done yet, as agreed: it is the next step, now on top of the tagging anchor.
3. The tagging simulation copies BAR-ATH's recall and lead, but not its false positives (unknown
   without labels).
4. Possessions are still too few on IDSSE (1264 vs 2705; DFL is finer than the challenge).
