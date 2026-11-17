# How the possession model works (v4.2)

A guide for the team: what the model does, which pieces it is made of, how a frame becomes a
label and why it was built this way. The commands, data and validation numbers are in `README.md`.

---

## 1. The job in one sentence

For each of the 199,319 video frames of the match, write who has the ball, **H** (home), **A**
(away) or **D** (dead ball), or **X** outside the two halves. Tracking is the only input, plus two
signals from the provided files: the kick-off tag in Smart Tagging and the referee's whistle in
the audio.

The scorer (spellF1) does not count frames. It counts **possessions** (spells): a predicted
spell matches a true one when the two overlap enough (IoU 0.3 to 0.7). Two consequences shape
the whole design:

- **A missed change costs two spells.** If the model misses a change of possession, the long spell
  it writes matches neither of the two true spells it covers.
- **A wrong change also costs.** A cut in the wrong place breaks a good spell into two bad ones.

A frame classifier followed by a smoother is good at saying *who* has the ball, but it rarely
cuts the short possessions, and that is where most of the points are lost.
This model is built to find **the changes**, not only the state.

---

## 2. The pipeline at a glance

```text
 provided files                    per frame (25 per second)                    whole match
 ──────────────                    ─────────────────────────                    ───────────

 tracking (ATD) ─► [1] features ─┬─► [2a] GRU  ──────────► P(home | alive)  ─┐
                    134 numbers  │                                          │
                    per frame    ├─► [2b] trees ─────────► P(dead)          ├─► [3] HMM decoder ─► H/A/D
                                 │                                          │     (Viterbi)
                                 └─► [2c] trees ─────────► P(change)  ───────┤
                                     (+ how P(home) moves) P(stop)          │  hazards: how cheap
                                                           P(restart) ──────┘  a cut is at frame t
                                                                                   │
 Smart Tagging "Kick Off" ───────────────────────────────────────► [4] Law 8 ◄────┘
                                                                     │  D before each kick-off
 audio ─► whistle blasts ────────────────────────────────────────► [5] whistle ◄──┘
                                                                     │  D from the blast if the
                                                                     ▼  players slow down
                                                              tikitalia.csv
```

Steps 1 to 3 are the **learned model**: trained on 18 external matches, never on the provided
one. Steps 4 and 5 are **rules** taken from the Laws of the Game. Their constants are set a
priori (metres, seconds, m/s).

---

## 3. Layer by layer

### [0] Input

- Positions in metres, origin at the centre spot. A player or ball that is not tracked is left
  empty (NaN) rather than guessed.
- Each team's attack direction is **inferred** from where its deepest players stand in each half.
  The same code therefore runs on any source (IDSSE, Metrica, PFF, the provided ATD).
- The two halves come from `games.half_video_blocks()`: those are the graded frames, and
  everything else is X.

### [1] Features: 134 numbers per frame, split into two families

The key trick is a symmetry. If you swap home and away (and their attack directions), the answer
"who has the ball" must flip, but "is the ball dead" must not change.

| family | count | behaviour under the swap | examples | used for |
|---|---|---|---|---|
| **odd** | 66 | the sign flips | control vote (closest player within 1.5 m of a ball slower than 8 m/s: +1 home, −1 away) now and averaged over 0.5 / 1 / 2 / 4 s, before and after the frame; pressure at 1 / 3 / 8 m; contacts (ball turns > 30° or changes speed by > 2 m/s); ball position in the attack frame; shape differences (width, length, lines, centroid, speed) | **who** has the ball |
| **even** | 68 | unchanged | ball seen / seconds since or until it was seen, ball speed, ball out of play; how many players are tracked, how fast they move, share walking; duels; **flip features**: did the control vote change sign across this frame? | **dead ball** and **changes** |

Because of the symmetry the model cannot learn "home has the ball more often", which would be a
bias of the training matches. Every constant here is a football quantity (radii, speeds, seconds)
chosen before seeing any data.

The model is run on the whole recorded match, so a feature at frame t may look **into the future**
too ("who controls the ball in the next 2 s"). That is legitimate, because the scorer also sees
the whole file, and it is what makes a change easy to place: before it one team controls the
ball, after it the other.

### [2] The heads: six probabilities per frame

| head | model | question it answers |
|---|---|---|
| P(home \| alive) | **bidirectional GRU** (recurrent network, 64 units, 2 layers) | if the ball is alive, how likely is it home's? |
| P(dead) | **gradient-boosted trees** (scikit-learn, 200 trees) | is the ball out of play here? |
| P(change) | boosted trees | is a home↔away change within ±0.5 s of this frame? |
| P(stop) | boosted trees | does play stop within ±0.5 s? |
| P(restart) | boosted trees | does play restart within ±0.5 s? |

Why two kinds of model (the "hybrid", v4):

- **The GRU reads a whole stretch of the match**, forwards and backwards, rather than a fixed set
  of windows. It names the team better: 94 % team accuracy against 93 % for the trees.
- **The trees date the boundaries better.** They receive the explicit windows plus how P(home)
  moves around the frame ("stacking"): a change is where P(home) swings from one side to the other.
- Each part does what it is best at. On 6 held-out matches the hybrid scored +0.016 spellF1 over
  the trees alone, and won on 6 of the 6.

The GRU is symmetric by construction: it is run on the frame and on its home/away mirror, and the
two answers are averaged. It needs torch only to train; prediction runs in plain numpy, so the
output does not depend on torch being installed.

### [3] The decoder: turning probabilities into possessions

A frame-by-frame argmax would flicker H-A-H-A. The decoder chooses the **single most likely
sequence** of labels for the whole half (Viterbi algorithm over a hidden Markov model with three
states, H, A and D). It balances two things:

1. **Evidence (emission).** At each frame, how strongly the heads favour H, A or D.
2. **The price of a cut (hazard).** Every time the label changes the sequence pays a cost, and
   that cost is **different at every frame**: it is cheap where the boundary heads expect a
   change and expensive everywhere else.

That second point is what differs from a standard HMM, which has one fixed switch
rate for the whole match. Raising a fixed rate adds cuts everywhere, the wrong ones included
Here the cut becomes cheap only where the evidence for a boundary is.

The same hazard also sets the price of **not** cutting. Staying in the current state costs
−ln(1 − h_change − h_stop) per frame: about 0.005 where nothing happens, but about 0.25 per frame
where P(change) = 0.7. Where a head sees a boundary, the decoder is therefore pushed to cut, and the
evidence then picks which state comes next.

Hard constraints:

- A possession lasts at least **20 frames (0.8 s)**.
- A dead ball lasts at least **50 frames (2 s)**.
- H→A and A→H go through the *change* hazard, H/A→D through *stop*, D→H/A through *restart*.

### [4] Law 8: nothing happens before the kick-off

In the provided match the second half is graded from frame 110,808, but the kick-off is only at
113,742. In between, nearly 3,000 frames show players warming up with balls on the pitch. A
tracking model sees "a player next to a ball" and invents possessions there.

The rule: inside the Smart Tagging "Kick Off (Start)" window, find the kick. That is the first
frame where the ball, at rest near the centre spot, moves more than 2 m within 1 s. Everything
before it is D. If no kick is found, the start of the tag window is used, which is the cautious
choice (fewer frames turned into D).

On the external matches the periods start at the kick-off, so the rule changes nothing there.
A version that guessed the kick-off from the tracking alone was dropped: it placed the kick-off
2 to 9 minutes late in 3 of the 36 external halves.

### [5] The whistle (Law 5)

The referee's blasts are detected in the audio with a frequency profile taken from the literature
(the band published for referee whistles in broadcast audio). A blast while the model says "alive" starts a stoppage **only
if the players then really slow down**: within 3 s, their mean speed drops to 70 % of what it was
before, or below walking pace (1.1 m/s).

The stoppage runs from the blast to the model's next stoppage if one starts within 15 s, otherwise
to the next kick from rest within 30 s. This mirrors Law 5: play stops at the whistle, not when the
tracking finally notices it. Blasts during a stoppage (the restart whistle) are ignored.

---

## 4. How a decision is made, frame by frame (with real numbers)

The decoder adds up, along each candidate sequence, **evidence minus the cost of cuts**, and keeps
the best one. With the settings of the model (gamma 8, w 0.25):

**Evidence per frame.** For a frame with P(dead) = 0.05 and P(home | alive) = 0.9:

```text
evidence for H over A = 0.25 × ln(0.9 / 0.1) ≈ 0.55 per frame
with P(home) = 0.7 instead:   0.25 × ln(0.7 / 0.3) ≈ 0.21 per frame
```

The weight w = 0.25 deliberately makes single frames weak. **No single frame decides**, only a
consistent stretch does.

**Cost of a cut** at frame t = −ln(hazard), where hazard = 8 × P(change) / 25:

| P(change) at that frame | hazard | cost of cutting there | frames of clear evidence (0.55 each) needed to pay for it |
|---|---|---|---|
| 0.001 (nothing happening) | 0.0003 | 8.0 | ~15 (0.6 s) |
| 0.02 | 0.006 | 5.1 | ~9 |
| 0.2 | 0.064 | 2.8 | ~5 |
| 0.8 (a change is very likely) | 0.26 | 1.4 | ~3 |

**Example: a one-second touch (Metrica sample game 3, frames 56,500–56,526).** Away has the ball,
a home player touches it for 26 frames, and away gets it back.

- **What the heads see.** P(home) rises only to about 0.6, but P(change) peaks above 0.8 at both
  ends of the touch.
- **The cuts.** Each cut costs about 1.2 instead of 8, and staying with away through the touch
  would cost about 4.4 over the 26 frames.
- **The result.** v4 writes A | H (1.0 s) | A, exactly as the truth.

The same probabilities decoded with a fixed switch cost (6.2 per cut, one change every 20 s) give a
single away spell of 18.4 s. That spell overlaps the first true away spell with IoU 0.56 and the
second with 0.36, so the second away spell and the home touch are lost at every threshold. On the
whole game the fixed-cost decoder predicts 292 possessions against 399 true ones. The game is also
one of the training matches, so this shows the mechanism, not the score.

**Example: a pass that only looks like a change.** A long ball travels 2 s with nobody in
control. The votes go quiet, but they do not flip, because the same team receives the ball.
P(change) stays low, the cut stays expensive, and H is kept. This is the "a touch is not a
change" rule that the labels are built on.

**Example: a stoppage.** Ball out for a throw-in:

- **Signals.** The ball leaves the lines, the players slow down and the ball is often lost by the
  tracker.
- **Heads.** P(dead) rises above 0.9 and P(stop) peaks at the exit.
- **Evidence.** For D over H that is about 0.65 per frame, which buys a D spell of at least 2 s.
- **End.** The spell ends where P(restart) peaks: the ball moves again and the players speed up.

---

## 5. How it was trained

| | |
|---|---|
| **Data** | 18 external matches with event labels: 3 Metrica sample games, 7 IDSSE (DFL), 8 PFF World Cup games. The provided match is never used to train or choose anything. |
| **Labels** | none written by hand by the team. Metrica: from the published events, by one rule taken from CHALLENGE.md (dead from the ball leaving play, the whistle or the goal until the restart kick; team = the team of the last on-ball event; a lost ball stays with the loser until the opponent's recovery or pass). IDSSE: DFL's own ball-status and ball-possession flags in the position data. PFF: PFF's game events attached to the tracking frames (`data_prep/fetch_pff.py`). |
| **Fitting** | trees: one frame in 4, 200 rounds, 31 leaves, at least 200 frames per leaf, seed 2027. GRU: windows of 60 s, swap augmentation, 40 epochs, fixed seed. Decoder: gamma in {4, 8, 16} and w in {0.25, 0.5} compared on 6 of the training matches, gamma 8 and w 0.25 chosen (the earlier Metrica-only runs swept 0.5–16 and 0.25–2 and chose the same point every time). |
| **Evaluation** | leave one match out: train on 17, score the 18th with the official scorer, repeat. |
| **Public window** | the 5 labelled minutes of the provided match are a check only: nothing is trained or chosen on them. |

---

## 6. How good it is

| | spellF1 | note |
|---|---|---|
| Official baseline (smooth 50) on the public window | 0.326 | |
| **v4.2 on the public window** | **0.691** | 2.1× the baseline; noise on 17 possessions is about ±0.17 |
| v2 (trees only), leave-one-out, PFF (real tracker) | 0.583 | constant-rate HMM on the same files: 0.501 |
| v2, leave-one-out, Metrica / IDSSE | 0.733 / 0.657 | |
| v4 hybrid vs v2, 6 held-out matches | 0.653 vs 0.637 | wins 6/6 |

---

## 7. Strengths and weaknesses

**Strong at**

- **Cutting changes.** Short possessions get split from long ones when the boundary heads see a
  swing, which is where a fixed-rate decoder loses the most.
- **Robustness.** It is trained on three different tracking sources, including a real broadcast
  tracker, and every constant is a football quantity. Nothing is tuned to the provided match.
- **Symmetry.** No home or away bias is possible, by construction.
- **Reproducibility.** It is deterministic: the same file, byte for byte, on every run (see
  `scripts/check_determinism.py`).

**Weak at**

- **Very short possessions.** Spells under 2 s are still found only about half the time: the
  0.8 s minimum and the cost of two cuts hold them back.
- **Stoppages the tracker hides.** When the camera cuts away, the ball and players vanish. The
  model has to guess from the seconds before and after, so some stoppages start late and end
  early. The whistle fixes part of this, not all.
- **Changes that look like passes.** Some interceptions, and loose balls between opponents, move
  P(home) only a little, so the cut stays expensive.
- **The 0.8 s minimum.** It rules out flicker, and also the very shortest true possessions.

---

## 8. Where things are

| file | what |
|---|---|
| `cpm/features.py` | the 134 features |
| `cpm/models.py`, `cpm/seq.py` | the trees and the GRU |
| `cpm/pipeline.py` | prepare → predict → decode |
| `cpm/decode.py` | the HMM with time-varying hazards (Viterbi) |
| `cpm/kickoff.py` | Law 8 |
| `cpm/whistle.py` | whistle detector and whistle rule |
| `predict.py` | the command that writes the submission |
| `scripts/check_determinism.py` | same output on every machine? |
| `models/cpm_v4_hybrid.pkl` | the trained model (GRU + trees) |
