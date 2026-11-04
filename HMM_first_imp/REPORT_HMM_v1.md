# HMM_first_imp — report v1 (30/10/2026)

v1 of the pipeline: steps 1, 2, 3, 5 and 9 of the improvement plan, aimed at **fragmented
possessions** and **too little dead ball**, plus **cross-machine reproducibility**. Everything of
v0 still holds (see [REPORT_HMM_v0.md](REPORT_HMM_v0.md)); this file only describes the changes.

```powershell
.\venv\Scripts\python.exe HMM_first_imp\scripts\make_submission.py                                  # v1
.\venv\Scripts\python.exe HMM_first_imp\scripts\make_submission.py --config-dir HMM_first_imp\config_v0   # v0 logic, for comparison
```

~80 s, ~1.2 GB RAM. Tests: 19/19 (`python -m unittest discover -s HMM_first_imp/tests`); team tests
39/39, team code untouched. Nothing committed.

---

## 1. What changed

| step | change | where |
|---|---|---|
| **1. Only rules change the team** | Time-varying H↔A transitions. A change *to* a team keeps its base rate only in [−0.5 s, +1 s] around the start of a rule episode of that team (P-DRIB, P-FREE, P-PASS, T-INTERCEPT, T-REV, R-TOUCH, R-KO). Elsewhere its probability is divided by 20 (≈ 3 nats). Same mechanism as the restart boost for D (`hmm.py`, 8 precomputed transition regimes) | `hmm.yaml → hmm.team_change` |
| **2. The NN does not pick the team** | `omega_nn`: the NN's H and A classes count the same for both teams. The NN says alive / dead / unseen; the team comes from the rules | `hmm.yaml → emission.omega_nn` |
| **3. TURNOVER rules** | **T-INTERCEPT** (+ the T-LOOSE case): the first contact of the other team after the last *confirmed* holder, confirmed by a control ≥ 1 s or by the next contact of the same team within `T_PASS_MAX` (a pass). A touch lost again is not a change. **T-REV**: reversal ≥ 120° of the ball direction with an opponent on the ball, confirmed by his team's contact | `rules/turnover.py` |
| **5. D-WHISTLE as an anchor** | Onset of the stop = all players ≤ 0.5 of their recent tempo for ≥ 1 s, both teams slowed (≤ 0.7). Confirmed by the ball held still or by nobody touching it within 3 s. Kind ANCHOR (q 0.7); D-HOLD now also follows D-WHISTLE and keeps the D until the restart | `rules/dead_extra.py`, `rules_hmm.yaml` |
| **9. Same file on every machine** | (a) one BLAS thread; (b) NN = **random-feature network** (1 hidden layer of 192 fixed random ReLU units + softmax output) trained by full-batch Nesterov gradient descent on a **convex** problem; (c) inputs on a 2^-16 grid; (d) probabilities rounded to 3 decimals; (e) emissions and log-transitions on a 2^-20 grid, so every Viterbi sum is exact | `nn.py`, `exact.py`, `hmm.yaml → nn.model: rf` |

### Why the NN changed (step 9)

Your PC and the cloud machine produced v0 files differing on 1.3% of the frames. Tests here:

| perturbation of the NN inputs (simulating another CPU / BLAS) | MLP (v0, minibatch Adam) | MLP, exact integer arithmetic | **RF net (v1)** |
|---|---|---|---|
| 1 vs 2 BLAS threads | 0 frames | – | 0 |
| ~1,000 values off by one float32 ulp | – | 0 | 0 |
| ~10 values moved by one grid step (2^-16) | – | **1,896 frames** | **0** |
| ~1,000 values moved by one grid step | – | – | **0** |

Exact integer arithmetic (`exact.py`) removes the summation-order effect, but minibatch SGD on a
non-convex network is **chaotic**: any input that differs between machines (for example a float32
`hypot` in the team features on Windows vs Linux) diverges the whole training. A convex problem
solved by full-batch gradient descent cannot amplify a difference, so a tiny input change stays
tiny and disappears in the rounding. The MLP is still there (`--set hmm.nn.model=mlp`).

Cost: the RF net reproduces its pseudo-labels on 85.5% of the training frames (MLP: 98%). Its
role is now limited to alive / dead / unseen, since the team comes from the rules.

## 2. Effect on the output (BAR-ATH, no ground truth)

| | v0 logic (`config_v0`) | **v1** |
|---|---|---|
| D share of graded frames | 21.3% | **26.5%** |
| dead spells / median length | 145 / 5.6 s | 124 / **7.8 s** |
| possessions / median length | 329 / 8.9 s | **205 / 17.0 s** |
| possessions shorter than 3 s | 92 | **19** |
| short (< 5 s) possessions with no rule behind them | 87 | **9** |
| H→A→H flips with a middle spell < 5 s | 72 | **11** |
| direct H↔A changes | 184 | 81 |
| same-team possessions split by a D < 2 s | 0 | 0 |
| D-WHISTLE agreement with the output | 8% | **90%** |

Agreement of the output with the anchors: D-OUT 99.8%, D-KO 100%, P-PASS 99.7%, P-DRIB 92.6%,
T-INTERCEPT 84.4% (103 confirmed turnovers), D-HOLD 75.2%.

## 3. Public 5 minutes — pipeline check, NOT selection

| file | macroF1 | spellF1 | possessions (truth 17) | mean length (truth 14.2 s) |
|---|---|---|---|---|
| v0 logic (regenerated) | 0.773 | 0.432 | 21 | 12.2 s |
| baseline_smooth125 | 0.581 | 0.424 | 16 | 18.5 s |
| **v1** | 0.741 | 0.345 | **12** | **21.2 s** |
| baseline_smooth50 (official floor) | 0.621 | 0.326 | 26 | 10.8 s |

The noise band on this window is ±0.36 spellF1 (bootstrap, 17 possessions), so the ranking means
nothing. **The structure does**, and it is consistent with the full-match diagnostics: v1 went
from too many possessions (v0) to probably **too few**. Some real changes of possession are now
blocked by the team gate. These are mostly changes where no rule sees a confirmed contact: the
ball is often unseen, and only 3 T-REV turnovers are confirmed. No parameter was changed after
this check.

## 4. What to do next (in this order)

1. **Rebalance the team gate** with an a-priori argument, not on the public window: the knobs
   are `hmm.team_change.closed_factor` (20 → e.g. 5–10), `window_post_s`, and `sources`. Better
   still, make more real changes visible: T-REV is almost never confirmed (3), and T-TEAMFLIP
   (both teams reverse direction) would open the gate during occlusions.
2. **Validation set on other games (B3)** — still the only way to pick between v0-like
   fragmentation and v1-like merging. Both are now one config switch apart.
3. Restart detection with the ball unseen (R-THROW, ball reappearing after a D): it decides
   where the D ends.
4. If the team decides to keep the MLP for quality, the remaining risk is reproducibility; the
   alternative is to ship trained weights with the repo — to be discussed against RULES 3.

## 5. Files added / changed

```text
config/            v1 defaults (team_change, omega_nn, rf net, D-WHISTLE anchor, T-* rules)
config_v0/         frozen v0 settings (--config-dir HMM_first_imp/config_v0)
hmm_v0/exact.py    integer matmul, polynomial exp, grids (machine-independent arithmetic)
hmm_v0/nn.py       RandomFeatureNet (default) + MLP with exact arithmetic
hmm_v0/hmm.py      8 transition regimes: restart x open_H x open_A; quantised log-transitions
hmm_v0/engine.py   team gates from rule episodes; quantised emissions
hmm_v0/votes.py    change_points()
hmm_v0/rules/turnover.py      T-INTERCEPT, T-REV
hmm_v0/rules/dead_extra.py    D-WHISTLE v1 (anchor, onset + confirmation), WhistleV0 kept
hmm_v0/diagnostics.py         fragmentation(): short / unsupported possessions, H-A-H flips
scripts/make_submission.py    one BLAS thread
tests/test_hmm_v0.py          +5 tests (team gate, turnover confirmation, RF-net stability, exact matmul)
```
