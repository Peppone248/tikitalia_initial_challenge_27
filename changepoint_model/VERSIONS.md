# Versions and attempts (changepoint_model)

Every row: leave-one-match-out on external matches, official scorer, mean spellF1 over the held-out
matches. Decoder settings (gamma, w) chosen in-sample on the training matches. Degradation seed
2027 (match i gets seed 2027 + i). "Metrica" = the 3 sample games (so each model is trained on 2).

| version | date | what | Metrica degraded | Metrica clean | note |
|---|---|---|---|---|---|
| baseline50 | – | official baseline | 0.325 | 0.400 | |
| v1 | 13/11 | frame heads (hgb), boundary hazards, k=12, min 0.8 s / 2 s, gamma grid ≤ 4 | 0.730 | 0.803 | constant-hazard HMM with the same emissions: 0.715 / 0.798 (+0.015 / +0.005 for the boundary hazards, 3/0 both) |
| v2 | 14/11 | + stacking (trajectory of P(home) into the boundary heads), gamma grid ≤ 16, w grid ≥ 0.25 | **0.747** | **0.808** | gamma 8, w 0.25 chosen everywhere: the hazards carry the decision |
| v2 k=6 | 14/11 | boundary target ±6 frames | 0.743 | 0.808 | within noise |
| v2 k=25 | 14/11 | boundary target ±25 frames | 0.736 | 0.806 | slightly worse |
| v2 min 0.5 s | 14/11 | minimum possession 12 frames | 0.739 | 0.812 | mixed, kept 20 frames |
| v2 seed 2028 | 14/11 | same as v2, degradation seed 2028 | 0.748 | – | harsh scenario: 0.676 (baseline 0.248) |
| v3 | 14/11 | GRU heads (hidden 64, 2 layers, 20 epochs, 100 windows of 60 s per epoch) | 0.708 | 0.801 | team acc 0.941 (> trees), D F1 0.832 (< trees); undertrained with 2 matches |

## 18 external matches (14/11, afternoon): 3 Metrica + 7 IDSSE + 8 PFF, LOMO, seed 2027

IDSSE and Metrica degraded (a priori scenario), PFF as given (real broadcast tracker, never
degraded). Decoder settings chosen on 4 training matches; gamma 8, w 0.25 on every fold.

| source | baseline50 | v2 spellF1 | macroF1 | D F1 | team acc | possessions pred / true | transF1@25 |
|---|---|---|---|---|---|---|---|
| Metrica (3) | 0.325 | 0.733 | 0.863 | 0.846 | 0.940 | 305 / 378 | 0.759 |
| IDSSE (7) | 0.265 | 0.657 | 0.917 | 0.975 | 0.909 | 331 / 437 | 0.690 |
| PFF (8) | 0.365 | **0.583** | 0.888 | 0.935 | 0.923 | 264 / 290 | 0.454 |
| all (18) | 0.319 | 0.637 | 0.895 | 0.936 | 0.920 | 297 / 362 | 0.596 |

Per match: Metrica 0.716 / 0.712 / 0.772; IDSSE 0.619 0.623 0.674 0.660 0.680 0.662 0.682; PFF 0.591
0.620 0.635 0.609 0.472 0.545 0.567 0.620. Final model `models/cpm_v2_all18.pkl` trained on all 18
(in-sample 0.721, gamma 8, w 0.25).

## v3 and v4 on 6 held-out matches of the 18 (14/11, evening)

Held out: metrica_g1, metrica_g3, J03WMX, J03WPY, 3812, 3816 (two per source), each model trained
on the other 17. GRU: hidden 64, 2 layers, 40 epochs. Hybrid (v4): P(home) of the GRU, dead ball
and boundaries of the v2 trees, decoder fixed in advance at v2's settings (gamma 8, w 0.25) and
emission prior (0.3, 0.3, 0.4) for both rows (`scripts/eval_hybrid.py`).

| held out | v2 trees | v3 GRU alone | **v4 hybrid** |
|---|---|---|---|
| metrica_g1 | 0.709 | 0.686 | 0.711 |
| metrica_g3 | 0.751 | 0.734 | 0.755 |
| J03WMX | 0.617 | 0.607 | 0.629 |
| J03WPY | 0.676 | 0.650 | 0.676 |
| 3812 (PFF) | 0.597 | 0.594 | 0.637 |
| 3816 (PFF) | 0.473 | 0.495 | 0.509 |
| mean | 0.637 | 0.628 | **0.653** |

v4 − v2: +0.0156 (6/0); with w 0.5 instead of 0.25: +0.0116 (6/0). The GRU names the team better
(team accuracy higher on 5/6), the trees find the dead ball and the boundaries better. Largest gain
on the real broadcast tracker (PFF +0.040, +0.036). One training seed of the GRU.

Final: `models/cpm_v4_hybrid.pkl` (GRU trained on all 18, trees of cpm_v2_all18.pkl). Prediction
in numpy only; Metrica game 3: same SHA-256 on two runs and in a second environment without torch,
with scikit-learn 1.7.2 (e15b07f4…).

BAR-ATH structure, v2 (provided match, 14/11): D 41.0 % of the graded frames, 333 possessions
(3.1/min), mean 11.3 s, 128 dead spells, home share of the alive frames 58.2 %; SHA d78f2942….

Public window, one check after the choice (15/11): v4 spellF1 0.667 [0.48-0.82], macroF1 0.808,
12/17 possessions at IoU 0.5, D recall 0.51; v2 0.588 / 0.806. Details in REPORT.md §0.2.

v4 + precise rule layers (D-SLOW, R4, placed, wall, R6), 15/11: hard -0.0075 (0/6), soft -0.0066
(1/5) on the v4 folds; v2 on 18 folds -0.006 (3/15, 0/18). Rejected by the rule fixed in advance.
Details in REPORT.md §0.3 and results/rule_union.md.

v4.1 (15/11): v4 + D before each period's kick-off from the Smart Tagging kick-off tag (Law 8;
FAQ: 2nd half graded from 110,808, kick-off 113,742). No-op without tags (external matches).
Tracking-only kick-off detector dropped (3/36 external periods 2-9 min late). REPORT.md §0.4.

v4.1 BAR-ATH (provided match, 15/11): 2nd-half kick-off at 113,660 (tag start; FAQ 113,742), 2,278 frames
set D, 293 possessions (v4: 306), D 42.5 % (v4: 41.1 %), SHA dae859b5....
Stoppage held until the restart kick (15/11): v4 -0.0026 (1/5), v2 -0.0019 (3/14). Rejected. REPORT §0.5.
R7a whistle: option `--whistle` of predict_atd.py, off by default.

**v4.2 (15/11) = v4.1 + whistle R7a: THE SUBMISSION.** BAR-ATH SHA 8682095e..., 292 possessions, D 44.4 %.
Public window (one check after the choice): spellF1 0.691 [0.51-0.86], macroF1 0.880, D F1 0.804, transF1@25 0.600,
12/17 possessions found (v4 / v4.1: 0.667 / 0.808). REPORT §0.6.

Bugs found and fixed on the way: the GRU training kept a second, team-swapped copy of every match in memory and was killed (out of memory) on 17 matches; the swap is now made per window. v1's first decoder did not add the emission when a chain stayed
in its last sub-state (it cycled through re-entries; caught by the synthetic test, now in tests/).

## Diagnostics (v2, Metrica degraded, LOMO)

- true H↔A changes 926: found within 1 s 63 %, within 2 s 71 %; predicted changes 775, 17 % not
  within 2 s of a true one;
- by length of the next possession: < 1 s (15 % of changes) found 56 %, 1–2 s 60 %, 2–5 s 68 %,
  ≥ 5 s 81 %;
- possessions 1018 predicted / 1133 true; stops found within 2 s 86 %, restarts 89 %.

## determinism check (16/11)

- `predict_atd.py` pins OMP/OpenBLAS/MKL/numba to one thread before numpy is imported: sums happen in
  the same order on every machine. Output unchanged on the public Metrica game 3 (SHA-256 e15b07f4…,
  same as before the pin; Linux x86-64, numpy 2.5.3, scikit-learn 1.9.1, OpenBLAS).
- `scripts/check_determinism.py` + `determinism_reference.json`: four steps (Metrica g3, v4.1, whistle
  blasts, v4.2), two runs each, SHA-256 compared across machines. Labels unchanged: no model or rule
  was touched.
