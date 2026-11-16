# Rule-layer union on external held-out matches

## Layer precision (graded frames; 'added' = frames the model called alive)

| layer | frames | truly dead | added | added and truly dead |
|---|---|---|---|---|
| slow | 455325 | 92.3% | 10944 | 24.3% |
| back | 27397 | 88.4% | 2120 | 42.2% |
| placed | 53045 | 94.1% | 2759 | 13.7% |
| wall | 195599 | 95.9% | 3824 | 52.4% |

## v2 (18 held-out matches)

| variant | spellF1 | macroF1 | D F1 | D rec | D prec | paired vs v2 | better / worse | PFF better / worse |
|---|---|---|---|---|---|---|---|---|
| v2 | 0.6401 | 0.8956 | 0.936 | 0.958 | 0.919 | +0.0000 | 0 / 0 | 0 / 0 |
| v2_hard | 0.6338 | 0.8928 | 0.932 | 0.964 | 0.908 | -0.0063 | 3 / 15 | 1 / 7 |
| v2_soft | 0.6343 | 0.8928 | 0.932 | 0.963 | 0.908 | -0.0058 | 0 / 18 | 0 / 8 |
- Metrica: v2 0.724, hard -0.0081, soft -0.0072
- IDSSE: v2 0.661, hard -0.0082, soft -0.0074
- PFF: v2 0.590, hard -0.0039, soft -0.0040

## v4 (6 held-out matches)

| variant | spellF1 | macroF1 | D F1 | D rec | D prec | paired vs v4 | better / worse | PFF better / worse |
|---|---|---|---|---|---|---|---|---|
| v4 | 0.6528 | 0.8934 | 0.925 | 0.959 | 0.900 | +0.0000 | 0 / 0 | 0 / 0 |
| v4_hard | 0.6453 | 0.8894 | 0.919 | 0.964 | 0.885 | -0.0075 | 0 / 6 | 0 / 2 |
| v4_soft | 0.6462 | 0.8894 | 0.919 | 0.964 | 0.885 | -0.0066 | 1 / 5 | 0 / 2 |
- Metrica: v4 0.733, hard -0.0114, soft -0.0106
- IDSSE: v4 0.653, hard -0.0094, soft -0.0071
- PFF: v4 0.573, hard -0.0018, soft -0.0022
