# Results

*State at 11 November 2026. This page collects the numbers in one place; method, parameters and
full tables are in `docs/`.*

**In short**: on the two graded halves of `BAR-ATH_tactical.mp4` the default ensemble
(`tikitalia_model` + `roboflow`) gives the ball in **75.8%** of the frames, 51.0% confirmed by
both models. In the ATD the frames with a ball go from 56.9% to **73.3%**.

There is no ground truth for the ball on this game: every number measured on the video is a
measure of coverage and consistency, not of accuracy. Accuracy is measured only on SoccerNet
game 11, held out from training (section 1).

## What was done

The goal is to know where the ball is in every frame of the tactical video.

1. **Data.** We took SoccerNet-Tracking 2023 (6 games, none of them the challenge game) and kept
   only the ball labels: 5 games for training, 1 for validation. We made two versions of the
   dataset, full frames and 640×640 tiles.
2. **Training.** We trained two YOLOv8 detectors with an extra head for small objects:
   `tikitalia_model` on full frames and `tikitalia_tiles` on tiles. A third detector, `roboflow`,
   is used as published, without training it.
3. **Detection.** Each detector ran on the whole video and saved up to 3 ball candidates per
   frame.
4. **Clean track per model.** For each detector we discard candidates outside the pitch, choose
   the sequence of candidates that moves like a real ball (no jumps above 40 m/s) and fill the
   short gaps.
5. **Ensemble.** We keep the ball where two detectors agree within 15 px, plus the parts of a
   track that pass through an agreed frame. The result is one position per frame, with a label
   that says how reliable it is.
6. **ATD.** We wrote a copy of the ATD in which the ball is the ensemble's, keeping the original
   one where the ensemble has none.
7. **Comparison.** We compared the three detectors and all their combinations, to see which
   ensemble covers more of the video.

The numbers of each step are below.

## 1. Detector training (validation on SoccerNet game 11)

| | `tikitalia_model` | `tikitalia_tiles` |
|---|---|---|
| Architecture | YOLOv8s-P2 | YOLOv8m-P2 |
| Input | Full frame at 1280 px | 640×640 tiles |
| Weights | `best.pt`, epoch 25 of 35 | `last.pt`, epoch 12 |
| Precision | 0.574 | 0.569 |
| Recall | 0.442 | 0.467 |
| mAP50 | 0.426 | 0.436 |
| mAP50-95 | 0.146 | 0.144 |

The two columns are **not comparable**: one is validated on full frames, the other on tiles. The
two models are compared on the video (section 3).

**Accuracy on the ball centre** (`train/eval_center.py`): a detection is correct if its centre is
within N px of an annotated ball.

| Model | Confidence | N | Precision | Recall | F1 |
|---|---|---|---|---|---|
| `tikitalia_model` | 0.25 | 5 px | 0.780 | 0.464 | 0.582 |
| `tikitalia_model` | 0.25 | 10 px | 0.913 | 0.543 | 0.681 |
| `tikitalia_model` | 0.20 (maximum F1) | 10 px | 0.852 | 0.610 | 0.711 |
| `tikitalia_tiles` | 0.25 | 5 px | 0.798 | 0.512 | 0.623 |
| `tikitalia_tiles` | 0.25 | 10 px | 0.907 | 0.582 | 0.709 |
| `tikitalia_tiles` | 0.20 (maximum F1) | 10 px | 0.859 | 0.662 | 0.748 |

`tikitalia_model` is evaluated on the 1,985 full frames (1,970 annotated balls), `tikitalia_tiles`
on the 5,066 tiles cut from the same frames (3,884 annotated balls: a ball in the overlap between
two tiles counts twice, and tiles where it is less than 50% visible are discarded). The two sets
of rows are therefore close but not strictly comparable.

Details: `docs/training.md`.

## 2. Default ensemble on the graded halves

`tikitalia_model` + `roboflow`, 158,857 frames (1st half 11,185-90,651, 2nd half 110,808-190,197).

| | Confirmed | Tracked | Recovered | Synthetic | **Ball present** |
|---|---|---|---|---|---|
| 1st half | 50.5% | 16.0% | 0.4% | 7.7% | **74.5%** |
| 2nd half | 51.6% | 16.4% | 0.2% | 8.9% | **77.1%** |
| **Total** | **51.0%** | **16.2%** | **0.3%** | **8.3%** | **75.8%** |

The remaining 24.2% of the frames have no ball, in 301 gaps (median about 2.4 s, the longest
60 s).

**Single models, after post-processing**

| Model | Half | On-pitch detections | Impossible jumps (raw) | Ball present (track) | Impossible jumps (track) |
|---|---|---|---|---|---|
| `tikitalia_model` | 1st | 54.9% | 1.5% | 69.1% | 0.3% |
| `tikitalia_model` | 2nd | 56.4% | 2.3% | 71.1% | 0.5% |
| `roboflow` | 1st | 97.3% | 19.1% | 97.1% | 2.5% |
| `roboflow` | 2nd | 90.9% | 17.0% | 92.3% | 2.3% |

`roboflow` almost always answers but 17-19% of its raw detections are impossible jumps;
`tikitalia_model` answers less often but is consistent. When both have a detection they agree
(within 15 px) in 93% of the cases.

**ATD**: frames with the ball from 113,488 (56.9%) to **146,039 (73.3%)** out of 199,319. Of
these, 140,018 come from the ensemble (107,467 replaced, 32,551 added) and 6,021 from the
original ATD.

Details: `docs/pipeline.md`.

## 3. Model and ensemble comparison on the whole video

199,319 frames, including the parts that are not graded (so the default ensemble reads 70.2% here
and 75.8% in section 2).

| Model | On-pitch detections | Impossible jumps (raw) | Ball present (track) | Segments | Impossible jumps (track) |
|---|---|---|---|---|---|
| `tikitalia_model` | 54.1% | 6.3% | 66.0% | 1,601 | 0.9% |
| `tikitalia_tiles` | 66.3% | 7.3% | 75.8% | 1,836 | 0.7% |
| `roboflow` | 94.6% | 20.3% | 95.0% | 6,440 | 2.6% |

| Ensemble | Confirmed | Tracked | Recovered | Synthetic | **Ball present** |
|---|---|---|---|---|---|
| `tikitalia_model` + `tikitalia_tiles` | 47.6% | 12.0% | 0.0% | 9.8% | **69.4%** |
| `tikitalia_model` + `roboflow` (default) | 48.9% | 14.1% | 0.2% | 7.0% | **70.2%** |
| `tikitalia_tiles` + `roboflow` | 54.3% | 13.4% | 0.3% | 7.4% | **75.4%** |
| All three (2 votes out of 3) | 60.7% | 9.4% | 0.2% | 7.2% | **77.5%** |

Replacing `tikitalia_model` with `tikitalia_tiles` adds 5.2 points of ball present and 5.4 of
confirmed frames; using all three adds 7.3 and 11.8. The default ensemble and the ATD still use
`tikitalia_model` + `roboflow`: the other combinations have not been measured on the graded
halves alone, and the ATD has not been regenerated with them.

Details: `docs/results_tiles.md` and the `comparison*.md` reports in
`runs/ball_pipeline/BAR-ATH_tactical/` (not versioned).
