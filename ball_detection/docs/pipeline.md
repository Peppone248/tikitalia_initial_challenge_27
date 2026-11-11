# Pipeline and results on the BAR-ATH tactical video

*Results of 11 November 2026. Commands: `inference/README.md`.*

**Goal**: ball position in every frame of `BAR-ATH_tactical.mp4` (199,319 frames, 25 fps).
The 1st half (frames 11,185-90,651) and the 2nd half (frames 110,808-190,197) are graded.

**Result**: the ensemble `tikitalia_model` + `roboflow` gives the ball in **75.8%** of the graded
frames (51.0% confirmed by both models).

## 1. Models

| Name | What it is | Trained on | Inference |
|---|---|---|---|
| `tikitalia_model` | Our YOLOv8s-P2 | SoccerNet-Tracking 2023, ball only | Full frame at 1280 px |
| `tikitalia_tiles` | Our YOLOv8m-P2 | Same data, in 640 tiles | 640×640 tiles, 20% overlap |
| `roboflow` | YOLOv8x `football-ball-detection` from [roboflow/sports](https://github.com/roboflow/sports) | [Roboflow dataset](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) | 640×640 tiles, 20% overlap |

Confidence threshold 0.15 (low: the real filter is post-processing and consensus); the 3 best
detections per frame are saved.

## 2. Training data

Every labelled dataset behind the models, as required by the challenge rules. **None of it comes
from the provided game** (FC Barcelona vs Athletic Club, LaLiga 2024-25), and we produced no
labels of our own, on this game or on any other.

| Dataset | Used for | Footage | Labels | Amount we used |
|---|---|---|---|---|
| [SoccerNet-Tracking 2023](https://github.com/SoccerNet/sn-tracking) | Training and validation of `tikitalia_model` and `tikitalia_tiles` | 30 s main-camera clips from the SoccerNet games; dataset published in 2022 (Cioppa et al., CVPR 2022) | Bounding boxes and tracklet IDs distributed with the dataset by its authors. We keep only the tracklets labelled `ball` and add or correct nothing | Public `train` + `test` splits: 106 sequences from 6 games. Full frames: 12,993 training images (12,838 with the ball) and 1,985 validation images (1,970 with the ball). Tiles: about 35,000 training tiles cut from the same frames |
| [Roboflow `football-ball-detection`](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) | Pre-trained `roboflow` detector, used as published: we did not train or fine-tune it | Per the [roboflow/sports](https://github.com/roboflow/sports/tree/main/examples/soccer) README, the original data comes from the Kaggle competition [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) | Published by Roboflow on Roboflow Universe | The whole dataset, as used by Roboflow to train the published weights (image count on the dataset page) |
| COCO | Initialisation only: `yolov8s.pt` / `yolov8m.pt` base weights from Ultralytics | General-purpose photographs, no football match data | COCO annotations | Pre-trained weights, downloaded automatically |

How the provided game is used:

- **No human labels and no ground truth.** Neither the public 5 minutes nor any other annotation
  of the game is read anywhere in this folder.
- **Inference only.** The detectors run on the tactical video and the post-processing uses the
  provided homography. The parameters are physical or generic (section 3), none is tied to
  specific frames.
- **Model comparison without labels.** The statistics in section 5 and in `docs/results_tiles.md`
  measure coverage and agreement between detectors, which no person decided.
- **Validation of our models** is done on SoccerNet game 11, held out from training
  (`docs/training.md`).

## 3. Method

**Step 1, detection.** Up to 3 candidates per frame and per model (centre and confidence).

**Step 2, post-processing of each model** → `track.csv`

1. Candidates outside the pitch are discarded (homography, 3 m margin).
2. Viterbi over the whole sequence: for every frame one candidate or "none". Score =
   logit of the confidence + continuity bonus (if reachable at ≤ 40 m/s) − linearity penalty
   (capped: a kick does not break the track). After 0.5 s without detections a new segment
   starts.
3. Segments with fewer than 3 detections are discarded.
4. Gaps: up to 0.5 s always (spline); up to 2 s only if the ball is slow (straight line). A
   discarded candidate within 25 px of the trajectory is recovered (`recovered`), otherwise a
   synthetic point is inserted (`interpolated`). The trajectory is computed in metres on the pitch.

**Step 3, ensemble** → `ball.csv`

1. Per-frame confirmation: candidates of different models within 15 px → `confirmed`, mean
   position. Isolated confirmations (no other one within ±0.2 s) are discarded.
2. Per-segment confirmation: a segment that goes through at least one confirmed frame is trusted;
   its other detections count as `tracked`. Segments that are never confirmed are ignored.
3. Gaps: as in step 2.

**Step 4, ATD.** Copy of the ATD (FIFA/EPTS format) in which `x`, `y` (normalised 0-1) and the
ball speed are the ensemble's; where the ensemble has no ball the original one stays. The row
with `frameCount` = k is frame k − 1 of the video.

### Parameters

| Parameter | Value | Reason |
|---|---|---|
| Maximum ball speed | 40 m/s | Physical limit |
| Tolerated position error | 0.5 m | Detection and homography noise |
| Continuity bonus | +2 | A detection on the trajectory counts even at conf 0.15 (logit −1.7) |
| Maximum penalty for a change of direction | 1 | Smaller than the bonus: a kick does not break the track |
| Track alive without detections | 0.5 s | Beyond that a touch or a bounce is likely |
| Minimum detections per segment | 3 | A real ball is seen for more than an instant |
| Gap always filled | 0.5 s | Beyond that, with a fast ball, no simple trajectory is reliable |
| Gap filled if the ball is slow | 2 s, ≤ 6 m/s | Ball at a player's feet or rolling |
| Agreeing models required | 2 | Detectors trained on different data rarely fail at the same spot |
| Agreement distance | 15 px | |
| Anti-isolation window | 0.2 s | |
| Tolerance for `recovered` | 25 px | The trajectory is an estimate |

## 4. CSV format

`ensemble_<models>/ball.csv`, one row per frame:

| Column | Meaning |
|---|---|
| `frame` | Index in the video (from 0) |
| `x`, `y` | Position in pixels (empty if `none`) |
| `pitch_x_m`, `pitch_y_m` | Position on the pitch in metres |
| `status` | `confirmed` > `tracked` > `recovered` > `interpolated` (decreasing reliability), `none` |
| `source` | Agreeing models, single model, or `spline` / `linear` |

`<model>/track.csv`: same columns with `status` = `detected` · `recovered` · `interpolated` ·
`none`, plus `conf` and `segment` (segment id; −1 for a synthetic point between two segments).

## 5. Results

On the two graded halves (158,857 frames). Without ground truth, behaviour and agreement are
measured. Tables on the whole video: `runs/ball_pipeline/BAR-ATH_tactical/comparison.md`.

**Ensemble `tikitalia_model` + `roboflow`**

| | Confirmed | Tracked | Recovered | Synthetic | **Ball present** |
|---|---|---|---|---|---|
| 1st half | 50.5% | 16.0% | 0.4% | 7.7% | **74.5%** |
| 2nd half | 51.6% | 16.4% | 0.2% | 8.9% | **77.1%** |
| **Total** | **51.0%** | **16.2%** | **0.3%** | **8.3%** | **75.8%** |

24.2% of the frames are left without the ball, in 301 gaps (median about 2.4 s, the longest 60 s).

**Single models**

| Model | Half | On-pitch detections | Impossible jumps (raw) | Ball present (track) | Impossible jumps (track) |
|---|---|---|---|---|---|
| `tikitalia_model` | 1st | 54.9% | 1.5% | 69.1% | 0.3% |
| `tikitalia_model` | 2nd | 56.4% | 2.3% | 71.1% | 0.5% |
| `roboflow` | 1st | 97.3% | 19.1% | 97.1% | 2.5% |
| `roboflow` | 2nd | 90.9% | 17.0% | 92.3% | 2.3% |

`roboflow` answers almost always but is not reliable on its own (17-19% impossible jumps);
`tikitalia_model` answers less often but is consistent. When both have a detection they agree
(≤ 15 px) in 93% of the cases.

**ATD**: frames with the ball from 113,488 (56.9%) to **146,039 (73.3%)**: 140,018 from the
ensemble (107,467 replaced, 32,551 added) and 6,021 from the original ATD.

Comparison with `tikitalia_tiles`: `docs/results_tiles.md`.

## 6. Licences

> Check made on the public sources, not legal advice.

| Component | Licence | Obligations |
|---|---|---|
| Ultralytics (YOLOv8) and base weights | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | Code published under a compatible licence |
| `tikitalia_model`, `tikitalia_tiles` | AGPL-3.0 (derivatives) | As above |
| SoccerNet-Tracking 2023 | Public access, research use | Cite Cioppa et al., 2022; do not redistribute the data |
| `roboflow/sports` code | [MIT](https://github.com/roboflow/sports/blob/main/LICENSE) | Keep the copyright notice |
| `football-ball-detection.pt` weights | AGPL-3.0 | As for Ultralytics |
| Roboflow dataset | CC BY 4.0 | Attribution to Roboflow |
