# YoloBallDetection

Ball position in every frame of a tactical football video: several YOLO detectors, a clean track
per model and a consensus ensemble.

This folder holds all the work on the ball, from data preparation to the final file with the
frame-by-frame position. This README gives the overall picture and the main commands; each
subfolder then has its own README with the details.

## Structure

The code is split by stage of the work, one folder each:

| Folder | Content |
|---|---|
| `data_prep/` | SoccerNet-Tracking → YOLO dataset (ball only) |
| `train/` | Training and evaluation of our models |
| `inference/` | The 4 steps of the pipeline on the video |
| `analysis/` | Comparison statistics between models and ensembles |
| `utils/` | Configuration, CSV reading, homography and physics |
| `weights/` | Model weights (ours are versioned, the others are downloaded) |
| `docs/` | Method, parameters, results, training data, licences |

Not versioned: `data/` (`soccernet_raw/` = downloaded SoccerNet, `yolo/` = converted datasets) and
`runs/` (training runs and pipeline output).

## Installation

```bash
python -m venv .venv && .venv\Scripts\activate
# PyTorch with CUDA: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

All commands are run **from this folder** (`ball_detection/`) with `python -m <folder>.<script>`.

## Pipeline from scratch

All the steps are listed below in order, from the dataset to the final result. If the weights are
already in `weights/`, you can start directly from step 3.

```bash
# 1. dataset (full frames and 640 tiles)
python -m data_prep.prepare_soccernet --val-games 11 --clean
python -m data_prep.prepare_soccernet --skip-download --val-games 11 --clean --tile 640

# 2. training, then copy the chosen weights to weights/ (see weights/README.md)
python -m train.train --model yolov8s-p2.yaml --epochs 50 --patience 10 --batch -1

# 3. detection on the video, one model at a time
python -m inference.detect_ball --model tikitalia_model
python -m inference.detect_ball --model roboflow

# 4. clean track of each model
python -m inference.post_process

# 5. ensemble -> runs/ball_pipeline/<video>/ensemble_<models>/ball.csv
python -m inference.ball_consensus

# 6. ATD with the ensemble's ball
python -m inference.update_atd
```

The default video, homography and ATD are looked up in `data/tactical/` in the repository root,
or in the folder set by the `MTAH_DATA_DIR` environment variable (as in `games.py`); `--video` and
`--homography` point to different ones. Details in each folder (`README.md`) and in `docs/`.
