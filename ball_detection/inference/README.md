# inference/

This folder holds the scripts that, starting from the video, produce the ball position in every
frame. They are run one after the other, in the order of the table.

The pipeline on the video, in 4 steps. Prerequisites: weights in `weights/`, video and per-frame
homography (defaults in `utils/config.py`, otherwise `--video` and `--homography`).

| Step | Script | Output in `runs/ball_pipeline/<video>/` |
|---|---|---|
| 1 | `detect_ball.py` | `<model>/detections.csv` (up to 3 candidates per frame) |
| 2 | `post_process.py` | `<model>/track.csv` (one ball per frame, gaps filled) |
| 3 | `ball_consensus.py` | `ensemble_<models>/ball.csv` (**final result**) |
| 4 | `update_atd.py` | copy of the ATD with the ensemble's ball, next to the original |

```bash
python -m inference.detect_ball --model tikitalia_model
python -m inference.detect_ball --model roboflow
python -m inference.detect_ball --model roboflow --resume           # resumes if interrupted
python -m inference.detect_ball --model roboflow --frames 500       # quick test

python -m inference.post_process
python -m inference.ball_consensus
python -m inference.update_atd
```

Models: `tikitalia_model`, `tikitalia_tiles`, `roboflow`. Steps 2-4 use `tikitalia_model` +
`roboflow` by default; for another combination:

```bash
python -m inference.post_process --model tikitalia_tiles roboflow
python -m inference.ball_consensus --models tikitalia_tiles roboflow
```

Each step reads the CSVs of the previous one: steps 2-4 can be redone in a few minutes without
re-running inference (8-9 hours on the whole match). Algorithms and parameters: `docs/pipeline.md`.
