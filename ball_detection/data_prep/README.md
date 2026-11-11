# data_prep/

This folder holds the script that prepares the training data. It is the first step of the work
and is only needed to retrain the models.

`prepare_soccernet.py` downloads SoccerNet-Tracking 2023 to `data/soccernet_raw/` and converts it
into a YOLO dataset with the single class `ball`.

```bash
# full frames -> data/yolo/soccernet_ball (game 11 in validation)
python -m data_prep.prepare_soccernet --val-games 11 --clean

# 640x640 tiles, 20% overlap -> data/yolo/soccernet_ball_tiles640
python -m data_prep.prepare_soccernet --skip-download --val-games 11 --clean --tile 640
```

- `--skip-download`: uses the data already in `data/soccernet_raw/tracking-2023/`.
- `--clean`: empties the output before converting (required if the split changes).
- Defaults: 1 frame every 5, 15% of the frames without the ball as negatives, boxes under 2 px discarded.

Output: `images/{train,val}`, `labels/{train,val}`, `data.yaml`. The download takes about 17 GB.

The labels are the ones distributed with SoccerNet-Tracking; the script adds and corrects none.
Source and amounts are in `docs/pipeline.md` (section "Training data").
