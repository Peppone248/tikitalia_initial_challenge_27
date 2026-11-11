# train/

This folder holds the scripts to train our models and to measure how accurately they find the
ball.

Prerequisite: dataset created with `data_prep/prepare_soccernet.py`.

| Script | What it does |
|---|---|
| `train.py` | Fine-tuning of YOLOv8-P2 (single class `ball`) |
| `eval_center.py` | Precision/recall on the ball centre (within N px) |

```bash
# tikitalia_model: YOLOv8s-P2, full frames at 1280 px
python -m train.train --model yolov8s-p2.yaml --epochs 50 --patience 10 --batch -1

# tikitalia_tiles: YOLOv8m-P2, 640 tiles
python -m train.train --model yolov8m-p2.yaml --tile 640 --epochs 50 --patience 10 --batch -1

# quick test / resume
python -m train.train --epochs 1 --fraction 0.05
python -m train.train --resume runs/train/<name>/weights/last.pt

# evaluation on the centre
python -m train.eval_center --weights weights/tikitalia_model.pt
python -m train.eval_center --weights weights/tikitalia_tiles.pt --tile 640 --radius 5 10 20
```

Output in `runs/train/<name>/` (`weights/best.pt`, `last.pt`, `results.csv`, plots). To use a
model in the pipeline, copy its weights to `weights/` (see `weights/README.md`).

`--cache disk` (default) speeds up the epochs but creates `.npy` files next to the images: tens of
GB, they can be deleted once training is over. With little disk space use `--cache none`.
