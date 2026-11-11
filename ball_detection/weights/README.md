# weights/

This folder holds the weights of the models used by the pipeline. The table says where each file
comes from, the commands below how to get the missing ones.

Only ours are versioned (`tikitalia_model.pt`, `tikitalia_tiles.pt`);
`roboflow_ball.pt` (130 MB) and the COCO bases are downloaded.

| File | Model | Origin | Licence |
|---|---|---|---|
| `tikitalia_model.pt` | YOLOv8s-P2, full frames at 1280 px | `runs/train/yolov8s-p2-4/weights/best.pt` | AGPL-3.0 |
| `tikitalia_tiles.pt` | YOLOv8m-P2, 640 tiles | `runs/train/yolov8m-p2-tiles640/weights/last.pt` (epoch 12) | AGPL-3.0 |
| `roboflow_ball.pt` | YOLOv8x `football-ball-detection` | [roboflow/sports](https://github.com/roboflow/sports) | AGPL-3.0 (weights), CC BY 4.0 (dataset) |
| `yolov8s.pt`, `yolov8m.pt` | COCO bases for training | Ultralytics, downloaded automatically | AGPL-3.0 |

```bash
# Roboflow weights (same link as examples/soccer/setup.sh in roboflow/sports)
gdown -O weights/roboflow_ball.pt "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"

# our models: after training, copy the chosen weights
copy runs\train\<name>\weights\best.pt weights\tikitalia_model.pt
```

The paths are defined in `utils/config.py`. Training data behind each model and licence details:
`docs/pipeline.md`.
