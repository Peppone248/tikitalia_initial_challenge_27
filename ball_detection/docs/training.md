# Training of `tikitalia_model`

*Results of 7 November 2026. Commands: `data_prep/README.md` and `train/README.md`.*

| | |
|---|---|
| Weights | `weights/tikitalia_model.pt` (run `runs/train/yolov8s-p2-4`, `best.pt`) |
| Architecture | YOLOv8s-P2, 10.9 M parameters |
| Initialisation | COCO weights of `yolov8s.pt` in the compatible layers; the P2 head starts from scratch |
| Data | SoccerNet-Tracking 2023, 6 games: 5 in training, 1 in validation |
| Duration | 35 epochs (early stop, best at 25), 7.4 hours on an RTX 3070 Ti Laptop 8 GB |
| Validation | precision 0.574 · recall 0.442 · **mAP50 0.426** · mAP50-95 0.146 |
| Software | Ultralytics 8.4.202, PyTorch with CUDA, Python 3.11 |

## 1. Dataset

[SoccerNet-Tracking 2023](https://github.com/SoccerNet/sn-tracking): 106 sequences of 30 s
(25 fps, 1920×1080) of main-camera footage, in MOT format. From the conversion:

- only the tracklets labelled `ball`, class 0;
- 1 frame every 5, boxes under 2 px discarded, 15% of the frames without the ball as negatives;
- images as hardlinks to the originals (no extra space).

The labels are the ones distributed with the dataset by its authors: we added and corrected none.
None of these games is the one provided for the challenge; the full list of labelled data behind
every model is in `docs/pipeline.md` (section "Training data").

Split **by game**, so that validation measures generalisation to a game never seen:

| Split | Games (gameID) | Sequences | Images | With ball | Box side (median, 10th-90th percentile) |
|---|---|---|---|---|---|
| Training | 4, 6, 7, 8, 9 | 92 | 12,993 | 12,838 | 16 px (10-23) |
| Validation | 11 | 14 | 1,985 | 1,970 | 13 px (8-18) |

## 2. Model and parameters

YOLOv8s-P2 adds a head at stride 4 to the three standard ones (8, 16, 32): with a ball of a few
pixels it keeps enough resolution to tell it apart.

| Parameter | Value | Reason |
|---|---|---|
| `imgsz` | 1280 | At 640 px the ball shrinks to a few pixels |
| `batch` | auto (2) | The maximum in 8 GB at 1280 px with the P2 head |
| `epochs` / `patience` | 50 / 10 | |
| `lr0` / `lrf`, `cos_lr` | 0.01 / 0.01, cosine | |
| `single_cls`, `amp` | yes | |
| `mosaic` / `close_mosaic` | 1.0 / 10 | Last 10 epochs on realistic images |
| `scale` | 0.6 | Also teaches balls smaller than SoccerNet's |
| `fliplr` | 0.5 | |
| `flipud` / `degrees` | 0 / 0 | The pitch neither flips nor rotates |
| `hsv_h` / `hsv_s` / `hsv_v` | 0.015 / 0.6 / 0.5 | Daylight and evening light, sharp shadows |
| `mixup` / `erasing` | 0 / 0 | With objects of a few pixels they confuse or erase the ball |

## 3. Results

| Epoch | 1 | 5 | 10 | 15 | 20 | **25** | 30 | 35 |
|---|---|---|---|---|---|---|---|---|
| mAP50 | 0.283 | 0.350 | 0.401 | 0.424 | 0.388 | **0.426** | 0.399 | 0.394 |
| mAP50-95 | 0.090 | 0.119 | 0.129 | 0.138 | 0.137 | **0.146** | 0.132 | 0.130 |

The low mAP50-95 is typical of objects this small: on a 13 px box a few pixels of error push the
IoU below the high thresholds.

**Evaluation on the centre** (`train/eval_center.py`): the pipeline needs the centre, not the box.
A detection is correct if its centre is at most N px away from an annotated ball.

| Validation, game 11 | N | P | R | F1 |
|---|---|---|---|---|
| conf 0.25 | 5 px | 0.780 | 0.464 | 0.582 |
| conf 0.25 | 10 px | 0.913 | 0.543 | 0.681 |
| conf 0.20 (maximum F1) | 10 px | 0.852 | 0.610 | 0.711 |

Plots: `runs/train/yolov8s-p2-4/`.

## 4. Tiled variant (`tikitalia_tiles`)

Training on 640×640 tiles with 20% overlap, the same grid as inference (8 tiles per frame,
`tile_grid` in `utils/pitch.py`). The ball stays at native resolution and each step costs about
4 times less: YOLOv8m-P2 becomes affordable on 8 GB.

- tiles with the ball: always; empty tiles: 10%;
- ball cut by the edge and less than 50% visible: tile discarded (it is whole in a neighbouring
  tile);
- about 35,000 training tiles, 4 GB.

The mAP on tiles is **not comparable** with the full-frame one: the two models are compared on
the video (`docs/results_tiles.md`).
