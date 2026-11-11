"""
Pipeline step 1: ball detection with ONE model on a video.
Each model runs separately and writes its own CSV in runs/ball_pipeline/<model>/.

Models (weights, origin and licences in weights/README.md):
  tikitalia_model  our YOLOv8s-P2 trained on SoccerNet (train/train.py), full frame at 1280 px
  tikitalia_tiles  our YOLOv8m-P2 trained on 640 tiles (train/train.py --tile 640), tiled
  roboflow         YOLOv8 'football-ball-detection' (roboflow/sports), 640x640 tiles with overlap

For every frame up to --topk detections are saved (ball centre + confidence):
    runs/ball_pipeline/<model>/detections.csv     frame, model, x, y, conf
    (frame = index in the given video, from 0)
The CSV is written in blocks: with --resume an interrupted run picks up where it stopped.

Threshold:
  --conf 0.15   low threshold for small objects: the real filter is the consensus

Usage (from the project root):
  python -m inference.detect_ball --model tikitalia_model                     # the whole video
  python -m inference.detect_ball --model roboflow
  python -m inference.detect_ball --model tikitalia_model --start 0 --frames 500   # quick test
  python -m inference.detect_ball --model roboflow --resume        # resumes an interrupted run
  python -m inference.detect_ball --model tikitalia_model --tikitalia-tile 640 \
      --tikitalia-weights runs/train/<name>-tiles640/weights/best.pt   # another tiled model
Then: post_process.py (track), ball_consensus.py (ensemble), analysis/compare_models.py (comparison).
"""
from pathlib import Path
import argparse
import csv
import time

import cv2
import numpy as np
import torch

from utils.config import (DEFAULT_VIDEO, TIKITALIA_WEIGHTS, ROBOFLOW_WEIGHTS, ALL_MODELS,
                          TIKITALIA_TILES_WEIGHTS, TIKITALIA_TILES_SIZE, detections_path, set_video)
from utils.io import read_progress
from utils.pitch import tile_grid


def topk_centers(xyxy, conf, k, dedup_px=8.0):
    """Centres of the k most confident detections, dropping duplicates (e.g. at tile borders)."""
    if len(conf) == 0:
        return []
    cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
    out = []
    for i in np.argsort(-conf):
        if all(np.hypot(cx[i] - x, cy[i] - y) > dedup_px for x, y, _ in out):
            out.append((float(cx[i]), float(cy[i]), float(conf[i])))
            if len(out) == k:
                break
    return out


class YoloFull:
    """YOLO on the full frame."""

    def __init__(self, weights, conf, imgsz, device):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def __call__(self, frames, k):
        res = self.model.predict(frames, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                 quantize=16 if self.device != "cpu" else None, verbose=False)
        return [topk_centers(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), k) for r in res]


class YoloTiled:
    """YOLO on square tiles with overlap: the ball covers more pixels than on the downscaled frame.
    All the tiles of all the frames in the batch go through the network in a single call."""

    def __init__(self, weights, conf, device, tile=640, imgsz=None, overlap=0.2):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf, self.device, self.tile, self.overlap = conf, device, tile, overlap
        self.imgsz = imgsz or tile          # imgsz > tile: the tile is upscaled
        self._grid = None

    def grid(self, h, w):
        if self._grid is None:
            self._grid = tile_grid(h, w, self.tile, self.overlap)
        return self._grid

    def __call__(self, frames, k):
        h, w = frames[0].shape[:2]
        grid = self.grid(h, w)
        tiles = [f[y:y + self.tile, x:x + self.tile] for f in frames for x, y in grid]
        res = self.model.predict(tiles, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                 quantize=16 if self.device != "cpu" else None, verbose=False)
        out = []
        for i in range(len(frames)):
            boxes, confs = [], []
            for (x, y), r in zip(grid, res[i * len(grid):(i + 1) * len(grid)]):
                b = r.boxes.xyxy.cpu().numpy()
                if len(b):
                    boxes.append(b + np.array([x, y, x, y]))
                    confs.append(r.boxes.conf.cpu().numpy())
            out.append(topk_centers(np.concatenate(boxes), np.concatenate(confs), k) if boxes else [])
        return out


def parse_args():
    p = argparse.ArgumentParser(description="Ball detection with one model")
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--start", type=int, default=0, help="first frame (of the given video)")
    p.add_argument("--frames", type=int, default=0, help="frames to process (0 = until the end)")
    p.add_argument("--tikitalia-weights", type=Path, default=TIKITALIA_WEIGHTS, help="weights of our model")
    p.add_argument("--tikitalia-tile", type=int, default=0,
                   help="tikitalia_model: 0 = full frame at 1280 px (default); N = NxN tiles with "
                        "20%% overlap, to be used with a model trained on tiles "
                        "(data_prep.prepare_soccernet --tile N, train.train --tile N)")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--tile", type=int, default=640, help="roboflow: tile side in video pixels")
    p.add_argument("--tile-imgsz", type=int, default=None,
                   help="roboflow: side the tile is resized to (default = --tile; "
                        "larger = ball upscaled)")
    p.add_argument("--topk", type=int, default=3, help="detections saved per frame")
    p.add_argument("--batch", type=int, default=4, help="frames per batch")
    p.add_argument("--out", type=Path, default=None, help="default: runs/ball_pipeline/<model>/detections.csv")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    set_video(args.video)
    args.out = args.out or detections_path(args.model)
    if args.model == "tikitalia_model" and args.tikitalia_tile > 0:
        detector = YoloTiled(args.tikitalia_weights, args.conf, device, args.tikitalia_tile)
    elif args.model == "tikitalia_model":
        detector = YoloFull(args.tikitalia_weights, args.conf, 1280, device)
    elif args.model == "tikitalia_tiles":
        detector = YoloTiled(TIKITALIA_TILES_WEIGHTS, args.conf, device, TIKITALIA_TILES_SIZE)
    else:
        detector = YoloTiled(ROBOFLOW_WEIGHTS, args.conf, device, args.tile, args.tile_imgsz)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if args.frames <= 0 else min(total, args.start + args.frames)

    # progress file: "first,last" completed frame (for --resume and for the following scripts)
    progress = args.out.with_suffix(".progress")
    first = start = args.start
    if args.resume and progress.exists():
        first, last = read_progress(progress)
        start = last + 1
        print(f"[INFO] Resuming from frame {start}")
    elif args.out.exists():
        args.out.unlink()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    print(f"[INFO] {args.video.name}: frames {start}-{end - 1} ({end - start} frames), model {args.model}")

    t0, done = time.time(), 0
    with open(args.out, "a", newline="") as fcsv:
        w = csv.writer(fcsv)
        if new_file:
            w.writerow(["frame", "model", "x", "y", "conf"])
        idx = start
        while idx < end:
            frames = []
            while len(frames) < args.batch and idx + len(frames) < end:
                ok, f = cap.read()
                if not ok:
                    end = idx + len(frames)
                    break
                frames.append(f)
            if not frames:
                break
            for i, dets in enumerate(detector(frames, args.topk)):
                for x, y, c in dets:
                    w.writerow([idx + i, args.model, f"{x:.1f}", f"{y:.1f}", f"{c:.3f}"])
            idx += len(frames)
            done += len(frames)
            if done % (args.batch * 250) == 0 or idx >= end:
                fcsv.flush()
                progress.write_text(f"{first},{idx - 1}")
                fps = done / (time.time() - t0)
                eta = (end - idx) / fps / 60
                print(f"[{idx - start}/{end - start}] {fps:.1f} frames/s, ETA {eta:.0f} min", flush=True)
    cap.release()
    print(f"[INFO] Detections saved to {args.out}")


if __name__ == "__main__":
    main()
