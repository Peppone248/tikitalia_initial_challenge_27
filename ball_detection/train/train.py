"""
YOLO fine-tuning for ball detection (single class 'ball').

Default model: yolov8s-p2 = YOLOv8s with an extra head at stride 4,
designed for very small objects (the ball in a tactical shot is ~5-10 px).
The P2 architecture has no pre-trained weights: those of yolov8s.pt are loaded
for all compatible layers (backbone and most of the neck).

Typical usage (from the project root):
    python -m train.train                                   # full training with the defaults
    python -m train.train --batch 2 --imgsz 1280            # if VRAM runs out
    python -m train.train --model weights/yolov8s.pt        # standard model without P2
    python -m train.train --resume runs/train/yolov8s-p2/weights/last.pt
    python -m train.train --epochs 1 --fraction 0.05        # quick test of the pipeline
    python -m train.train --model yolov8m-p2.yaml --tile 640 --batch -1
        # 640x640 tiles (dataset from data_prep.prepare_soccernet --tile 640), imgsz 640,
        # output in runs/train/yolov8m-p2-tiles640
"""
from pathlib import Path
import argparse
import torch
from ultralytics import YOLO

from utils.config import TRAIN_RUNS, WEIGHTS_DIR

DEFAULT_DATA = Path("data/yolo/soccernet_ball/data.yaml")

def build_model(model_name: str, pretrained: str | None):
    """Builds the model. For a .yaml it loads the compatible pre-trained weights."""
    if model_name.endswith(".yaml"):
        model = YOLO(model_name)
        if pretrained:
            print(f"[INFO] Transferring weights from {pretrained} to {model_name}")
            model.load(pretrained)
        else:
            print("[WARNING] No pre-trained weights: training from scratch (not recommended).")
        return model
    return YOLO(model_name)


def default_pretrained(model_name: str):
    """yolov8s-p2.yaml -> weights/yolov8s.pt (same size; downloaded from Ultralytics if missing)."""
    if model_name.endswith(".yaml"):
        name = Path(model_name).name.replace("-p2", "").replace("-p6", "").replace(".yaml", ".pt")
        return str(WEIGHTS_DIR / name)
    return None


def parse_args():
    p = argparse.ArgumentParser(description="YOLO fine-tuning for ball detection.")
    p.add_argument("--data", type=Path, default=None,
                   help="data.yaml of the dataset (default: data/yolo/soccernet_ball, or "
                        "data/yolo/soccernet_ball_tiles<N> with --tile N)")
    p.add_argument("--tile", type=int, default=0, metavar="N",
                   help="0 = full frames (default). N = NxN tiled dataset created with "
                        "data_prep.prepare_soccernet --tile N: default imgsz N, name with suffix -tiles<N>")
    p.add_argument("--model", default="yolov8s-p2.yaml", help=".yaml config or .pt weights")
    p.add_argument("--pretrained", default="auto",
                   help="Weights to transfer if --model is a .yaml ('auto', a .pt, or 'none')")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20, help="Early stopping if validation does not improve")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Training resolution (default 1280 on full frames: otherwise the ball "
                        "disappears in the resize; N with --tile N: the tile stays at native resolution)")
    p.add_argument("--batch", type=int, default=4,
                   help="4 is safe for 8 GB of VRAM at 1280 with P2; -1 = auto-batch")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", choices=["none", "ram", "disk"], default="disk",
                   help="'disk' speeds up the following epochs but takes a lot of space")
    p.add_argument("--fraction", type=float, default=1.0, help="Fraction of the train set to use")
    p.add_argument("--project", default=str(TRAIN_RUNS),
                   help="Runs folder (absolute path: Ultralytics does not add runs/detect)")
    p.add_argument("--name", default=None, help="Experiment name (default: model name)")
    p.add_argument("--resume", type=Path, default=None, help="Resumes from a last.pt")
    p.add_argument("--mosaic", type=float, default=1.0,
                   help="Mosaic probability (4 images merged into one). 0 = disabled, "
                        "recommended with --tile: the tiles are already crops of the frame")
    p.add_argument("--save-period", type=int, default=-1,
                   help="Saves the weights every N epochs (weights/epochN.pt); -1 = best/last only. Useful "
                        "to choose the checkpoint with eval_center.py instead of mAP50-95")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume:
        print(f"[INFO] Resuming training from {args.resume}")
        # cache and workers can be changed on resume; the other parameters stay those of the checkpoint
        YOLO(str(args.resume)).train(resume=True, cache=False if args.cache == "none" else args.cache,
                                     workers=args.workers)
        return

    if args.data is None:
        args.data = Path(f"data/yolo/soccernet_ball_tiles{args.tile}/data.yaml") if args.tile else DEFAULT_DATA
    if args.imgsz is None:
        args.imgsz = args.tile or 1280
    if not args.data.exists():
        hint = f" --tile {args.tile}" if args.tile else ""
        raise SystemExit(f"[ERROR] {args.data} not found. Run "
                         f"python -m data_prep.prepare_soccernet{hint} first.")
    print(f"[INFO] Dataset {args.data}, imgsz {args.imgsz}")

    use_cuda = torch.cuda.is_available()
    print("CUDA available:", use_cuda)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("[WARNING] Training on CPU: at 1280 px it will be extremely slow.")

    pretrained = default_pretrained(args.model) if args.pretrained == "auto" else (
        None if args.pretrained == "none" else args.pretrained)
    model = build_model(args.model, pretrained)
    name = args.name or Path(args.model).stem + (f"-tiles{args.tile}" if args.tile else "")

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        fraction=args.fraction,
        device=0 if use_cuda else "cpu",
        project=str(Path(args.project).resolve()),
        name=name,
        seed=args.seed,
        save_period=args.save_period,
        single_cls=True,
        amp=True,
        cos_lr=True,
        close_mosaic=10,      # last 10 epochs without mosaic: more "realistic" images

        # --- Augmentations ---
        # scale: random zoom in the range [1-scale, 1+scale]. SoccerNet is more zoomed in than the
        # tactical shot: a wide range also teaches the model smaller balls.
        scale=0.6,
        mosaic=args.mosaic,
        translate=0.1,
        fliplr=0.5,
        flipud=0.0,           # the pitch does not flip vertically
        degrees=0.0,
        # Strong brightness/saturation changes: sharp shadows on the pitch, daylight/evening light
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.5,
        mixup=0.0,            # with objects of a few pixels mixup confuses more than it helps
        erasing=0.0,          # random erasing could erase the ball itself
    )

    # Final evaluation of best.pt on the validation set
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\n[INFO] Evaluating {best}")
    metrics = YOLO(str(best)).val(data=str(args.data), imgsz=args.imgsz, batch=args.batch,
                                  device=0 if use_cuda else "cpu", split="val",
                                  project=str(save_dir), name="final_val", exist_ok=True)
    print("\n========== RESULTS (val) ==========")
    print(f"Precision : {metrics.box.mp:.3f}")
    print(f"Recall    : {metrics.box.mr:.3f}")
    print(f"mAP50     : {metrics.box.map50:.3f}")
    print(f"mAP50-95  : {metrics.box.map:.3f}")
    print(f"Weights   : {best}")

    # Evaluation on the centre (within 5 and 10 px): closer to what the pipeline needs than mAP
    from train.eval_center import evaluate
    evaluate(best, args.data.parent / "images" / "val", args.imgsz, radii=(5, 10),
             device=0 if use_cuda else "cpu")


if __name__ == "__main__":
    main()
