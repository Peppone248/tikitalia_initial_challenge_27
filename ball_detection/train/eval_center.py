"""
Evaluation of a ball model on the CENTRE: a detection is correct if its centre is at most
N pixels away from the centre of an annotated ball.

Why: with a ball of ~13 px, mAP (IoU >= 0.5) also counts as an error a detection at the right
spot but with a box 2-3 px larger or smaller. The pipeline (tracking and ensemble) only needs
the centre, so this measure is closer to what matters.

Rules:
  - the detections of each image are taken in order of confidence; each one is matched to the
    nearest free annotated ball within N px (true positive), otherwise it is a false positive;
  - the annotated balls left without a detection are false negatives;
  - N is in pixels of the original image (640 tile or 1920x1080 frame).

For every N it prints precision, recall and F1 at the --conf threshold and at the threshold that maximises F1.

Usage (from the project root):
  python -m train.eval_center --weights weights/tikitalia_tiles.pt --tile 640
  python -m train.eval_center --weights weights/tikitalia_model.pt                 # full frames
  python -m train.eval_center --weights ... --radius 3 5 10 20 --conf 0.15
"""
from pathlib import Path
import argparse

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def match_image(pred_xy, pred_conf, gt_xy, radius):
    """TP (True) / FP (False) flag for every detection, in order of decreasing confidence.
    The greedy matching by confidence makes the result at a higher threshold identical to the one
    that would be obtained by re-running the model with that threshold."""
    order = np.argsort(-pred_conf)
    free = np.ones(len(gt_xy), dtype=bool)
    tp = np.zeros(len(order), dtype=bool)
    for k, i in enumerate(order):
        if not free.any():
            break
        d = np.hypot(*(gt_xy - pred_xy[i]).T)
        d[~free] = np.inf
        j = int(np.argmin(d))
        if d[j] <= radius:
            free[j] = False
            tp[k] = True
    return pred_conf[order], tp


def gt_centers(label_path: Path, w, h):
    if not label_path.exists():
        return np.zeros((0, 2))
    rows = [list(map(float, l.split()[1:3])) for l in label_path.read_text().splitlines() if l.strip()]
    return np.array(rows, dtype=float).reshape(-1, 2) * [w, h]


def scores(confs, tps, n_gt, thr):
    keep = confs >= thr
    tp = int(tps[keep].sum())
    p = tp / max(int(keep.sum()), 1)
    r = tp / max(n_gt, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def evaluate(weights, images_dir, imgsz, radii=(5, 10), conf=0.25, min_conf=0.05, batch=16,
             device=None):
    """Returns {N: dict(...)} and prints the table. Also usable from train.py."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"[ERROR] no images in {images_dir}")
    labels_dir = Path(str(images_dir).replace("images", "labels"))

    per_radius = {r: ([], []) for r in radii}
    n_gt = 0
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        res = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=min_conf, device=device,
                            verbose=False)
        for p, r in zip(chunk, res):
            h, w = r.orig_shape
            gt = gt_centers(labels_dir / f"{p.stem}.txt", w, h)
            n_gt += len(gt)
            xyxy = r.boxes.xyxy.cpu().numpy()
            pc = r.boxes.conf.cpu().numpy()
            pxy = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2], 1)
            for rad in radii:
                c, t = match_image(pxy, pc, gt, rad)
                per_radius[rad][0].append(c)
                per_radius[rad][1].append(t)

    thresholds = np.round(np.arange(min_conf, 0.951, 0.05), 2)
    out = {}
    print(f"\n===== Evaluation on the centre: {len(images)} images, {n_gt} annotated balls =====")
    print(f"{'N px':>5} | {'conf':>5} {'P':>6} {'R':>6} {'F1':>6} | {'conf*':>5} {'P':>6} {'R':>6} {'F1':>6}")
    for rad in radii:
        confs = np.concatenate(per_radius[rad][0])
        tps = np.concatenate(per_radius[rad][1])
        p, r, f = scores(confs, tps, n_gt, conf)
        best = max(((t, *scores(confs, tps, n_gt, t)) for t in thresholds), key=lambda x: x[3])
        out[rad] = dict(conf=conf, precision=p, recall=r, f1=f,
                        best_conf=float(best[0]), best_precision=best[1], best_recall=best[2], best_f1=best[3])
        print(f"{rad:>5} | {conf:>5.2f} {p:>6.3f} {r:>6.3f} {f:>6.3f} | "
              f"{best[0]:>5.2f} {best[1]:>6.3f} {best[2]:>6.3f} {best[3]:>6.3f}")
    print("conf* = confidence threshold with maximum F1")
    return out


def main():
    p = argparse.ArgumentParser(description="Ball evaluation on the centre (within N px).")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--tile", type=int, default=0,
                   help="0 = full-frame dataset (imgsz 1280); N = NxN tiled dataset (imgsz N)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Dataset folder (default: data/yolo/soccernet_ball or _tiles<N>)")
    p.add_argument("--split", default="val")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--radius", type=float, nargs="+", default=[5, 10], help="Distances N in pixels")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for the first table")
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    data_dir = args.data_dir or Path(
        f"data/yolo/soccernet_ball_tiles{args.tile}" if args.tile else "data/yolo/soccernet_ball")
    imgsz = args.imgsz or args.tile or 1280
    evaluate(args.weights, data_dir / "images" / args.split, imgsz,
             radii=[int(r) if float(r).is_integer() else r for r in args.radius],
             conf=args.conf, batch=args.batch)


if __name__ == "__main__":
    main()
