"""
Downloads SoccerNet-Tracking (2023) and converts it into a YOLO dataset with the single class 'ball'.

Expected structure of each SoccerNet sequence (MOT format):
    SNMOT-XXX/
        img1/000001.jpg ...
        gt/gt.txt          -> frame,id,x,y,w,h,conf,-1,-1,-1
        seqinfo.ini        -> imWidth, imHeight, imDir, imExt, ...
        gameinfo.ini       -> trackletID_<id>= <class>;<info>   (e.g. "ball;1")

Output:
    <out>/images/{train,val}/<seq>_<frame>.jpg
    <out>/labels/{train,val}/<seq>_<frame>.txt   (class 0 = ball)
    <out>/data.yaml

Typical usage (from the project root):
    python -m data_prep.prepare_soccernet                       # download + convert
    python -m data_prep.prepare_soccernet --skip-download       # conversion only
    python -m data_prep.prepare_soccernet --frame-step 3 --empty-ratio 0.1
    python -m data_prep.prepare_soccernet --skip-download --val-games 11 --clean
        # split by game: the sequences of game 11 go to val, all the others
        # (from both SoccerNet's train and test) go to train
    python -m data_prep.prepare_soccernet --skip-download --val-games 11 --clean --tile 640
        # 640x640 tiled dataset (20% overlap) in data/yolo/soccernet_ball_tiles640:
        # the model is trained at 640 px with the ball at native resolution
        # (train.train --tile 640)
"""
from pathlib import Path
import argparse
import configparser
import os
import random
import shutil
import zipfile
from collections import defaultdict

TASK = "tracking-2023"
# SoccerNet split -> YOLO split. Train and val are different sequences: no leakage between consecutive frames.
SPLIT_MAP = {"train": "train", "test": "val"}


# ----------------------------------------------------------------------------- download

def download(root: Path, splits):
    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=str(root))
    print(f"[INFO] Downloading SoccerNet {TASK} splits {splits} to {root} (several GB, it may take a while)...")
    downloader.downloadDataTask(task=TASK, split=list(splits))


def extract_zips(task_dir: Path, splits):
    for split in splits:
        zip_path = task_dir / f"{split}.zip"
        dest = task_dir / split
        if dest.exists() and any(dest.rglob("seqinfo.ini")):
            print(f"[INFO] {split}: already extracted in {dest}")
            continue
        if not zip_path.exists():
            print(f"[WARNING] {zip_path} not found, skipping split '{split}'.")
            continue
        print(f"[INFO] Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            # The zip may or may not contain the '<split>/' folder at the top: extract into a
            # dedicated folder and then look for the sequences through seqinfo.ini.
            zf.extractall(dest)


# ----------------------------------------------------------------------------- parsing

def read_ini(path: Path):
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str  # keeps the case of the keys
    cp.read(path, encoding="utf-8")
    return cp


def ball_track_ids(gameinfo_path: Path):
    """Returns the IDs of the tracklets labelled as 'ball' in gameinfo.ini."""
    cp = read_ini(gameinfo_path)
    ids = set()
    for section in cp.sections():
        for key, value in cp.items(section):
            if not key.lower().startswith("trackletid_"):
                continue
            label = value.split(";")[0].strip().lower()
            if label == "ball":
                ids.add(int(key.split("_", 1)[1]))
    return ids


def game_id(gameinfo_path: Path):
    """Returns the gameID of the sequence (None if absent)."""
    value = read_ini(gameinfo_path).get("Sequence", "gameID", fallback=None)
    return int(value) if value is not None and value.strip().isdigit() else None


def read_ball_boxes(gt_path: Path, ball_ids):
    """frame -> list of boxes (x, y, w, h) in pixels for the ball tracklets only."""
    boxes = defaultdict(list)
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame, track_id = int(float(parts[0])), int(float(parts[1]))
            if track_id not in ball_ids:
                continue
            x, y, w, h = map(float, parts[2:6])
            boxes[frame].append((x, y, w, h))
    return boxes


def to_yolo(box, img_w, img_h):
    """MOT box (x,y,w,h in pixels, top-left corner) -> normalised YOLO, with clipping."""
    x, y, w, h = box
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
    return cx, cy, (x2 - x1) / img_w, (y2 - y1) / img_h


# ----------------------------------------------------------------------------- conversion

def place_image(src: Path, dst: Path, mode: str):
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different disks or unsupported FS: fall back to copying
    shutil.copy2(src, dst)


def write_labels(path: Path, labels):
    with open(path, "w", encoding="utf-8") as f:
        for cx, cy, w, h in labels:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def write_tiles(src: Path, name: str, frame_boxes, img_w, img_h, img_out: Path, lbl_out: Path,
                yolo_split: str, args, rng, stats):
    """Cuts the frame into square tiles (same grid as detect_ball.py) and saves them.

    - tiles with the ball: always kept;
    - empty tiles: kept with probability --tile-empty-ratio (negative examples);
    - tiles in which the ball is cut by the edge and less than --tile-min-visible of it remains
      visible: discarded, because they are neither a clean positive nor a negative
      (with 20% overlap the same ball is whole in a neighbouring tile anyway).
    """
    import cv2
    from utils.pitch import tile_grid

    img = None  # read only if at least one tile is needed
    t = args.tile
    for tx, ty in tile_grid(img_h, img_w, t, args.tile_overlap):
        tw, th = min(t, img_w - tx), min(t, img_h - ty)
        labels, ambiguous = [], False
        for x, y, w, h in frame_boxes:
            # box clipped to the image first, then to the tile
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                continue
            cx1, cy1 = max(x1, tx), max(y1, ty)
            cx2, cy2 = min(x2, tx + tw), min(y2, ty + th)
            if cx2 <= cx1 or cy2 <= cy1:
                continue  # ball outside the tile
            visible = (cx2 - cx1) * (cy2 - cy1) / area
            if min(w, h) < args.min_box_px or visible < args.tile_min_visible:
                ambiguous = True
                continue
            labels.append((((cx1 + cx2) / 2 - tx) / tw, ((cy1 + cy2) / 2 - ty) / th,
                           (cx2 - cx1) / tw, (cy2 - cy1) / th))

        if not labels and (ambiguous or rng.random() >= args.tile_empty_ratio):
            continue
        if img is None:
            img = cv2.imread(str(src))
            if img is None:
                print(f"[WARNING] cannot read {src}")
                return
        tile_name = f"{name}_{tx:04d}_{ty:04d}"
        cv2.imwrite(str(img_out / f"{tile_name}.jpg"), img[ty:ty + th, tx:tx + tw],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_labels(lbl_out / f"{tile_name}.txt", labels)
        stats[f"{yolo_split}_images"] += 1
        stats[f"{yolo_split}_ball"] += bool(labels)


def convert_sequence(seq_dir: Path, out_dir: Path, yolo_split: str, args, rng, stats):
    seqinfo_path, gameinfo_path = seq_dir / "seqinfo.ini", seq_dir / "gameinfo.ini"
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        stats["no_gt"] += 1
        return
    if not gameinfo_path.exists():
        print(f"[WARNING] {seq_dir.name}: gameinfo.ini missing, skipping.")
        return

    seq = read_ini(seqinfo_path)["Sequence"]
    img_w, img_h = int(seq["imWidth"]), int(seq["imHeight"])
    img_dir = seq_dir / seq.get("imDir", "img1")
    img_ext = seq.get("imExt", ".jpg")
    seq_len = int(seq.get("seqLength", 0)) or len(list(img_dir.glob(f"*{img_ext}")))

    ball_ids = ball_track_ids(gameinfo_path)
    if not ball_ids:
        print(f"[WARNING] {seq_dir.name}: no 'ball' tracklet in gameinfo.ini, skipping.")
        stats["no_ball_track"] += 1
        return
    boxes = read_ball_boxes(gt_path, ball_ids)

    img_out = out_dir / "images" / yolo_split
    lbl_out = out_dir / "labels" / yolo_split

    for frame in range(1, seq_len + 1, args.frame_step):
        src = img_dir / f"{frame:06d}{img_ext}"
        if not src.exists():
            continue

        labels = []
        for b in boxes.get(frame, []):
            if min(b[2], b[3]) < args.min_box_px:
                continue
            yb = to_yolo(b, img_w, img_h)
            if yb:
                labels.append(yb)

        # Frames without the ball: only a fraction is kept as negatives
        if not labels and rng.random() >= args.empty_ratio:
            continue

        name = f"{seq_dir.name}_{frame:06d}"
        stats["box_sizes"].extend(min(b[2], b[3]) for b in boxes.get(frame, []))
        if args.tile:
            write_tiles(src, name, boxes.get(frame, []), img_w, img_h, img_out, lbl_out,
                        yolo_split, args, rng, stats)
            continue

        place_image(src, img_out / f"{name}{img_ext}", args.copy_mode)
        write_labels(lbl_out / f"{name}.txt", labels)
        stats[f"{yolo_split}_images"] += 1
        stats[f"{yolo_split}_ball"] += bool(labels)

    stats[f"{yolo_split}_sequences"] += 1
    stats[f"{yolo_split}_games"].add(game_id(gameinfo_path))


def write_data_yaml(out_dir: Path):
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: ball\n",
        encoding="utf-8",
    )
    return yaml_path


def main():
    p = argparse.ArgumentParser(description="SoccerNet-Tracking -> YOLO dataset (ball only).")
    p.add_argument("--root", type=Path, default=Path("data/soccernet_raw"), help="SoccerNet download folder")
    p.add_argument("--out", type=Path, default=None,
                   help="YOLO dataset folder (default: data/yolo/soccernet_ball, or "
                        "data/yolo/soccernet_ball_tiles<N> with --tile N)")
    p.add_argument("--splits", nargs="+", default=["train", "test"], choices=list(SPLIT_MAP))
    p.add_argument("--skip-download", action="store_true", help="Uses the zips/folders already present")
    p.add_argument("--frame-step", type=int, default=5,
                   help="Takes 1 frame every N (consecutive frames at 25 fps are almost identical)")
    p.add_argument("--empty-ratio", type=float, default=0.15,
                   help="Fraction of frames WITHOUT the ball to keep as negative examples (0-1)")
    p.add_argument("--min-box-px", type=float, default=2.0, help="Discards ball boxes smaller than this")
    p.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink",
                   help="hardlink saves space if root and out are on the same disk")
    p.add_argument("--val-games", type=int, nargs="+", default=None, metavar="ID",
                   help="gameIDs to put in val; all the other games (of any SoccerNet "
                        "split) go to train. Default: official split train->train, test->val")
    p.add_argument("--clean", action="store_true",
                   help="Empties the output's images/ and labels/ before converting (recommended "
                        "when the split changes, otherwise files stay in the old split)")
    p.add_argument("--tile", type=int, default=0, metavar="N",
                   help="0 = full frames (default). N = cuts every frame into NxN tiles with the "
                        "same grid as detect_ball.py; the tiles are new JPEGs, not hardlinks")
    p.add_argument("--tile-overlap", type=float, default=0.2, help="Overlap between tiles (0-1)")
    p.add_argument("--tile-empty-ratio", type=float, default=0.1,
                   help="Fraction of tiles WITHOUT the ball to keep as negatives (0-1)")
    p.add_argument("--tile-min-visible", type=float, default=0.5,
                   help="A ball cut by the tile edge stays labelled if at least this fraction "
                        "of it is visible; otherwise the tile is discarded")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.frame_step < 1 or not 0 <= args.empty_ratio <= 1:
        p.error("--frame-step must be >= 1 and --empty-ratio between 0 and 1")
    if args.tile < 0 or not 0 <= args.tile_overlap < 1 or not 0 <= args.tile_empty_ratio <= 1:
        p.error("--tile must be >= 0, --tile-overlap in [0, 1), --tile-empty-ratio in [0, 1]")
    if args.out is None:
        args.out = Path(f"data/yolo/soccernet_ball_tiles{args.tile}" if args.tile else "data/yolo/soccernet_ball")
    print(f"[INFO] Output: {args.out}" + (f" (tiles {args.tile}x{args.tile})" if args.tile else ""))

    task_dir = args.root / TASK
    if not args.skip_download:
        download(args.root, args.splits)
    extract_zips(task_dir, args.splits)

    if args.clean:
        for sub in ("images", "labels"):
            if (args.out / sub).exists():
                print(f"[INFO] Removing {args.out / sub}")
                shutil.rmtree(args.out / sub)
    elif args.val_games and any((args.out / "images").glob("*/*")):
        p.error(f"{args.out / 'images'} is not empty: with --val-games add --clean, "
                "otherwise the sequences that change split would also stay in the old one")

    for sub in ("images", "labels"):
        for split in SPLIT_MAP.values():
            (args.out / sub / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    stats = defaultdict(int)
    stats["box_sizes"] = []
    for split in SPLIT_MAP.values():
        stats[f"{split}_games"] = set()

    val_games = set(args.val_games or [])
    found_games = set()
    for sn_split in args.splits:
        split_dir = task_dir / sn_split
        seq_dirs = sorted(pth.parent for pth in split_dir.rglob("seqinfo.ini"))
        if not seq_dirs:
            print(f"[WARNING] No sequences found in {split_dir}")
            continue
        if val_games:
            print(f"[INFO] {sn_split}: {len(seq_dirs)} sequences -> split by gameID (val = {sorted(val_games)})")
        else:
            print(f"[INFO] {sn_split}: {len(seq_dirs)} sequences -> YOLO split '{SPLIT_MAP[sn_split]}'")
        for seq_dir in seq_dirs:
            if val_games:
                gid = game_id(seq_dir / "gameinfo.ini") if (seq_dir / "gameinfo.ini").exists() else None
                found_games.add(gid)
                yolo_split = "val" if gid in val_games else "train"
            else:
                yolo_split = SPLIT_MAP[sn_split]
            convert_sequence(seq_dir, args.out, yolo_split, args, rng, stats)

    missing = val_games - found_games
    if missing:
        print(f"[WARNING] gameIDs requested for val but not found: {sorted(missing)} "
              f"(available: {sorted(g for g in found_games if g is not None)})")

    yaml_path = write_data_yaml(args.out)

    print("\n========== SUMMARY ==========")
    for split in SPLIT_MAP.values():
        n, nb = stats[f"{split}_images"], stats[f"{split}_ball"]
        games = sorted(g for g in stats[f"{split}_games"] if g is not None)
        print(f"{split:5s}: {stats[f'{split}_sequences']} sequences, {n} images, {nb} with ball, "
              f"{n - nb} negatives, games {games}")
    if stats["no_gt"]:
        print(f"Sequences without gt/gt.txt (skipped): {stats['no_gt']}")
    if stats["no_ball_track"]:
        print(f"Sequences without a 'ball' tracklet (skipped): {stats['no_ball_track']}")
    sizes = sorted(stats["box_sizes"])
    if sizes:
        med = sizes[len(sizes) // 2]
        print(f"Shorter side of the ball box (px): min {sizes[0]:.0f}, median {med:.0f}, max {sizes[-1]:.0f}")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
