"""
Reading of the CSV files produced by the pipeline steps (detections.csv, .progress, track.csv).
"""
from pathlib import Path

import pandas as pd

from utils.config import detections_path, track_path


def read_progress(path):
    """.progress file of detect_ball.py: "first,last" completed frame."""
    parts = [int(v) for v in Path(path).read_text().split(",")]
    return (parts[0], parts[1]) if len(parts) == 2 else (0, parts[0])


def processed_range(model):
    """(first, last) frame processed by detect_ball.py for that model (.progress file)."""
    progress = detections_path(model).with_suffix(".progress")
    if not progress.exists():
        raise SystemExit(f"[ERROR] {progress} is missing: run "
                         f"python -m inference.detect_ball --model {model} first")
    return read_progress(progress)


def load_detections(models):
    """frame -> [(model, x, y, conf)] for the given models, and the frame range common to all of them."""
    by_frame = {}
    lo, hi = 0, None
    for m in models:
        df = pd.read_csv(detections_path(m))
        for f, x, y, c in zip(df["frame"].to_numpy(), df["x"].to_numpy(), df["y"].to_numpy(),
                              df["conf"].to_numpy()):
            by_frame.setdefault(int(f), []).append((m, float(x), float(y), float(c)))
        a, b = processed_range(m)
        lo, hi = max(lo, a), b if hi is None else min(hi, b)
    return by_frame, range(lo, hi + 1)


def load_track(model):
    """Post-processed track of a model, indexed by frame."""
    path = track_path(model)
    if not path.exists():
        raise SystemExit(f"[ERROR] {path} is missing: run "
                         f"python -m inference.post_process --model {model} first")
    return pd.read_csv(path).set_index("frame")
