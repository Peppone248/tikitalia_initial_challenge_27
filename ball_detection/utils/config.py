"""
Pipeline configuration: default paths, models, weights and the location of each step's CSV files.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Match inputs: tactical video, per-frame homography and ATD.
# Same rule as games.py in the repository root: the MTAH_DATA_DIR environment variable if set,
# otherwise the repository's data/ folder.
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR") or ROOT.parent / "data")
CHALLENGE_DATA = DATA_DIR / "tactical"
DEFAULT_VIDEO = CHALLENGE_DATA / "BAR-ATH_tactical.mp4"
DEFAULT_HOMOGRAPHY = CHALLENGE_DATA / "BAR-ATH_tactical_homography.csv"

# Model weights (origin and licences in weights/README.md)
WEIGHTS_DIR = ROOT / "weights"
TIKITALIA_WEIGHTS = WEIGHTS_DIR / "tikitalia_model.pt"        # YOLOv8s-P2, full frames at 1280 px
TIKITALIA_TILES_WEIGHTS = WEIGHTS_DIR / "tikitalia_tiles.pt"  # YOLOv8m-P2, 640 tiles
TIKITALIA_TILES_SIZE = 640
ROBOFLOW_WEIGHTS = WEIGHTS_DIR / "roboflow_ball.pt"           # roboflow/sports, 640 tiles

# Fixed order: it also decides the name of the ensemble folders (ensemble_<a>+<b>)
ALL_MODELS = ["tikitalia_model", "tikitalia_tiles", "roboflow"]
# Models used when a script receives no --model/--models (the reference pipeline)
DEFAULT_MODELS = ["tikitalia_model", "roboflow"]

# Pipeline output: runs/ball_pipeline/<video name>/
OUT_ROOT = ROOT / "runs" / "ball_pipeline"
TRAIN_RUNS = ROOT / "runs" / "train"


def video_name(video):
    """Video name without extension, e.g. 'BAR-ATH_tactical'."""
    return Path(video).name.rsplit(".mp4", 1)[0]


OUT_DIR = OUT_ROOT / video_name(DEFAULT_VIDEO)


def set_video(video):
    """Sets the output folder for the given video. Every script calls it after reading --video."""
    global OUT_DIR
    OUT_DIR = OUT_ROOT / video_name(video)
    return OUT_DIR


def out_dir():
    return OUT_DIR


def detections_path(model):
    """CSV with the detections of a single model (inference/detect_ball.py)."""
    return OUT_DIR / model / "detections.csv"


def track_path(model):
    """CSV with the post-processed track of a single model (inference/post_process.py)."""
    return OUT_DIR / model / "track.csv"


def ensemble_name(models):
    return "ensemble_" + "+".join(m for m in ALL_MODELS if m in models)


def ball_path(models):
    """CSV with the final ball of an ensemble (inference/ball_consensus.py)."""
    return OUT_DIR / ensemble_name(models) / "ball.csv"
