"""
Shared geometry and physics:
  - tile_grid: tile grid (the same in training and inference)
  - Homographies: per-frame homography, pixels <-> metres on the pitch, "inside the pitch" test
  - add_common_params / fill_gaps: common physical parameters and gap filling of a trajectory
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd

from utils.config import DEFAULT_VIDEO, DEFAULT_HOMOGRAPHY

# Pitch size in metres (the challenge's games.py: 105.3 x 68.0)
PITCH_M = (105.3, 68.0)


def tile_grid(h, w, tile=640, overlap=0.2):
    """Top-left corners (x, y) of the square tiles covering an h x w image.
    Step = tile * (1 - overlap); the last tile of each row/column is aligned to the edge.
    Used both in inference (detect_ball.py) and to build the dataset (prepare_soccernet.py)."""
    step = int(tile * (1 - overlap))

    def starts(size):
        s = list(range(0, max(size - tile, 0) + 1, step))
        if s[-1] + tile < size:
            s.append(size - tile)
        return s

    return [(x, y) for y in starts(h) for x in starts(w)]


class Homographies:
    """Per-frame homographies of the video. The matrix maps normalised pitch
    coordinates (0-1) to pixels; here we work in metres on the pitch."""

    def __init__(self, csv_path=DEFAULT_HOMOGRAPHY):
        df = pd.read_csv(csv_path)
        cols = [f"h{r}{c}" for r in range(3) for c in range(3)]
        n = int(df["frame"].max()) + 1
        self.H = np.full((n, 3, 3), np.nan)
        self.Hinv = np.full((n, 3, 3), np.nan)
        valid = df["valid"].to_numpy() == 1
        frames = df["frame"].to_numpy()[valid]
        mats = df.loc[valid, cols].to_numpy().reshape(-1, 3, 3)
        self.H[frames] = mats
        self.Hinv[frames] = np.linalg.inv(mats)
        self.valid = np.zeros(n, dtype=bool)
        self.valid[frames] = True

    def has(self, frame):
        return 0 <= frame < len(self.valid) and self.valid[frame]

    def to_pitch(self, frame, x, y):
        """Pixels -> metres on the pitch (ground plane)."""
        p = self.Hinv[frame] @ np.array([x, y, 1.0])
        return np.array([p[0] / p[2] * PITCH_M[0], p[1] / p[2] * PITCH_M[1]])

    def to_image(self, frame, m):
        """Metres on the pitch -> pixels."""
        p = self.H[frame] @ np.array([m[0] / PITCH_M[0], m[1] / PITCH_M[1], 1.0])
        return p[0] / p[2], p[1] / p[2]

    def on_pitch(self, frame, x, y, margin_m=3.0):
        """True if the point falls inside the pitch (with a margin in metres)."""
        u, v = self.to_pitch(frame, x, y)
        return -margin_m <= u <= PITCH_M[0] + margin_m and -margin_m <= v <= PITCH_M[1] + margin_m


# ----------------------------------------------------------------------------- common physical parameters

def add_common_params(p):
    """Parameters shared by post_process.py, ball_consensus.py and compare_models.py.
    Chosen from physical considerations."""
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--homography", type=Path, default=DEFAULT_HOMOGRAPHY)
    p.add_argument("--fps", type=float, default=25)
    p.add_argument("--max-speed", type=float, default=40, help="m/s, physical limit of the ball")
    p.add_argument("--noise-m", type=float, default=0.5,
                   help="m, tolerated position error (detection + homography)")
    p.add_argument("--max-gap-s", type=float, default=0.5, help="gap always filled (spline)")
    p.add_argument("--long-gap-s", type=float, default=2.0,
                   help="gap filled only if the ball is slow (straight line)")
    p.add_argument("--slow-speed", type=float, default=6.0,
                   help="m/s, mean speed below which the ball is 'slow' (at a player's feet, rolling gently)")
    p.add_argument("--vel-frames", type=int, default=3, help="frames used to estimate the velocity at the gap edges")
    p.add_argument("--recover-px", type=float, default=25,
                   help="px, a detection this close to the estimated trajectory replaces it")


def hermite(p0, v0, p1, v1, n, t):
    """Cubic Hermite spline between p0 (t=0) and p1 (t=n); v0, v1 in metres/frame."""
    s = t / n
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * p0 + h10 * n * v0 + h01 * p1 + h11 * n * v1


def edge_velocity(known, f, direction, k_max):
    """Velocity (m/frame) at the edge of a gap, from the consecutive known points before (-1) or after (+1) f."""
    for k in range(k_max, 0, -1):
        if all(f + direction * i in known for i in range(1, k + 1)):
            g = f + direction * k
            return (known[f] - known[g]) / k if direction < 0 else (known[g] - known[f]) / k
    return None


def fill_gaps(known_px, hom, args, candidates=None):
    """Fills the gaps between known positions.

    known_px:   frame -> (x, y) in pixels, the known positions
    candidates: frame -> [(source, x, y)] unused detections; if one falls within --recover-px of the
                estimated trajectory it is used ('recovered') instead of the synthetic point
    Rules:
      gap <= --max-gap-s      filled if the mean speed is <= --max-speed. Hermite spline with the
                              velocity before and after the gap; straight line if the direction reverses
      gap <= --long-gap-s     filled with a straight line only if the mean speed is <= --slow-speed:
                              ball at a player's feet or rolling gently, the line is a good estimate
    The trajectory is computed in metres on the pitch and reprojected with each frame's homography.
    Returns frame -> (x, y, 'recovered'|'interpolated', source|'spline'|'linear').
    """
    max_step, slow_step = args.max_speed / args.fps, args.slow_speed / args.fps
    short, long_ = round(args.max_gap_s * args.fps), round(args.long_gap_s * args.fps)
    known = {f: hom.to_pitch(f, *p) for f, p in known_px.items() if hom.has(f)}
    anchors = sorted(known)
    filled = {}
    for a, b in zip(anchors, anchors[1:]):
        n = b - a
        if n <= 1:
            continue
        pa, pb = known[a], known[b]
        chord = (pb - pa) / n
        speed = np.linalg.norm(chord)
        if n - 1 <= short and speed <= max_step:
            va = edge_velocity(known, a, -1, args.vel_frames)
            vb = edge_velocity(known, b, +1, args.vel_frames)
            va = chord if va is None else va
            vb = chord if vb is None else vb
            curved = speed > 1e-6 and all(np.dot(v, chord) > 0 and np.linalg.norm(v) <= max_step
                                          for v in (va, vb))
        elif n - 1 <= long_ and speed <= slow_step:
            curved = False
        else:
            continue
        for f in range(a + 1, b):
            if not hom.has(f):
                continue
            t = f - a
            m = hermite(pa, va, pb, vb, n, t) if curved else pa + chord * t
            xi, yi = hom.to_image(f, m)
            near = [c for c in (candidates or {}).get(f, []) if math.hypot(c[1] - xi, c[2] - yi) <= args.recover_px]
            if near:
                src, x, y = min(near, key=lambda c: math.hypot(c[1] - xi, c[2] - yi))[:3]
                filled[f] = (x, y, "recovered", src)
            else:
                filled[f] = (xi, yi, "interpolated", "spline" if curved else "linear")
    return filled
