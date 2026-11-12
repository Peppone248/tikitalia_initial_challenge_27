"""Common match format, controlled labels and the IDSSE loader. Coordinates in metres, origin at
the centre of the pitch. Label codes as in evaluation/possession_metrics.label_codes."""
import os
from dataclasses import dataclass

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "config.yaml")) as f:
    CFG = yaml.safe_load(f)
H, A, D, X = (CFG["labels"][k] for k in "HADX")


@dataclass
class Match:
    name: str
    px: np.ndarray            # (T, P) float32
    py: np.ndarray            # (T, P)
    team: np.ndarray          # (P,) int8, 0 = H, 1 = A
    ball_x: np.ndarray        # (T,)
    ball_y: np.ndarray        # (T,)
    period: np.ndarray        # (T,) int8
    label: np.ndarray         # (T,) int8
    length: float
    width: float
    home_attacks_right: dict  # {period: bool}, True when home attacks towards +x


def controlled_labels(label, min_frames=None):
    """A team run shorter than min_frames between two runs of the other team takes the other
    team's label. D is never touched (and bounds the runs)."""
    min_frames = CFG["min_run_frames"] if min_frames is None else min_frames
    out = np.array(label, copy=True)
    change = np.flatnonzero(np.diff(out)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(out)]])
    vals = out[starts]
    for i in range(1, len(starts) - 1):
        if vals[i] in (H, A) and vals[i - 1] in (H, A) and vals[i - 1] == vals[i + 1] \
                and vals[i - 1] != vals[i] and ends[i] - starts[i] < min_frames:
            out[starts[i]:ends[i]] = vals[i - 1]
    return out


def _deep_sides(x, per, team_name):
    """Median over the frames with a tracked player of the per-frame min and max x."""
    ok = np.isfinite(x).any(axis=1)
    if not ok.any():
        raise ValueError(f"no tracked {team_name} player in period {per}")
    x = x[ok]
    lo = np.where(np.isfinite(x), x, np.inf).min(axis=1)
    hi = np.where(np.isfinite(x), x, -np.inf).max(axis=1)
    return float(np.median(lo)), float(np.median(hi))


def attack_direction(m):
    """Per period: each team's deepest player (median over the period of the per-frame min/max
    x) is its goalkeeper. Home attacks right (+x) if its goalkeeper side is at negative x.
    Frames without a tracked player of a team are ignored; a team never tracked in a whole
    period raises ValueError."""
    home, away = m.team == 0, m.team == 1
    result = {}
    for per in np.unique(m.period):
        sel = m.period == per
        h_neg, h_pos = _deep_sides(m.px[sel][:, home], per, "home")
        a_neg, a_pos = _deep_sides(m.px[sel][:, away], per, "away")
        # keeper depth = distance from the centre of the deepest player on that side;
        # compare "home keeper left, away keeper right" with the mirrored hypothesis
        result[int(per)] = bool(abs(h_neg) + abs(a_pos) >= abs(h_pos) + abs(a_neg))
    return result


def load_idsse(path):
    with np.load(path) as z:
        poss = z["ball_possession"]
        label = np.where(poss == 1, H, np.where(poss == 2, A, X)).astype(np.int8)
        label[~z["ball_alive"].astype(bool)] = D
        length = float(z["pitch_x"]) or CFG["default_length"]
        width = float(z["pitch_y"]) or CFG["default_width"]
        m = Match(name=os.path.splitext(os.path.basename(path))[0],
                  px=z["px"].astype(np.float32), py=z["py"].astype(np.float32),
                  team=z["player_team"].astype(np.int8), ball_x=z["ball_x"], ball_y=z["ball_y"],
                  period=z["period"].astype(np.int8), label=label, length=length, width=width,
                  home_attacks_right={})
    m.home_attacks_right = attack_direction(m)
    return m
