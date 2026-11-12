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
    graded_periods: tuple = (1, 2)  # periods whose frames are graded; the others are written as X


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


# Period codes of frames outside the two graded halves (before kick-off, half-time, after the
# final whistle): their own values so that decoding and the TCN never run across half-time.
_PRE, _BREAK, _POST = 0, 3, 4


def match_from_atd(ds, pitch, blocks, name="atd", min_mean_tracked=None, min_tracked_share=None):
    """kloppy TrackingDataset (coordinates normalised 0-1, metrica convention) -> Match in metres
    with origin at the centre, x = (x01 - 0.5) * length, y = (y01 - 0.5) * width (the same axis
    scaling as starter/baseline.py). Row i = video frame i = frame_id - 1; missing -> NaN; label all X.
    pitch = (length, width) in metres; blocks = [(half, first_video_frame, last_video_frame)]
    in play order: frames of half h get period h, the others _PRE / _BREAK / _POST."""
    min_mean_tracked = CFG["atd"]["min_mean_tracked"] if min_mean_tracked is None else min_mean_tracked
    min_tracked_share = CFG["atd"]["min_tracked_share"] if min_tracked_share is None else min_tracked_share
    length, width = float(pitch[0]), float(pitch[1])
    home_team = ds.metadata.teams[0]
    players = [p for t in ds.metadata.teams for p in t.players]      # one entry per track slot
    team = np.array([0 if t is home_team else 1 for t in ds.metadata.teams for _ in t.players],
                    dtype=np.int8)
    ids = np.array([int(r.frame_id) - 1 for r in ds.records])
    if ids.min() < 0:
        raise ValueError("ATD frame ids must start from 1")
    T = int(ids.max()) + 1
    px = np.full((T, len(players)), np.nan, dtype=np.float32)
    py = np.full_like(px, np.nan)
    bx, by = np.full(T, np.nan), np.full(T, np.nan)
    col = {p: i for i, p in enumerate(players)}
    for r, i in zip(ds.records, ids):
        for p, d in r.players_data.items():
            c = d.coordinates
            if c is not None and p in col:
                px[i, col[p]] = (c.x - 0.5) * length
                # y sign is irrelevant: every feature uses y only through |y|, differences or hypot
                py[i, col[p]] = (c.y - 0.5) * width
        if r.ball_coordinates is not None:
            bx[i] = (r.ball_coordinates.x - 0.5) * length
            by[i] = (r.ball_coordinates.y - 0.5) * width
    period = np.empty(T, dtype=np.int8)
    last = -1
    period[:] = _PRE
    for k, (half, a, b) in enumerate(blocks):
        period[a:b + 1] = half
        if k > 0:
            period[last + 1:a] = _BREAK
        last = b
    period[last + 1:] = _POST
    m = Match(name=name, px=px, py=py, team=team, ball_x=bx, ball_y=by, period=period,
              label=np.full(T, X, dtype=np.int8), length=length, width=width, home_attacks_right={},
              graded_periods=tuple(int(h) for h, _, _ in blocks))
    g = np.isin(period, m.graded_periods)
    n_tracked = np.isfinite(px[g]).sum(axis=1)
    if g.any() and (n_tracked.mean() < min_mean_tracked or (n_tracked > 0).mean() < min_tracked_share):
        raise ValueError(
            f"ATD conversion implausible: {n_tracked.mean():.1f} tracked players per graded frame on "
            f"average, {(n_tracked > 0).mean():.2f} of frames with any (need >= {min_mean_tracked} and "
            f">= {min_tracked_share}): player slots probably not matched to the records")
    sub = Match(name=name, px=px[g], py=py[g], team=team, ball_x=bx[g], ball_y=by[g], period=period[g],
                label=m.label[g], length=length, width=width, home_attacks_right={})
    m.home_attacks_right = {int(p): True for p in np.unique(period)}      # outside the halves: never used
    m.home_attacks_right.update(attack_direction(sub))
    return m


def load_atd(path_xml, path_txt):
    """The provided ATD (FIFA/EPTS xml + txt) as a Match, read through starter/load_atd.py (kloppy).
    Pitch size and graded halves come from games.py. The record count is checked against the video."""
    import sys
    root = os.path.dirname(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import games
    from starter import load_atd as starter
    from kloppy import metrica
    ds = metrica.load_tracking_epts(meta_data=path_xml, raw_data=path_txt)
    rep = starter.check_alignment(ds)
    if not rep["agrees"]:
        raise ValueError(f"ATD is not one record per video frame from 1: {rep}")
    return match_from_atd(ds, games.GAMES[games.GAME]["pitch"], games.half_video_blocks(games.GAME),
                          name=os.path.splitext(os.path.basename(path_xml))[0])
