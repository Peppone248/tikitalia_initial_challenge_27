"""Per-frame features for the sequence model. Team-level x quantities are computed in each team's
own attacking frame and reported as H - A (odd under team swap); ball_x_att is the ball x in H's
attacking frame. See config.yaml `features` for every constant."""
import dataclasses

import numpy as np

from .match import CFG

FC = CFG["features"]
FPS = CFG["fps"]
RADII = FC["near_radii"]

FEATURES = (["ball_seen", "ball_unseen_s", "ball_speed", "ball_line_dist", "ball_x_att",
             "nearest_dist"] + [f"n_within_{r}" for r in RADII] +
            ["centroid_x", "def_line", "front_line", "width", "length", "v_forward",
             "mean_speed", "mean_speed_all", "ball_out", "slow_unseen"])
_EVEN = {"ball_seen", "ball_unseen_s", "ball_speed", "ball_line_dist", "mean_speed_all",
         "ball_out", "slow_unseen"}
ODD = np.array([f not in _EVEN for f in FEATURES])


def swap_teams(m):
    """Home and away exchanged: team ids, H/A labels and attack directions."""
    lab = np.where(m.label == 0, 1, np.where(m.label == 1, 0, m.label)).astype(m.label.dtype)
    return dataclasses.replace(
        m, team=(1 - m.team).astype(m.team.dtype), label=lab,
        home_attacks_right={p: not v for p, v in m.home_attacks_right.items()})


def _smooth(x, w):
    """Centred moving average over axis 0, ignoring NaN (NaN only where the window is empty)."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    pad = [(w // 2 + 1, w // 2)] + [(0, 0)] * (x.ndim - 1)
    cs = np.cumsum(np.pad(v, pad), axis=0)
    cn = np.cumsum(np.pad(ok.astype(np.float64), pad), axis=0)
    T = len(x)
    s, n = cs[w:w + T] - cs[:T], cn[w:w + T] - cn[:T]
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _vel(x):
    """Frame difference x fps of smoothed positions (NaN where unknown, 0 on the first frame)."""
    s = _smooth(x, FC["smooth_frames"])
    v = np.zeros_like(s)
    v[1:] = np.diff(s, axis=0) * FPS
    return v


def _nanmean0(x):
    """Row mean over finite entries; 0 for rows with none."""
    ok = np.isfinite(x)
    n = ok.sum(axis=1)
    return np.where(n > 0, np.where(ok, x, 0.0).sum(axis=1) / np.maximum(n, 1), 0.0)


def _quantile(sorted_x, n, q):
    """Linear-interpolated quantile of rows sorted ascending with NaN last; n = valid count."""
    top = np.maximum(n - 1, 0)
    pos = q * top
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, top)
    r = np.arange(len(sorted_x))
    frac = pos - lo
    val = sorted_x[r, lo] * (1 - frac) + sorted_x[r, hi] * frac
    return np.where(n > 0, val, 0.0)


def _team_block(px, py, vx, vy, sgn, bx, by):
    """Team quantities in the team's own frame; neutral values when nobody is tracked."""
    ok = np.isfinite(px) & np.isfinite(py)
    n = ok.sum(axis=1)
    top = np.maximum(n - 1, 0)
    x = np.where(ok, px, np.nan) * sgn[:, None]
    y = np.where(ok, py, np.nan)
    xs, ys = np.sort(x, axis=1), np.sort(y, axis=1)       # NaN sorts last
    q_lo, q_hi = FC["shape_quantiles"]
    r = np.arange(len(px))
    out = {
        "centroid_x": _nanmean0(x),
        "def_line": np.where(n > 0, xs[r, np.minimum(1, top)], 0.0),
        "front_line": np.where(n > 0, xs[r, top], 0.0),
        "width": _quantile(ys, n, q_hi) - _quantile(ys, n, q_lo),
        "length": _quantile(xs, n, q_hi) - _quantile(xs, n, q_lo),
        "v_forward": _nanmean0(vx * sgn[:, None]),
        "mean_speed": _nanmean0(np.hypot(vx, vy)),
    }
    d = np.where(ok, np.hypot(px - bx[:, None], py - by[:, None]), np.inf)
    out["nearest_dist"] = np.where(n > 0, d.min(axis=1), FC["neutral_nearest"])
    for rad in RADII:
        out[f"n_within_{rad}"] = (d <= rad).sum(axis=1).astype(float)
    return out


def _period_features(m):
    T = len(m.period)
    ball_x, ball_y = np.asarray(m.ball_x, float), np.asarray(m.ball_y, float)
    seen = np.isfinite(ball_x) & np.isfinite(ball_y)
    last = np.maximum.accumulate(np.where(seen, np.arange(T), -1))
    src = np.maximum(last, 0)
    bx = np.where(last >= 0, np.nan_to_num(ball_x)[src], 0.0)    # last seen, centre spot before
    by = np.where(last >= 0, np.nan_to_num(ball_y)[src], 0.0)
    unseen_s = np.minimum((np.arange(T) - last) / FPS, FC["unseen_cap_s"])

    bvx = _vel(np.where(seen, ball_x, np.nan))
    bvy = _vel(np.where(seen, ball_y, np.nan))
    bsp = np.nan_to_num(np.hypot(bvx, bvy))
    ball_speed = np.where(seen & (bsp <= FC["max_ball_speed"]), bsp, 0.0)   # glitch -> neutral 0

    hl, hw = m.length / 2, m.width / 2
    line_dist = np.minimum(hl - np.abs(bx), hw - np.abs(by))
    raw = np.minimum(hl - np.abs(np.nan_to_num(ball_x)), hw - np.abs(np.nan_to_num(ball_y)))
    ball_out = seen & (raw < -FC["out_margin"])

    px, py = m.px.astype(np.float64), m.py.astype(np.float64)
    vx, vy = _vel(px), _vel(py)
    with np.errstate(invalid="ignore"):
        bad = np.hypot(vx, vy) > FC["max_player_speed"]       # impossible speed = unknown
    vx, vy = np.where(bad, np.nan, vx), np.where(bad, np.nan, vy)
    home_right = np.array([m.home_attacks_right[int(p)] for p in m.period], dtype=bool)
    sg_h = np.where(home_right, 1.0, -1.0)
    blocks = []
    for tid, sg in ((0, sg_h), (1, -sg_h)):
        c = m.team == tid
        blocks.append(_team_block(px[:, c], py[:, c], vx[:, c], vy[:, c], sg, bx, by))
    h, a = blocks
    mean_all = _nanmean0(np.hypot(vx, vy))

    cols = {
        "ball_seen": seen.astype(float), "ball_unseen_s": unseen_s, "ball_speed": ball_speed,
        "ball_line_dist": line_dist, "ball_x_att": bx * sg_h,
        "mean_speed_all": mean_all, "ball_out": ball_out.astype(float),
        "slow_unseen": ((mean_all < FC["slow_speed"]) & ~seen).astype(float),
    }
    for k in h:
        cols[k] = h[k] - a[k]
    return np.stack([cols[f] for f in FEATURES], axis=1).astype(np.float32)


def frame_features(m):
    """(T, len(FEATURES)) float32, always finite. Each period is an independent sequence:
    smoothing, velocities, last-seen ball and absence counter never cross half-time."""
    out = np.zeros((len(m.period), len(FEATURES)), dtype=np.float32)
    for per in np.unique(m.period):
        sel = m.period == per
        sub = dataclasses.replace(
            m, px=m.px[sel], py=m.py[sel], ball_x=np.asarray(m.ball_x)[sel],
            ball_y=np.asarray(m.ball_y)[sel], period=m.period[sel], label=m.label[sel])
        out[sel] = _period_features(sub)
    return out
