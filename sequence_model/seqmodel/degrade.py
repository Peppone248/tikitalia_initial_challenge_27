"""Degrade clean tracking so it looks like the provided match's automatic tracking: ball missing
mostly in dead ball, gaps in live play, broken player tracks. Returns a copy; deterministic per seed."""
from dataclasses import replace

import numpy as np

from .match import CFG, D


def _draw_gap_s(rng, quantiles):
    """Piecewise-linear inverse CDF: pick a segment between consecutive quantiles uniformly,
    then a uniform length inside it."""
    i = rng.integers(0, len(quantiles) - 1)
    return rng.uniform(quantiles[i], quantiles[i + 1])


def _hide_live_gaps(rng, hidden, live, target, quantiles, fps):
    """Cut gaps at random live positions until the share of live frames seen is <= target;
    the last gap is clipped so the share lands on the target."""
    n_live = int(live.sum())
    if n_live == 0:
        return
    live_idx = np.flatnonzero(live)
    T = len(hidden)
    for _ in range(100000):
        need = int((~hidden & live).sum()) - int(round(target * n_live))
        if need <= 0:
            return
        length = min(max(1, int(round(_draw_gap_s(rng, quantiles) * fps))), need)
        start = int(rng.choice(live_idx))
        hidden[start:min(T, start + length)] = True


def degrade(m, seed, cfg):
    rng = np.random.default_rng(seed)
    fps = CFG["fps"]
    T = len(m.label)
    hidden = np.zeros(T, dtype=bool)

    # dead ball: each maximal D run is hidden entirely with probability dead_hide_p
    is_d = m.label == D
    change = np.flatnonzero(np.diff(is_d.astype(np.int8))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [T]])
    for s, e in zip(starts, ends):
        if is_d[s] and rng.random() < cfg["dead_hide_p"]:
            hidden[s:e] = True

    # live ball: gaps until the seen share of live frames reaches the target
    live = ~is_d
    _hide_live_gaps(rng, hidden, live, cfg["live_seen_target"], cfg["live_gap_s"], fps)

    ball_x, ball_y = m.ball_x.astype(float).copy(), m.ball_y.astype(float).copy()
    ball_x[hidden] = np.nan
    ball_y[hidden] = np.nan

    # players: short random stretches of lost tracking
    px, py = m.px.copy(), m.py.copy()
    for j in range(px.shape[1]):
        orig_nan = np.isnan(px[:, j]) | np.isnan(py[:, j])
        drop = np.zeros(T, dtype=bool)
        target = int(round(cfg["player_drop_frac"] * (~orig_nan).sum()))
        for _ in range(100000):
            need = target - int((drop & ~orig_nan).sum())
            if need <= 0:
                break
            length = min(int(rng.integers(fps, 10 * fps + 1)), need)
            start = int(rng.integers(0, T))
            drop[start:min(T, start + length)] = True
        px[drop, j] = np.nan
        py[drop, j] = np.nan

    return replace(m, px=px, py=py, ball_x=ball_x, ball_y=ball_y, label=m.label.copy(),
                   team=m.team.copy(), period=m.period.copy(),
                   home_attacks_right=dict(m.home_attacks_right))
