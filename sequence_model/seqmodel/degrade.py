"""Degrade clean tracking so it looks like the provided match's automatic tracking: ball missing
mostly in dead ball, lost in the keeper's hands, gaps in live play, broken player tracks. Returns a
copy; deterministic per seed."""
from dataclasses import replace

import numpy as np

from .keeper import deepest, in_own_box, runs, team_signs
from .match import CFG, D


def _draw_gap_s(rng, quantiles):
    """Piecewise-linear inverse CDF: pick a segment between consecutive quantiles uniformly,
    then a uniform length inside it."""
    i = rng.integers(0, len(quantiles) - 1)
    return rng.uniform(quantiles[i], quantiles[i + 1])


def _hide_live_gaps(rng, hidden, live, target, quantiles, fps, max_iter):
    """Cut gaps at random live positions until the share of live frames seen is <= target;
    the last gap is clipped so the share lands on the target."""
    n_live = int(live.sum())
    if n_live == 0:
        return
    live_idx = np.flatnonzero(live)
    T = len(hidden)
    for _ in range(max_iter):
        need = int((~hidden & live).sum()) - int(round(target * n_live))
        if need <= 0:
            return
        length = min(max(1, int(round(_draw_gap_s(rng, quantiles) * fps))), need)
        start = int(rng.choice(live_idx))
        hidden[start:min(T, start + length)] = True
    raise RuntimeError(f"live_seen_target {target} not reached in {max_iter} gaps")


def keeper_holds(m, cfg):
    """Keeper holds on CLEAN tracking: stretches of >= keeper_hold_s where the ball is higher than
    keeper_hold_z, within keeper_hold_r of a team's deepest player, and that player is inside his own
    penalty area. Returns [(start, end, team)) with team 0 = H, 1 = A; never across a period
    boundary; [] when the match has no ball height."""
    if m.ball_z is None:
        return []
    fps = CFG["fps"]
    bx, by = np.asarray(m.ball_x, float), np.asarray(m.ball_y, float)
    bz = np.asarray(m.ball_z, float)
    sg_h = team_signs(m)
    out = []
    for tid, sg in ((0, sg_h), (1, -sg_h)):
        c = m.team == tid
        kx, ky, _ = deepest(m.px[:, c].astype(float), m.py[:, c].astype(float), sg)
        with np.errstate(invalid="ignore"):
            hold = ((bz > cfg["keeper_hold_z"]) & (np.hypot(bx - kx, by - ky) <= cfg["keeper_hold_r"])
                    & in_own_box(kx, ky, sg, m.length))
        for per in np.unique(m.period):
            idx = np.flatnonzero(m.period == per)
            for s, e in runs(hold[idx]):
                if e - s >= round(cfg["keeper_hold_s"] * fps):
                    out.append((int(idx[s]), int(idx[e - 1]) + 1, tid))
    return sorted(out)


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

    # keeper holds: the ball is lost in the keeper's hands with probability keeper_hide_p. A separate
    # random stream, so keeper_hide_p = 0 reproduces the previous degradation exactly
    if cfg.get("keeper_hide_p", 0) > 0:
        rng_k = np.random.default_rng([seed, 1])
        for s, e, _ in keeper_holds(m, cfg):
            if rng_k.random() < cfg["keeper_hide_p"]:
                hidden[s:e] = True

    # live ball: gaps until the seen share of live frames reaches the target
    live = ~is_d
    _hide_live_gaps(rng, hidden, live, cfg["live_seen_target"], cfg["live_gap_s"], fps,
                    cfg["max_iter"])

    ball_x, ball_y = m.ball_x.astype(float).copy(), m.ball_y.astype(float).copy()
    ball_x[hidden] = np.nan
    ball_y[hidden] = np.nan

    # players: short random stretches of lost tracking
    px, py = m.px.copy(), m.py.copy()
    lo_s, hi_s = cfg["player_drop_s"]
    for j in range(px.shape[1]):
        orig_nan = np.isnan(px[:, j]) | np.isnan(py[:, j])
        drop = np.zeros(T, dtype=bool)
        target = int(round(cfg["player_drop_frac"] * (~orig_nan).sum()))
        for _ in range(cfg["max_iter"]):
            need = target - int((drop & ~orig_nan).sum())
            if need <= 0:
                break
            length = min(int(rng.integers(round(lo_s * fps), round(hi_s * fps) + 1)), need)
            start = int(rng.integers(0, T))
            drop[start:min(T, start + length)] = True
        else:
            raise RuntimeError(f"player_drop_frac {cfg['player_drop_frac']} not reached for "
                               f"player {j} in {cfg['max_iter']} stretches")
        px[drop, j] = np.nan
        py[drop, j] = np.nan

    ball_z = None
    if m.ball_z is not None:
        ball_z = np.asarray(m.ball_z, float).copy()
        ball_z[hidden] = np.nan
    return replace(m, px=px, py=py, ball_x=ball_x, ball_y=ball_y, ball_z=ball_z, label=m.label.copy(),
                   team=m.team.copy(), period=m.period.copy(),
                   home_attacks_right=dict(m.home_attacks_right))
