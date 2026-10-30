"""Approach B: the official baseline, boosted with almost no logic.

    1. TEAM     nearest tracked player to the ball, on the frames where the ball is seen, then a
                centred majority vote over +/- window frames (exactly starter/baseline.py and
                validation.baseline_labels).
    2. DEAD     our dead-ball mask written on top (the D of approach C: v1 HMM D, plus the Smart
                Tagging anchor in v3). The baseline's own "D where nobody votes" is dropped:
                a frame that is not dead keeps the last team that had a vote (CHALLENGE.md: a
                loose ball stays with the last team), the first spell of a stretch takes the next.
    3. MIN      possessions shorter than min_poss_s inside an alive stretch are given to the team
                before them (to the next one when they open the stretch), shortest first, so a
                flicker of the vote cannot split a possession.

No model, no state machine: three arrays and two passes. Everything else (which D, window,
min_poss_s) is a parameter, chosen on the IDSSE validation only.
"""

import numpy as np

from tikitalia.preprocessing import runs

HOME, AWAY = 0, 1


def nearest_votes(ball_x, ball_y, px, py, team_of_slot):
    """(n,) int8: 0 home / 1 away for the nearest tracked player where the ball is seen, -1 else."""
    dist = np.hypot(px - ball_x[:, None], py - ball_y[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    j = np.argmin(dist, axis=1)
    seen = ~np.isnan(ball_x) & np.isfinite(dist[np.arange(dist.shape[0]), j])
    codes = np.full(ball_x.size, -1, dtype=np.int8)
    codes[seen] = np.asarray(team_of_slot)[j[seen]]
    return codes


def majority(codes, window):
    """Centred majority over +/- window frames: 'H', 'A' or '' (no vote in the window)."""
    n = codes.size
    hc = np.concatenate(([0], np.cumsum(codes == HOME)))
    ac = np.concatenate(([0], np.cumsum(codes == AWAY)))
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(n, "", dtype="<U1")
    ok = (h + a) > 0
    out[ok] = np.where(h[ok] >= a[ok], "H", "A")
    return out


def _fill_alive(team, alive):
    """Inside every alive stretch: frames without a vote take the last team seen (forward), the
    leading ones the first team of the stretch (backward). A stretch with no vote at all: ''."""
    out = team.copy()
    for s, e in runs(alive):
        seg = out[s:e]
        has = np.flatnonzero(seg != "")
        if not has.size:
            continue
        idx = np.where(seg != "", np.arange(seg.size), -1)
        idx = np.maximum.accumulate(idx)
        idx[idx < 0] = has[0]
        out[s:e] = seg[idx]
    return out


def _min_possession(lab, alive, k):
    """Spells of H/A shorter than k frames inside an alive stretch go to the neighbour team
    (the one before; the one after when the spell opens the stretch), shortest first."""
    if k <= 1:
        return lab
    out = lab.copy()
    for s, e in runs(alive):
        seg = out[s:e]
        while True:
            b = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
            ends = np.r_[b[1:], seg.size]
            if b.size <= 1:
                break
            lens = ends - b
            i = int(np.argmin(lens))
            if lens[i] >= k:
                break
            seg[b[i]:ends[i]] = seg[b[i] - 1] if i > 0 else seg[ends[i]]
        out[s:e] = seg
    return out


def moving_average(x, size):
    """Centred moving average over `size` samples, edges replicated (same as
    scipy.ndimage.uniform_filter1d(mode="nearest"); numpy only, so the repository needs no scipy)."""
    x = np.asarray(x, float)
    size = int(size)
    if size <= 1:
        return x.copy()
    left = size // 2
    pad = np.pad(x, (left, size - 1 - left), mode="edge")
    c = np.concatenate(([0.0], np.cumsum(pad)))
    return (c[size:] - c[:-size]) / size


def spread_team(px, py, team_of_slot, smooth_frames=75):
    """'H'/'A' per frame: the team whose players are more spread out (std of x + std of y).
    The team in possession opens up, the defending team closes down: on IDSSE (degraded, 7
    matches) this alone names the team of 74% of the alive frames, 71% with the ball unseen."""
    t = np.asarray(team_of_slot)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sp = [np.nanstd(px[:, t == k], 1) + np.nanstd(py[:, t == k], 1) for k in (HOME, AWAY)]
    diff = np.nan_to_num(sp[0] - sp[1])
    if smooth_frames > 1:
        diff = moving_average(diff, smooth_frames)
    return np.where(diff >= 0, "H", "A"), diff


def mean_player_speed(px, py, fps=25, smooth_frames=50):
    """Mean speed (m/s) of the tracked players, smoothed over smooth_frames; 5 m/s where nobody is
    tracked (so a missing frame never looks like a stoppage)."""
    import warnings
    v = np.hypot(np.gradient(px, axis=0), np.gradient(py, axis=0)) * fps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(v, axis=1)
    return moving_average(np.nan_to_num(m, nan=5.0), smooth_frames)


def slow_dead(ball_x, px, py, half, cfg, fps=25):
    """D-SLOW (point 1: stoppages for fouls/offsides, which the Smart Tagging does not contain).
    Everybody walking = the ball is dead: on IDSSE a mean player speed under 1.0-1.25 m/s is a
    dead ball 82-100% of the time, and 84% of the dead frames our D misses have the ball unseen.
    A span is kept if the mean speed stays under v_max_ms for >= min_s (gaps < close_s joined)
    AND the ball is unseen on >= unseen_share of it: where the ball is seen the D rules already
    see the stoppage (D recall 94%), and on clean tracking this rule must change nothing."""
    from .segment_c import _clean_dead
    g = half > 0
    slow = (mean_player_speed(px, py, fps, cfg.get("smooth_frames", 50)) < cfg["v_max_ms"]) & g
    slow = _clean_dead(slow, half, fps, {"close_gap_s": cfg["close_s"],
                                         "min_dead_s": cfg["min_s"], "min_alive_s": 0})
    unseen = np.isnan(ball_x)
    for s, e in runs(slow):
        if unseen[s:e].mean() < cfg["unseen_share"]:
            slow[s:e] = False
    return slow


def boosted_baseline(ball_x, ball_y, px, py, team_of_slot, graded, dead, window=50,
                     min_poss_s=0.0, fps=25, fill_unseen=True, unseen="last", spread_smooth=75):
    """Labels H/A/D on graded frames, X elsewhere.
    unseen: what a live frame with no ball vote in its window gets -- 'last' (the last team
    voted, CHALLENGE.md's loose ball) or 'spread' (the more spread-out team)."""
    codes = nearest_votes(ball_x, ball_y, px, py, team_of_slot)
    team = majority(codes, int(window))
    if unseen == "spread":
        st, _ = spread_team(px, py, team_of_slot, spread_smooth)
        team = np.where(team == "", st, team)
    dead = np.asarray(dead, bool) & graded
    alive = graded & ~dead
    if fill_unseen:
        team = _fill_alive(team, alive)
    lab = np.full(ball_x.size, "X", dtype="<U1")
    lab[alive] = team[alive]
    lab[alive & (lab == "")] = "D"            # an alive stretch with no vote at all
    lab[dead] = "D"
    lab = _min_possession(lab, alive & (lab != "D"), int(round(min_poss_s * fps)))
    lab[~graded] = "X"
    return lab
