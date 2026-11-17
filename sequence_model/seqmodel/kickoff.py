"""Kick-off detection and the "no play before the kick-off" rule: by the Laws of the Game a period
starts with the kick-off, so every graded frame of a period before it is dead ball (D). The
kick-off is recognised from the PLAYERS (Law 8), not the ball (the tracker does not see the ball
resting on the spot and at times follows another object); a ball seen at the centre is only
supporting evidence. Logic after the teammate's detector (src/tikitalia/events.py on
origin/approach-b-broadcast), reimplemented here. Constants: config.yaml `kickoff`.

Formation (per frame): enough tracked players per team; each team on one side of the halfway line
(players inside the centre circle do not count for the sides, a line tolerance counts for neither),
the two teams on opposite sides; at most max_intruders players of ONE team strictly inside the
circle. A kick-off is a formation run (runs separated by < 0.5 s merged) lasting >= setup_s in
which a player of exactly one team stands at the centre spot, and where the other team keeps at
most max_intruders players inside the circle (median over the run). The kick frame is the first
frame after the run (the formation breaks when play starts)."""
import sys

import numpy as np

from .features import _vel
from .match import CFG

KC = CFG["kickoff"]
FPS = CFG["fps"]
D = CFG["labels"]["D"]


def _runs(mask):
    """[(start, end)) of the True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _sides(x, counted, tol):
    left = ((x < -tol) & counted).sum(axis=1)
    right = ((x > tol) & counted).sum(axis=1)
    return left, right, np.where(left > right, -1, 1)


def find_kickoffs(m, lo, hi):
    """Kick-offs whose formation lies in rows [lo, hi): list of dicts {start, kick, kicker
    (0 home / 1 away), kicker_s, ball_seen}, in time order."""
    lo, hi = max(0, lo), min(len(m.px), hi)
    if hi <= lo:
        return []
    x, y = m.px[lo:hi].astype(np.float64), m.py[lo:hi].astype(np.float64)
    dist = np.where(np.isfinite(x) & np.isfinite(y), np.hypot(x, y), np.inf)
    vis = np.isfinite(x)
    in_circle = dist < KC["circle_radius"]
    strict = dist < KC["circle_radius"] - KC["circle_tol"]
    at_spot = dist <= KC["kicker_r"]
    home, away = m.team == 0, m.team == 1
    nh, na = vis[:, home].sum(axis=1), vis[:, away].sum(axis=1)
    lh, rh, sh = _sides(x[:, home], vis[:, home] & ~in_circle[:, home], KC["line_tol"])
    la, ra, sa = _sides(x[:, away], vis[:, away] & ~in_circle[:, away], KC["line_tol"])
    own = ((np.minimum(lh, rh) <= KC["max_wrong_side"]) & (np.minimum(la, ra) <= KC["max_wrong_side"])
           & (sh != sa))
    ih, ia = strict[:, home].sum(axis=1), strict[:, away].sum(axis=1)
    formation = ((nh >= KC["min_players"]) & (na >= KC["min_players"]) & own
                 & (np.minimum(ih, ia) <= KC["max_intruders"]))
    sp_h, sp_a = at_spot[:, home].any(axis=1), at_spot[:, away].any(axis=1)
    kicking = np.where(sp_h & ~sp_a, 0, np.where(sp_a & ~sp_h, 1, -1))

    merged = []
    for s, e in _runs(formation):
        if merged and s - merged[-1][1] <= int(round(0.5 * FPS)):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    ball_c = np.isfinite(m.ball_x[lo:hi]) & (np.hypot(m.ball_x[lo:hi], m.ball_y[lo:hi]) <= KC["centre_r"])
    out = []
    for s, e in merged:
        if e - s < int(round(KC["setup_s"] * FPS)):
            continue
        kickers = kicking[s:e][kicking[s:e] >= 0]
        if kickers.size == 0:
            continue
        kicker = int(np.bincount(kickers).argmax())
        other = (ia if kicker == 0 else ih)[s:e]
        if np.median(other) > KC["max_intruders"]:
            continue
        out.append(dict(start=lo + int(s), kick=lo + int(min(e, hi - 1)), kicker=kicker,
                        kicker_s=kickers.size / FPS,
                        ball_seen=bool(ball_c[s:min(hi - lo, e + int(FPS))].any())))
    return out


def live_share(m, a, b):
    """Share of the rows [a, b) where the mean speed of the tracked players exceeds run_speed: play
    in progress (running); the walk-on before a kick-off is walking."""
    if b - a < 2:
        return 0.0
    sp = np.hypot(_vel(m.px[a:b]), _vel(m.py[a:b]))
    sp[sp > CFG["features"]["max_player_speed"]] = np.nan
    ok = np.isfinite(sp).any(axis=1)
    if not ok.any():
        return 0.0
    ms = np.nanmean(sp[ok], axis=1)
    return float((ms > KC["run_speed"]).mean())


def detect_kickoff(m, period, info=False):
    """Row of the kick-off of `period`: the first kick-off (see find_kickoffs) whose kick frame lies
    inside the period or up to 1 s before it and within max_after_start_s of its start (a later one follows
    a goal). None if there is none. With info=True returns the dict (or None)."""
    rows = np.flatnonzero(m.period == period)
    if len(rows) == 0:
        return None
    r0 = int(rows[0])
    lo = r0 - int(round(KC["lookback_s"] * FPS))
    last = min(int(rows[-1]), r0 + int(round(KC["max_after_start_s"] * FPS)))
    slack = int(round(FPS))
    for ko in find_kickoffs(m, lo, last + 2):
        if r0 - slack <= ko["kick"] <= last:
            # the period must have been dead until the formation: play before it (running) means
            # the real kick-off was missed and this formation is a restart after a goal
            ko["live_share"] = live_share(m, r0, ko["start"])
            if ko["live_share"] > KC["max_live_share"]:
                continue
            return ko if info else max(ko["kick"], r0)
    return None


def apply_kickoff(labels, m, periods):
    """Copy of labels with D on every frame of each given period before its detected kick-off."""
    out = np.array(labels, copy=True)
    for per in periods:
        k = detect_kickoff(m, per)
        if k is None:
            print(f"WARNING: no kick-off detected in period {per} of {m.name}: labels unchanged",
                  file=sys.stderr)
            continue
        sel = np.flatnonzero(m.period == per)
        out[sel[sel < k]] = D
    return out
