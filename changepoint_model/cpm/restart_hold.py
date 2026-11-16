"""A stoppage lasts until the restart kick (Laws 13, 15, 16, 17: the ball is in play when it is
kicked, or thrown, and clearly moves).

Why (debug on the provided video, REPORT §0.4): in 14 of the 31 decidable stretches v4 ended a
stoppage too early. The ball was seen but moving slowly (1-6 m/s): players placing it, passing it to
the taker, the keeper setting it down for a goal kick. The model read those touches as control and
restarted play, often for the wrong team.

Rule, applied to the decoded labels, period by period. After a D spell that ends at t0:
    k = the first restart kick after t0: the ball seen faster than KICK_SPEED over KICK_DT, having
        been seen slower than REST_SPEED within the REST_S before;
    if k - t0 <= MAX_HOLD_S, no ball faster than KICK_SPEED is seen in [t0, k) and the ball is seen on
    at least MIN_SEEN of [t0, k): the stoppage is extended to k.
    if the next D spell starts at t1 before k (an island of "play" between two stoppages, inside
    MAX_HOLD_S) with no fast ball in [t0, t1) and the ball seen on MIN_SEEN of it: [t0, t1) is D.
Otherwise nothing changes. All constants a priori: KICK_SPEED and REST_SPEED are those of the team's
restart-kick rule R1; MAX_HOLD_S and MIN_SEEN round numbers, set before any run.
"""

import numpy as np

from .match import D, X

KICK_SPEED = 8.0
KICK_DT = 0.2
REST_SPEED = 2.0
REST_S = 1.0
MAX_HOLD_S = 10.0
MIN_SEEN = 0.5


def ball_speed(ball, fps, dt=KICK_DT):
    """Speed (m/s) from the seen positions over dt (forward difference); NaN where either is unseen."""
    k = max(1, int(round(dt * fps)))
    T = len(ball)
    v = np.full(T, np.nan)
    d = np.hypot(ball[k:, 0] - ball[:-k, 0], ball[k:, 1] - ball[:-k, 1])
    v[:T - k] = d / (k / fps)
    return v


def apply_restart_hold(labels, m, max_hold_s=MAX_HOLD_S, kick_speed=KICK_SPEED):
    """Returns (labels, info)."""
    out = labels.copy()
    fps = m.fps
    seen = ~np.isnan(m.ball[:, 0])
    v = ball_speed(m.ball, fps)
    fast = np.nan_to_num(v, nan=0.0) > kick_speed
    slow = np.nan_to_num(v, nan=np.inf) < REST_SPEED
    r = int(round(REST_S * fps))
    H = int(round(max_hold_s * fps))
    info = {"extended": 0, "islands": 0, "frames_set_D": 0}
    for p, s, e in m.period_slices():
        lab = out[s:e]
        n = e - s
        t = 1
        while t < n:
            if lab[t - 1] == D and lab[t] < D:              # a stoppage ends at t
                t0 = t
                lim = min(n, t0 + H + 1)
                nxt = t0 + int(np.argmax(lab[t0:lim] == D)) if (lab[t0:lim] == D).any() else None
                f = fast[s + t0:s + lim]
                k = None
                for u in np.flatnonzero(f):
                    g = s + t0 + int(u)
                    if slow[max(s, g - r):g].any():
                        k = t0 + int(u)
                        break
                first_fast = t0 + int(np.argmax(f)) if f.any() else None
                end = None
                kind = None
                if nxt is not None and (first_fast is None or first_fast >= nxt):
                    end, kind = nxt, "islands"
                elif k is not None and (first_fast is None or first_fast >= k):
                    end, kind = k, "extended"
                if end is not None and end > t0 and seen[s + t0:s + end].mean() >= MIN_SEEN:
                    seg = lab[t0:end]
                    info["frames_set_D"] += int(np.sum((seg != D) & (seg != X)))
                    lab[t0:end] = np.where(seg == X, X, D)
                    info[kind] += 1
                    t = end
                    continue
            t += 1
        out[s:e] = lab
    return out, info
