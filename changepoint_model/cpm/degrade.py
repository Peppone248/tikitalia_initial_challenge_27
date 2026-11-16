"""A priori degradation of clean external tracking, so that it looks like automatic tracking.

Nothing here is measured on the provided match: every constant is a round a priori value, written
down before any run (DATA.md says only that the ball is missing on 41% of the graded frames and
that it goes missing "disproportionately on the frames that matter"; the team's own reports say
the ball is seen on about 8 of 10 alive frames and on about 3 of 10 dead frames in the provided
tracking, and that 3 of 4 missing frames sit in gaps longer than 15 s).

Three effects, each switchable, all driven by one seeded generator (deterministic):

1. dead_hide:   each dead-ball spell loses the ball with probability `dead_hide_p` (where the clean
                source still shows it; Metrica already hides it).
2. live_gaps:   gaps are cut into the alive ball until `live_seen` of the alive frames are left,
                lengths log-uniform between `gap_s` (1 s .. 30 s): long gaps dominate the missing
                frames, as in the provided match.
3. player_drop: each player loses `player_drop` of his frames in stretches of 1-10 s (broken
                tracks). Teams are never swapped here.

Two named scenarios are used in the evaluation:
    clean      nothing
    degraded   dead_hide 0.8, live_seen 0.8, player_drop 0.07  (the default scenario)
"""

import numpy as np

from .match import D

SCENARIOS = {
    "clean": dict(dead_hide_p=0.0, live_seen=None, player_drop=0.0),
    "degraded": dict(dead_hide_p=0.8, live_seen=0.80, player_drop=0.07),
    "harsh": dict(dead_hide_p=0.9, live_seen=0.65, player_drop=0.12),
}
GAP_S = (1.0, 30.0)
DROP_S = (1.0, 10.0)


def degrade(m, scenario="degraded", seed=2027, **override):
    params = dict(SCENARIOS[scenario])
    params.update(override)
    rng = np.random.RandomState(seed)
    out = m.copy()
    out.name = m.name
    out.extra = dict(m.extra, scenario=scenario, seed=seed)
    labels = m.labels
    seen = ~np.isnan(out.ball[:, 0])
    fps = m.fps
    if params["dead_hide_p"] > 0 and labels is not None:
        dead = labels == D
        edges = np.flatnonzero(np.diff(np.concatenate([[0], dead.astype(int), [0]])))
        for s, e in zip(edges[0::2], edges[1::2]):
            if rng.rand() < params["dead_hide_p"]:
                out.ball[s:e] = np.nan
    if params["live_seen"] is not None and labels is not None:
        alive = labels < D
        seen = ~np.isnan(out.ball[:, 0])
        target = params["live_seen"]
        n_alive = int(alive.sum())
        if n_alive > 0 and seen[alive].mean() > target:
            to_hide = int(seen[alive].sum() - target * n_alive)
            hidden = 0
            lo, hi = np.log(GAP_S[0] * fps), np.log(GAP_S[1] * fps)
            alive_idx = np.flatnonzero(alive & seen)
            guard = 0
            while hidden < to_hide and guard < 100000:
                guard += 1
                L = int(np.exp(rng.uniform(lo, hi)))
                s = int(alive_idx[rng.randint(len(alive_idx))])
                e = min(s + L, m.T)
                seg = out.ball[s:e, 0]
                n_new = int(np.sum(~np.isnan(seg) & alive[s:e]))
                out.ball[s:e] = np.nan
                hidden += n_new
    if params["player_drop"] > 0:
        P = out.px.shape[1]
        lo, hi = DROP_S[0] * fps, DROP_S[1] * fps
        for j in range(P):
            tracked = np.flatnonzero(~np.isnan(out.px[:, j]))
            if len(tracked) == 0:
                continue
            to_drop = int(params["player_drop"] * len(tracked))
            dropped, guard = 0, 0
            while dropped < to_drop and guard < 10000:
                guard += 1
                L = int(rng.uniform(lo, hi))
                s = int(tracked[rng.randint(len(tracked))])
                e = min(s + L, m.T)
                n_new = int(np.sum(~np.isnan(out.px[s:e, j])))
                out.px[s:e, j] = np.nan
                out.py[s:e, j] = np.nan
                dropped += n_new
    return out
