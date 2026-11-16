"""One representation of a match, whatever the source.

All positions are in metres, origin at the centre spot, x along the length of the pitch, y along
the width. NaN = not tracked on that frame. Frames are consecutive video frames at `fps`.

    period      (T,) int      1 / 2 (3 / 4 for extra time); 0 = not part of the match
    ball        (T, 2) float  NaN where the ball is not seen
    px, py      (T, P) float  player positions, NaN where the player is not tracked
    team        (P,) int      0 = home, 1 = away
    attack      {period: +1 / -1}  +1 if the HOME team attacks towards +x in that period
    labels      (T,) int or None   0 = H, 1 = A, 2 = D, 3 = X (no label); the codes of the scorer
    pitch       (length, width) metres
    name        a short identifier
    frame_index (T,) int      the frame number of the source (ATD frame_id - 1 for the provided
                              match: the row of the submission)

The attack direction is *inferred* from the positions (the deepest home player in the first
minute of each period), so the same code runs on every source; where metadata exists it is only
checked against the inference.
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3
FPS = 25


class Match:
    def __init__(self, name, period, ball, px, py, team, pitch, labels=None, frame_index=None,
                 fps=FPS, attack=None, extra=None):
        self.name = name
        self.period = np.asarray(period, dtype=np.int16)
        self.ball = np.asarray(ball, dtype=np.float64)
        self.px = np.asarray(px, dtype=np.float32)
        self.py = np.asarray(py, dtype=np.float32)
        self.team = np.asarray(team, dtype=np.int8)
        self.pitch = (float(pitch[0]), float(pitch[1]))
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int8)
        T = len(self.period)
        self.frame_index = (np.arange(T) if frame_index is None
                            else np.asarray(frame_index, dtype=np.int64))
        self.fps = fps
        self.extra = extra or {}
        self.attack = attack if attack is not None else infer_attack(self)
        self.check()

    @property
    def T(self):
        return len(self.period)

    @property
    def n_players(self):
        return self.px.shape[1]

    def check(self):
        T = self.T
        assert self.ball.shape == (T, 2), self.ball.shape
        assert self.px.shape == self.py.shape and self.px.shape[0] == T
        assert self.team.shape == (self.px.shape[1],)
        assert self.labels is None or self.labels.shape == (T,)
        assert self.frame_index.shape == (T,)

    def graded(self):
        """Frames that belong to a period of play."""
        return self.period > 0

    def periods(self):
        return [int(p) for p in np.unique(self.period) if p > 0]

    def period_slices(self):
        """[(period, start, stop)] with stop exclusive; each period is one contiguous block."""
        out = []
        for p in self.periods():
            idx = np.flatnonzero(self.period == p)
            assert idx[-1] - idx[0] + 1 == len(idx), "period {} is not contiguous".format(p)
            out.append((p, int(idx[0]), int(idx[-1]) + 1))
        return out

    def copy(self, **kw):
        fields = dict(name=self.name, period=self.period.copy(), ball=self.ball.copy(),
                      px=self.px.copy(), py=self.py.copy(), team=self.team.copy(),
                      pitch=self.pitch,
                      labels=None if self.labels is None else self.labels.copy(),
                      frame_index=self.frame_index.copy(), fps=self.fps,
                      attack=dict(self.attack), extra=dict(self.extra))
        fields.update(kw)
        return Match(**fields)

    def swapped(self):
        """The same match with home and away exchanged (labels and attack direction follow)."""
        labels = None
        if self.labels is not None:
            labels = self.labels.copy()
            h, a = labels == H, labels == A
            labels[h], labels[a] = A, H
        return self.copy(team=1 - self.team, labels=labels,
                         attack={p: -s for p, s in self.attack.items()})


def infer_attack(m, seconds=60):
    """+1 if home attacks towards +x in each period, from the deepest home player vs the deepest
    away player over the first `seconds` of the period (the goalkeepers)."""
    out = {}
    for p, s, e in m.period_slices():
        e2 = min(e, s + int(seconds * m.fps))
        home = m.team == 0
        hx = np.nanmedian(np.nanmin(m.px[s:e2][:, home], axis=1)) if home.any() else np.nan
        ax = np.nanmedian(np.nanmin(m.px[s:e2][:, ~home], axis=1)) if (~home).any() else np.nan
        hxm = np.nanmedian(np.nanmax(m.px[s:e2][:, home], axis=1)) if home.any() else np.nan
        axm = np.nanmedian(np.nanmax(m.px[s:e2][:, ~home], axis=1)) if (~home).any() else np.nan
        # home keeper on the left (its min x is the smallest) -> home attacks +x
        score = (ax - hx) + (axm - hxm)   # > 0 when home sits further left
        out[p] = 1 if (np.isnan(score) or score >= 0) else -1
    return out


def spells(labels, period):
    """[(label, start, stop)] maximal runs of equal label inside a period (stop exclusive)."""
    out = []
    T = len(labels)
    if T == 0:
        return out
    start = 0
    for t in range(1, T + 1):
        if t == T or labels[t] != labels[start] or period[t] != period[start]:
            out.append((int(labels[start]), start, t))
            start = t
    return out
