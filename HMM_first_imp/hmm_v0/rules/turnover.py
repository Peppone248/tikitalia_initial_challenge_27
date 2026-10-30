"""TURNOVER rules (REGOLE_PSEUDO_LABEL 6.2): they say WHEN the ball changed team, and only if a
control of the new team confirms it within T_TURN_CONFIRM (a touch, a deflection or an aerial
duel without control is not a change, CHALLENGE.md). Their episodes open the team gate of the
HMM (hmm.py) at that instant and vote for the new team over the next T_TURN_SPAN.
"""

import numpy as np

from tikitalia.preprocessing import runs
from ..noise import strength
from .base import HmmRule, LABEL_OF_TEAM
from .registry import register


def _confirmed(eps, i, fps, p):
    """Episode i (new team) is a control (>= T_CONTROL), or the new team plays the ball on: the
    next contact is of the same team within T_PASS_MAX (a pass reception can take that long),
    before any contact of the other team. Returns (ok, how)."""
    e = eps[i]
    if e["dur_s"] >= p.T_CONTROL:
        return True, "control"
    for nxt in eps[i + 1:]:
        if nxt["start"] - e["end"] > p.T_PASS_MAX * fps:
            break
        if nxt["team"] != e["team"]:
            return False, "lost_again"
        return True, "next_touch"
    return False, "no_follow_up"


@register
class Intercept(HmmRule):
    """T-INTERCEPT (+ T-LOOSE) -- the next contact after a contact of team T is by the other team
    (a pass or a carry interrupted, a tackle, or a loose / unseen ball recovered: no limit on the
    gap inside a half), confirmed (see _confirmed). Instant: the first frame of the new team's
    contact. Strength: control strength of that contact (MAD z-scores) * 1 if it is a control of
    >= T_CONTROL, 0.8 if confirmed by the next touch of the same team."""
    emits = ("H", "A")

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        out = self.new_output(match)
        eps = context["control_episodes"]
        span = int(round(p.T_TURN_SPAN * fps))
        holder, holder_end, half = None, 0, None     # last CONFIRMED holder of the ball
        for i, e in enumerate(eps):
            h = int(match.half[e["start"]])
            if h != half:
                holder, half = None, h               # memory never crosses half-time
            if holder is None:
                holder, holder_end = e["team"], e["end"]
                continue
            if e["team"] == holder:
                holder_end = e["end"]
                continue
            ok, how = _confirmed(eps, i, fps, p)
            if not ok:
                out.record(status="unconfirmed", frame=int(e["start"]), reason=how)
                continue                              # a touch / deflection: holder unchanged
            s = float(strength([e["z"][:max(1, int(fps))]], p.Z_SAT).mean())
            s *= 1.0 if how == "control" else 0.8
            out.emit(e["start"], min(match.n, e["start"] + span), LABEL_OF_TEAM[e["team"]], s,
                     confirmed_by=how, gap_s=round((e["start"] - holder_end) / fps, 2))
            holder, holder_end = e["team"], e["end"]
        return out


@register
class Reversal(HmmRule):
    """T-REV -- the ball reverses direction (angle >= ANGLE_REV between its mean velocity in the
    T_REV_WIN before and after), it was going towards the goal attacked by the team that had it
    and now goes towards that team's own goal, and at that instant the nearest player is an
    opponent within R_CONTROL; confirmed by a contact of the new team starting within
    T_TURN_CONFIRM. Instant: the frame of the sharpest reversal in the run.
    Strength: min(distance z / Z_SAT, (angle - ANGLE_REV) / (180 - ANGLE_REV))."""
    emits = ("H", "A")

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        f, noise = context["features"], context["noise"]
        out = self.new_output(match)
        n = match.n
        w = max(2, int(round(p.T_REV_WIN * fps)))
        vx, vy = match.ball_vx, match.ball_vy

        def window_mean(v, before):
            ok = np.isfinite(v)
            c = np.concatenate(([0.0], np.cumsum(np.where(ok, v, 0.0))))
            k = np.concatenate(([0], np.cumsum(ok)))
            idx = np.arange(n)
            lo, hi = (np.clip(idx - w, 0, n), idx) if before else (idx, np.clip(idx + w, 0, n))
            cnt = k[hi] - k[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                m = (c[hi] - c[lo]) / cnt
            m[cnt < w // 2] = np.nan
            return m

        bx, by = window_mean(vx, True), window_mean(vy, True)
        ax, ay = window_mean(vx, False), window_mean(vy, False)
        nb, na = np.hypot(bx, by), np.hypot(ax, ay)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (bx * ax + by * ay) / (nb * na)
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        moving = (np.nan_to_num(nb) > p.V_REV_MIN) & (np.nan_to_num(na) > p.V_REV_MIN)
        cand = moving & (np.nan_to_num(angle) >= p.ANGLE_REV) & match.graded

        eps = context["control_episodes"]
        starts = np.array([e["start"] for e in eps]) if eps else np.array([], dtype=int)
        ends = np.array([e["end"] for e in eps]) if eps else np.array([], dtype=int)
        attack = context["attack_dir"]
        span = int(round(p.T_TURN_SPAN * fps))
        for s0, e0 in runs(cand):
            fr = int(s0 + np.nanargmax(angle[s0:e0]))
            k = np.searchsorted(ends, fr, side="right") - 1          # last contact before
            if k < 0 or fr - ends[k] > p.T_PASS_MAX * fps:
                continue
            prev_team = eps[k]["team"]
            new = 1 - prev_team
            d = attack.get((prev_team, int(match.half[fr])))
            if d is None or not (bx[fr] * d > 0 and ax[fr] * d < 0):
                continue                                              # not "forward then back"
            if f.nearest_team[fr] != new:
                continue
            z_d = float(noise.z(p.R_CONTROL - f.nearest_dist[fr, new], "nearest_dist"))
            if not z_d > 0:
                continue
            j = np.searchsorted(starts, fr - int(0.2 * fps))
            nxt = [eps[i] for i in range(j, len(eps))
                   if eps[i]["start"] <= fr + p.T_TURN_CONFIRM * fps]
            if not any(e["team"] == new for e in nxt[:1]):
                out.record(status="unconfirmed", frame=fr)
                continue
            s = min(z_d / p.Z_SAT, (angle[fr] - p.ANGLE_REV) / (180.0 - p.ANGLE_REV))
            out.emit(fr, min(n, fr + span), LABEL_OF_TEAM[new], float(np.clip(s, 0, 1)),
                     angle=round(float(angle[fr]), 1))
        return out
