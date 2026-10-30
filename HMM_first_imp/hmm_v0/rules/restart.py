"""RESTART rules (REGOLE_PSEUDO_LABEL 7): who restarts play after a D. They label the first
T_RESTART_SPAN seconds of the new possession; the HMM also uses their frames to allow D -> H/A."""

import numpy as np

from .base import HmmRule, LABEL_OF_TEAM
from .registry import register


@register
class RestartTouch(HmmRule):
    """R-TOUCH -- at every restart kick (ball still, then kicked): the team of the nearest
    player in [kick - T_RESTART_LOOKBACK, kick]. Strength = kick strength (acceleration vs
    ACC_KICK) * share of the window where that team is the nearest."""
    emits = ("H", "A")

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        f = context["features"]
        out = self.new_output(match)
        back = int(round(p.T_RESTART_LOOKBACK * fps))
        span = int(round(p.T_RESTART_SPAN * fps))
        for r in context["restarts"]:
            if r["kind"] != "still_kick":
                continue
            fr = r["frame"]
            near = f.nearest_team[max(0, fr - back):fr + 1]
            near = near[near >= 0]
            if near.size == 0:
                continue
            counts = np.bincount(near, minlength=2)
            team = int(counts.argmax())
            share = counts[team] / float(near.size)
            out.emit(fr, min(match.n, fr + span), LABEL_OF_TEAM[team], r["strength"] * share,
                     kind=r["kind"])
        return out


@register
class RestartKickOff(HmmRule):
    """R-KO -- the team kicking off (player on the centre spot during the formation, team
    detector) has the ball after the kick."""
    emits = ("H", "A")

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        out = self.new_output(match)
        span = int(round(p.T_RESTART_SPAN * fps))
        for ko in context["kickoffs"]:
            if ko.get("rejected") or ko["kicker"] < 0:
                continue
            out.emit(ko["kick"], min(match.n, ko["kick"] + span), LABEL_OF_TEAM[ko["kicker"]],
                     1.0, half=ko["half"])
        return out
