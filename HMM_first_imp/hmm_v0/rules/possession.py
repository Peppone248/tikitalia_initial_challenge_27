"""H/A rules (REGOLE_PSEUDO_LABEL 6). Strength s = clip(min_i z_i / Z_SAT) * duration factor,
z_i = margin_i / sigma_i with sigma_i = MAD noise of the feature (noise.py)."""

import numpy as np

from tikitalia.preprocessing import HOME, AWAY, runs
from ..events import in_pitch, merge_runs
from ..noise import strength
from .base import HmmRule, LABEL_OF_TEAM
from .registry import register


def _duration_factor(observed_s, required_s):
    return float(min(1.0, observed_s / (2.0 * required_s))) if required_s > 0 else 1.0


@register
class Dribble(HmmRule):
    """P-DRIB -- control episode of a team lasting at least T_CONTROL -> that team.

    Span: the whole episode (from its first frame: the change is dated at the start of the
    control, IMPROVEMENTS A2). Strength per frame: z of the control condition (distance, gap to
    the nearest opponent, relative speed), times min(1, duration / 2 T_CONTROL).
    """
    emits = ("H", "A")

    def fires(self, match, pitch, context):
        p = self.p
        out = self.new_output(match)
        for ep in context["control_episodes"]:
            if ep["dur_s"] < p.T_CONTROL:
                continue
            factor = _duration_factor(ep["dur_s"], p.T_CONTROL)
            s = strength([ep["z"]], p.Z_SAT, factor)
            out.emit(ep["start"], ep["end"], LABEL_OF_TEAM[ep["team"]], s,
                     dur_s=round(ep["dur_s"], 2), slot=ep["slot"])
        return out


@register
class FreeBall(HmmRule):
    """P-FREE -- uncontested possession: nearest of the team within R_CONTROL, nearest opponent
    beyond R_FREE_OPP, ball slower than V_PASS, ball inside the pitch, for >= T_CONTROL."""
    emits = ("H", "A")

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        f, noise = context["features"], context["noise"]
        out = self.new_output(match)
        inside = in_pitch(match, pitch)
        speed = match.ball_speed
        for team in (HOME, AWAY):
            d = f.nearest_dist[:, team]
            d_o = np.nan_to_num(f.nearest_dist[:, 1 - team], nan=np.inf)
            z1 = np.nan_to_num(noise.z(p.R_CONTROL - d, "nearest_dist"), nan=-1)
            z2 = np.minimum(noise.z(d_o - p.R_FREE_OPP, "nearest_dist"), 1e3)
            z3 = np.nan_to_num(noise.z(p.V_PASS - speed, "ball_speed"), nan=-1)
            cond = inside & (z1 > 0) & (z2 > 0) & (z3 > 0)
            for start, end in runs(cond):
                dur = (end - start) / fps
                if dur < p.T_CONTROL:
                    continue
                s = strength([z1[start:end], z2[start:end], z3[start:end]], p.Z_SAT,
                             _duration_factor(dur, p.T_CONTROL))
                out.emit(start, end, LABEL_OF_TEAM[team], s, dur_s=round(dur, 2))
        return out


@register
class PassChain(HmmRule):
    """P-PASS -- at least N_PASS_ANCHOR receptions in a row by the same team -> that team, from
    the first control of the chain to the last, passes in flight included (offline, REGOLE 6).

    This is the rule that keeps a possession in ONE piece across passes (spellF1). Strength:
    mean control strength of the touches, times min(1, receptions / (2 N_PASS_ANCHOR)) + 0.5
    saturation (a chain of N_PASS_ANCHOR receptions already counts as 'more than two passes').
    """
    emits = ("H", "A")

    def fires(self, match, pitch, context):
        p = self.p
        out = self.new_output(match)
        for ch in context["pass_chains"]:
            if ch["receptions"] < p.N_PASS_ANCHOR:
                continue
            touch_s = np.mean([float(strength([e["z"]], p.Z_SAT).mean())
                               for e in ch["episodes"]])
            factor = min(1.0, 0.5 + ch["receptions"] / (2.0 * p.N_PASS_ANCHOR))
            out.emit(ch["start"], ch["end"], LABEL_OF_TEAM[ch["team"]], touch_s * factor,
                     receptions=ch["receptions"], dur_s=round((ch["end"] - ch["start"]) /
                                                              match.fps, 2))
        return out


@register
class ShapeOpen(HmmRule):
    """P-SHAPE-OPEN -- SOFT, continuous vote from the shape of the two teams (REGOLE 6).

    score(team) = options.w_half * (players in the opponent half, team - other)
                + options.w_vfwd * (forward speed, team - other)    [m/s]
    s = clip(|score| / options.scale, 0, 1); vote for the team with the higher score. The shape
    lags a change of possession by seconds: it fills gaps, never decides a change.
    """
    emits = ("H", "A")

    def fires(self, match, pitch, context):
        f = context["features"]
        o = self.options
        out = self.new_output(match)
        diff_half = f.n_opp_half[:, HOME].astype(float) - f.n_opp_half[:, AWAY]
        diff_v = np.nan_to_num(f.v_forward[:, HOME] - f.v_forward[:, AWAY])
        score = o.get("w_half", 1.0) * diff_half + o.get("w_vfwd", 1.0) * diff_v
        # smooth over a second: the shape is a slow signal
        k = int(round(match.fps))
        score = np.convolve(score, np.ones(k) / k, mode="same")
        s = np.clip(np.abs(score) / float(o.get("scale", 4.0)), 0, 1)
        valid = match.graded & (f.n_visible.min(axis=1) >= o.get("min_players", 7))
        for lab, sign in (("H", 1), ("A", -1)):
            m = valid & (np.sign(score) == sign) & (s > 0)
            for start, end in runs(m):
                out.emit(start, end, lab, s[start:end])
        return out


@register
class SmartTaggingPhase(HmmRule):
    """S-PHASE -- SOFT: Smart Tagging phase with a team. Possession codes vote for the team,
    defending codes for the other team (REGOLE 8). Codes are listed in options. Strength 1;
    the quality q (rules_hmm.yaml) is low: phases are not possessions."""
    emits = ("H", "A")

    def fires(self, match, pitch, context):
        out = self.new_output(match)
        poss = set(c.upper() for c in self.options.get("possession_codes", []))
        dfn = set(c.upper() for c in self.options.get("defending_codes", []))
        for t in match.smart_tagging:
            if t["team"] not in (HOME, AWAY):
                continue
            if t["code"] in poss:
                team = t["team"]
            elif t["code"] in dfn:
                team = 1 - t["team"]
            else:
                continue
            start, end = max(0, t["start_frame"]), min(match.n, t["end_frame"])
            out.emit(start, end, LABEL_OF_TEAM[team], 1.0, code=t["code"])
        return out
