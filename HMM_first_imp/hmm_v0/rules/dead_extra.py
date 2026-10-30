"""D rules missing in the team code (REGOLE_PSEUDO_LABEL 5): D-KO, D-HOLD, D-WHISTLE, D-GAP,
D-SP, D-STILL. D-OUT and D-GOAL are the team ones (tikitalia.rules.dead), reused unchanged."""

import numpy as np

from tikitalia.preprocessing import runs
from ..noise import strength
from .base import HmmRule
from .registry import register

SET_PIECES_CODE = "SET PIECES"


def _taper(n, n_taper, low):
    """Strength profile: 1, then linearly down to `low` over the last n_taper frames."""
    s = np.ones(n)
    k = min(n, n_taper)
    if k > 0:
        s[n - k:] = np.linspace(1.0, low, k)
    return s


@register
class KickOffDead(HmmRule):
    """D-KO -- D during every kick-off formation found by the team detector (Law 8), and from the
    first graded frame of a half to its first kick-off (FAQ: the 2nd half is graded ~2 min
    before its kick-off; nothing is hard-coded, the kick-off is detected).

    The detected kick is the end of the formation, which can be late by a fraction of a second:
    the strength tapers to options.end_strength over the last options.taper_s seconds, so the
    exact restart frame is decided by the HMM with the other evidence.
    """
    emits = ("D",)

    def fires(self, match, pitch, context):
        out = self.new_output(match)
        fps = match.fps
        n_taper = int(round(self.options.get("taper_s", 1.0) * fps))
        low = float(self.options.get("end_strength", 0.3))
        st_factor = float(self.options.get("no_st_factor", 0.8))
        for ko in context["kickoffs"]:
            if ko.get("rejected"):
                continue
            half = ko["half"]
            first_graded = int(np.flatnonzero(match.half == half)[0])
            start = ko["start"]
            if ko.get("first_of_half"):
                start = min(start, first_graded)
            start, end = max(start, first_graded), ko["kick"]
            if end <= start:
                continue
            factor = 1.0 if ko.get("st_tag") else st_factor
            out.emit(start, end, "D", factor * _taper(end - start, n_taper, low), half=half,
                     first_of_half=bool(ko.get("first_of_half")), kicker=ko["kicker"])
        return out


@register
class DeadHold(HmmRule):
    """D-HOLD -- stage 2: after a D-OUT / D-GOAL episode, D until the restart.

    The D continues from the end of the episode to the first of: a restart kick (events), the
    start of a control episode of at least T_CONTROL inside the pitch while the players are not
    collectively slowed down (median slowdown of all >= SLOWDOWN_RATIO: a player carrying the
    ball to the restart spot is not a restart), T_HOLD_MAX. Strength
    options.strength * exp(-t / options.decay_s), t = seconds since the D episode ended: it is
    persistence, weaker than the anchor that opened the D, and fading if no restart is seen.
    """
    emits = ("D",)

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        out = self.new_output(match)
        restarts = np.array(sorted(r["frame"] for r in context["restarts"]) or [match.n])
        slow_all = context["features"].slowdown[:, 2]

        def live(e):
            seg = slow_all[e["start"]:e["end"]]
            return np.all(np.isnan(seg)) or np.nanmedian(seg) >= p.SLOWDOWN_RATIO

        controls = np.array(sorted(e["start"] for e in context["control_episodes"]
                                   if e["dur_s"] >= p.T_CONTROL and live(e)) or [match.n])
        level = float(self.options.get("strength", 0.8))
        for rid in self.options.get("after", ["D-OUT", "D-GOAL"]):
            src = context["stage_outputs"].get(rid)
            if src is None:
                continue
            for ep in src.episodes:
                if "end" not in ep or ep.get("label") != "D":
                    continue
                start = ep["end"]
                cap = start + int(round(p.T_HOLD_MAX * fps))
                r = restarts[np.searchsorted(restarts, start)] if start <= restarts[-1] else match.n
                c = controls[np.searchsorted(controls, start)] if start <= controls[-1] else match.n
                end = int(min(r, c, cap, match.n))
                if end > start:
                    t = np.arange(end - start) / fps
                    s = level * np.exp(-t / float(self.options.get("decay_s", 30.0)))
                    out.emit(start, end, "D", s, after=rid,
                             end_reason=["restart", "control", "cap"][int(np.argmin([r, c, cap]))])
        return out


@register
class Whistle(HmmRule):
    """D-WHISTLE -- a whistle seen from the players: the collective tempo drops.

    Onset: median speed of ALL players <= SLOWDOWN_RATIO of their recent tempo for at least
    T_WHISTLE_MIN, and both teams slowed (median over the run of the faster team's ratio <=
    SLOWDOWN_RATIO_BOTH). The ratio marks the START of a stoppage, not its length (FEATURES.md:
    in a long stop the reference drops too), so the rule only opens the D: D-HOLD (stage 2,
    after: D-WHISTLE) keeps it until the restart.
    Confirmed if, in [onset, end of the run + T_WHISTLE_CONFIRM], the ball is held still for
    T_STILL_RESTART, OR nobody of either team touches it (no control episode): a slow build-up
    with the ball played between team-mates is not a whistle (REGOLE D-WHISTLE edge cases).
    Unconfirmed onsets are only recorded.
    Strength: z of the slowdown margin (MAD noise) * duration factor; 0.8x if confirmed only by
    the absence of contacts.
    """
    emits = ("D",)

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        f, noise = context["features"], context["noise"]
        out = self.new_output(match)
        sl_all = f.slowdown[:, 2]
        faster = np.max(f.slowdown[:, :2], axis=1)
        slow = np.nan_to_num(sl_all, nan=np.inf) <= p.SLOWDOWN_RATIO
        touched = np.zeros(match.n, dtype=bool)
        for e in context["control_episodes"]:
            touched[e["start"]:e["end"]] = True
        conf = int(round(p.T_WHISTLE_CONFIRM * fps))
        for start, end in runs(slow & match.graded):
            dur = (end - start) / fps
            if dur < p.T_WHISTLE_MIN:
                continue
            both = faster[start:end]
            if np.all(np.isnan(both)) or np.nanmedian(both) > p.SLOWDOWN_RATIO_BOTH:
                continue
            hi = min(match.n, end + conf)
            still = bool((match.ball_still_for[start:hi] >= p.T_STILL_RESTART).any())
            untouched = not touched[start:hi].any()
            if not (still or untouched):
                out.record(status="unconfirmed", frame=int(start), dur_s=round(dur, 2))
                continue
            z = noise.z(p.SLOWDOWN_RATIO - sl_all[start:end], "slowdown")
            s = strength([z], p.Z_SAT, min(1.0, dur / (2 * p.T_WHISTLE_MIN)))
            s = s * (1.0 if still else 0.8)
            out.emit(start, end, "D", s, dur_s=round(dur, 2),
                     confirmed_by="ball_still" if still else "no_contact")
        return out


@register
class WhistleV0(HmmRule):
    """D-WHISTLE v0 (variant A, frozen for config_v0) -- both teams slow down below SLOWDOWN_RATIO of their recent tempo
    for >= T_WHISTLE_MIN, and during that stretch the ball is held still (>= T_STILL_RESTART) or
    not seen -> D over the stretch (retroactive from its start, REGOLE 5)."""
    emits = ("D",)

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        f, noise = context["features"], context["noise"]
        out = self.new_output(match)
        sl = np.max(f.slowdown[:, :2], axis=1)           # both teams slow <=> the faster one is
        slow = np.nan_to_num(sl, nan=np.inf) <= p.SLOWDOWN_RATIO
        ball_dead_hint = (match.ball_still_for >= p.T_STILL_RESTART) | ~match.ball_visible
        for start, end in runs(slow & match.graded):
            dur = (end - start) / fps
            if dur < p.T_WHISTLE_MIN or not ball_dead_hint[start:end].any():
                continue
            z = noise.z(p.SLOWDOWN_RATIO - sl[start:end], "slowdown")
            s = strength([z], p.Z_SAT, min(1.0, dur / (2 * p.T_WHISTLE_MIN)))
            out.emit(start, end, "D", s, dur_s=round(dur, 2))
        return out


@register
class LongGap(HmmRule):
    """D-GAP -- the ball is missing for >= T_GAP_LONG AND the play slows down collectively
    (median slowdown of all players over the first part of the gap <= SLOWDOWN_RATIO).
    A gap alone is never enough: the ball is missing in 41% of graded frames (REGOLE 5)."""
    emits = ("D",)

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        f, noise = context["features"], context["noise"]
        out = self.new_output(match)
        head = int(round(p.REF_WIN * fps))
        for start, end in runs(~match.ball_visible & match.graded):
            dur = (end - start) / fps
            if dur < p.T_GAP_LONG:
                continue
            seg = f.slowdown[start:min(end, start + head), 2]
            if np.all(np.isnan(seg)):
                continue
            med = float(np.nanmedian(seg))
            if med > p.SLOWDOWN_RATIO:
                continue
            s = strength([noise.z(p.SLOWDOWN_RATIO - med, "slowdown")], p.Z_SAT)
            out.emit(start, end, "D", float(s), dur_s=round(dur, 1), slowdown=round(med, 2))
        return out


@register
class SetPiece(HmmRule):
    """D-SP -- Smart Tagging SET PIECES interval and the ball held still for >= T_STILL_RESTART
    inside it -> D from the start of the still ball to the next restart kick (not the whole
    interval: it lasts ~35 s and contains play after the restart, REGOLE 5)."""
    emits = ("D",)

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        out = self.new_output(match)
        restarts = np.array(sorted(r["frame"] for r in context["restarts"]) or [match.n])
        for t in match.smart_tagging:
            if t["code"] != SET_PIECES_CODE:
                continue
            lo, hi = max(0, t["start_frame"]), min(match.n, t["end_frame"])
            still = np.flatnonzero(match.ball_still_for[lo:hi] >= p.T_STILL_RESTART)
            if still.size == 0:
                continue
            f_still = lo + int(still[0])
            start = max(lo, f_still - int(round(match.ball_still_for[f_still] * fps)) + 1)
            k = np.searchsorted(restarts, f_still)
            end = int(restarts[k]) if k < len(restarts) and restarts[k] <= hi else hi
            if end > start:
                out.emit(start, end, "D", 1.0, type=t["type"])
        return out


@register
class StillBall(HmmRule):
    """D-STILL -- SOFT: ball visible and at rest for >= T_STILL_DEAD while the players are
    slower than their recent tempo (slowdown of all < 1): a ball placed for a restart. Covers
    the tip "speed ~0 for several frames -> D". Span: the whole still stretch."""
    emits = ("D",)

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        f, noise = context["features"], context["noise"]
        out = self.new_output(match)
        still = match.ball_still_for > 0
        for start, end in runs(still & match.graded):
            dur = (end - start) / fps
            if dur < p.T_STILL_DEAD:
                continue
            sl = f.slowdown[start:end, 2]
            if np.all(np.isnan(sl)) or np.nanmedian(sl) >= 1.0:
                continue
            z = noise.z(1.0 - float(np.nanmedian(sl)), "slowdown")
            s = strength([z], p.Z_SAT, min(1.0, dur / (2 * p.T_STILL_DEAD)))
            out.emit(start, end, "D", float(s), dur_s=round(dur, 1))
        return out
