"""Dead-ball rules (D-*). Definitions: docs/REGOLE_PSEUDO_LABEL.md 5."""

import numpy as np

from ..preprocessing import runs
from .base import DeadRule, RuleOutput
from .registry import register
from .strength import combine_and, duration_factor, to_strength, z_score

GOAL_CODE = "GOALS"
SET_PIECES_CODE = "SET PIECES"
KO_AFTER_GOAL = "kick off (after goal)"
# Restart types compatible with the line the ball crossed (coherence check only, never a label)
COMPATIBLE_RESTARTS = {
    "touch": {"throw-in"},
    "goal": {"corner kick", "goal kick", "kick off (after goal)"},
}


def merge_runs(run_list, max_gap):
    """Join runs separated by at most max_gap frames."""
    merged = []
    for start, end in run_list:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def overlapping(intervals, lo, hi):
    """Smart Tagging entries whose [start_frame, end_frame) overlaps [lo, hi)."""
    return [t for t in intervals if t["start_frame"] < hi and t["end_frame"] > lo]


@register
class BallOut(DeadRule):
    """D-OUT -- ball visible beyond the pitch boundary by more than OUT_MARGIN -> D.

    Strength: z = (distance beyond the line - OUT_MARGIN) / SIGMA_POS, peak over the episode,
    times a duration factor (an excursion shorter than 2 * T_MIN_OUT is penalised).
    Span: only the frames where the ball is seen out (plus merged short gaps). Extending the D
    until the restart is D-HOLD's job, not this rule's.
    """

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        out = RuleOutput(self.id, match.n)
        dist = pitch.dist_outside(match.ball_x, match.ball_y)
        beyond = match.ball_visible & (np.nan_to_num(dist, nan=-np.inf) > p.OUT_MARGIN)
        set_pieces = [t for t in match.smart_tagging if t["code"] == SET_PIECES_CODE]
        inside = match.ball_visible & (np.nan_to_num(dist, nan=np.inf) <= 0)

        for start, end in merge_runs(runs(beyond), int(round(p.MERGE_GAP * fps))):
            seen = beyond[start:end]
            z = z_score(dist[start:end] - p.OUT_MARGIN, p.SIGMA_POS)
            z = np.where(seen, z, np.nan)
            peak = int(start + np.nanargmax(z))
            observed_s = seen.sum() / fps
            factor = duration_factor(observed_s, p.T_MIN_OUT)
            episode_strength = float(to_strength(np.nanmax(z), p.Z_SAT)) * factor
            per_frame = np.where(seen, to_strength(np.nan_to_num(z), p.Z_SAT) * factor,
                                 episode_strength)

            line = str(pitch.crossed_line(match.ball_x[peak], match.ball_y[peak]))
            st = overlapping(set_pieces, start, start + int(p.ST_MATCH_WIN * fps))
            st_type = min(st, key=lambda t: t["start_frame"])["type"] if st else ""
            if not st_type:
                coherence = "no_tag"
            elif st_type in COMPATIBLE_RESTARTS.get(line, set()):
                coherence = "coherent"
            else:
                coherence = "incoherent"

            back = np.flatnonzero(inside[end:])
            out.emit(start, end, "D", per_frame,
                     half=int(match.half[start]), graded=bool(match.half[start] > 0),
                     line=line, duration_s=round((end - start) / fps, 2),
                     observed_s=round(observed_s, 2),
                     peak_dist_m=round(float(dist[peak]), 2), peak_z=round(float(np.nanmax(z)), 2),
                     duration_factor=round(factor, 2),
                     end_reason="ball_lost" if end < match.n and not match.ball_visible[end]
                     else "back_inside",
                     reappear_inside_s=round(back[0] / fps, 2) if back.size else None,
                     st_type=st_type, st_coherence=coherence)
        return out


@register
class Goal(DeadRule):
    """D-GOAL (v0.4) -- start from the kick-off, look back for the goal.

    Anchor: a kick-off that is not the first of its half. Under Law 8 a kick-off during a half
    only follows a goal, and the team kicking off is the one that conceded. The ATD has no Z, so
    a ball seen between the posts beyond the line is NOT enough on its own (a shot over the bar
    looks the same from above); the kick-off is what decides.

    Search window [kick-off start - T_GOAL_KO, kick-off start], towards the goal of the team
    kicking off (from attack_dir):
      in_goal    ball seen beyond that goal line, between the posts (+ GOAL_TOL)
      lost_shot  ball lost (gap >= T_GAP_LOST) while moving towards that goal: the straight-line
                 extrapolation crosses the line between the posts within T_SHOT_EXTRAP
      lost_box   ball lost (gap >= T_GAP_LOST), last seen inside that penalty box
    Candidates closer than T_GOAL_CLUSTER are one event (goal + ball fetched from the net):
    the goal is the first frame of the latest cluster. No candidate -> 'no_candidate': only the
    kick-off formation is labelled.

    Strength (docs/REGOLE_PSEUDO_LABEL.md 4.1):
      s = clip(z / Z_SAT) * W_SRC[source] * (1 | W_NO_ST) * (1 | W_NO_ST_KO)
      z: in_goal   min(depth behind the line, lateral margin inside the posts) / SIGMA_POS
         lost_shot lateral margin at the extrapolated crossing / (SIGMA_POS + SIGMA_EXTRAP_GROWTH * t)
         lost_box  depth inside the box / SIGMA_POS
    Smart Tagging (GOALS, Kick Off (After Goal)) only raises or lowers the strength.

    Also recorded, labelling nothing: 'void' -- ball between the posts beyond the line with no
    kick-off after it (shot over the bar, disallowed goal, or a missed kick-off); Smart Tagging
    GOALS with no confirmed goal nearby.
    """

    def fires(self, match, pitch, context):
        p, fps = self.p, match.fps
        out = RuleOutput(self.id, match.n)
        bx, by, visible = match.ball_x, match.ball_y, match.ball_visible
        beyond = pitch.dist_beyond_goal_line(bx)
        mouth = pitch.mouth_margin(by, p.GOAL_TOL)
        in_goal = visible & (np.nan_to_num(beyond, nan=-np.inf) > p.OUT_MARGIN) \
            & (np.nan_to_num(mouth, nan=-np.inf) > 0)
        st_goals = [t for t in match.smart_tagging if t["code"] == GOAL_CODE]
        window = int(p.T_GOAL_KO * fps)
        covered = np.zeros(match.n, dtype=bool)          # search windows of accepted kick-offs
        confirmed = []

        for ko in context["kickoffs"]:
            if ko.get("first_of_half") or ko.get("rejected") or ko["kicker"] < 0:
                continue
            half, kicker = ko["half"], ko["kicker"]
            sign = context["attack_dir"].get((kicker, half))
            if sign is None:
                out.record(status="no_attack_dir", frame=ko["start"], half=half)
                continue
            end_x = 0.0 if sign == 1 else pitch.length       # own goal of the team kicking off
            lo, hi = max(0, ko["start"] - window), ko["start"]
            covered[lo:hi] = True
            cands = self._in_goal(in_goal, beyond, mouth, bx, end_x, pitch, lo, hi, p) \
                + self._lost(match, pitch, end_x, lo, hi, p)
            goal = self._pick(cands, int(p.T_GOAL_CLUSTER * fps))

            st_lo = (goal["frame"] if goal else lo) - int(p.ST_GOAL_WIN * fps)
            st_goal = bool(overlapping(st_goals, st_lo, ko["kick"]))
            st_ko = ko.get("st_tag", "") == KO_AFTER_GOAL
            factor = (1.0 if st_goal else p.W_NO_ST) * (1.0 if st_ko else p.W_NO_ST_KO)
            info = dict(half=half, scorer=1 - kicker, st_goal=st_goal, st_kickoff_after_goal=st_ko,
                        kicker_s=ko.get("kicker_s"), ball_seen_at_kickoff=ko.get("ball_seen"),
                        ball_after_kickoff_m=ko.get("ball_after_m"),
                        kickoff_rescued_by_st=ko.get("rescued_by_st"))
            if goal is None:
                strength = p.W_SRC_NONE * factor
                out.emit(ko["start"], ko["kick"], "D", strength, status="no_candidate",
                         source="none", z=None, frame=ko["start"], to_kickoff_s=0.0, **info)
                continue
            weight = {"in_goal": 1.0, "lost_shot": p.W_SRC_LOST_SHOT,
                      "lost_box": p.W_SRC_LOST_BOX}[goal["source"]]
            strength = float(to_strength(goal["z"], p.Z_SAT)) * weight * factor
            out.emit(goal["frame"], ko["kick"], "D", strength, status="confirmed",
                     source=goal["source"], z=round(goal["z"], 2), frame=goal["frame"],
                     to_kickoff_s=round((ko["start"] - goal["frame"]) / fps, 1), **info)
            confirmed.append(goal["frame"])

        for start, end in runs(in_goal & ~covered):
            out.record(status="void", source="in_goal", frame=int(start),
                       half=int(match.half[start]),
                       z=round(float(z_score(min(np.max(beyond[start:end]) - p.OUT_MARGIN,
                                                 mouth[start]), p.SIGMA_POS)), 2))
        for t in st_goals:
            near = [f for f in confirmed if t["start_frame"] - window <= f < t["end_frame"]]
            if not near:
                out.record(status="st_goal_without_goal", frame=t["start_frame"],
                           half=int(match.half[min(t["start_frame"], match.n - 1)]))
        return out

    # -- candidates ------------------------------------------------------------------------------
    @staticmethod
    def _towards_end(end_x, pitch):
        return -1.0 if end_x < pitch.length / 2.0 else 1.0

    def _in_goal(self, in_goal, beyond, mouth, bx, end_x, pitch, lo, hi, p):
        found = []
        for start, end in runs(in_goal[lo:hi]):
            start, end = lo + start, lo + end
            if pitch.goal_end(bx[start]) != end_x:
                continue
            z = combine_and(z_score(np.max(beyond[start:end]) - p.OUT_MARGIN, p.SIGMA_POS),
                            z_score(mouth[start], p.SIGMA_POS))
            found.append({"frame": int(start), "source": "in_goal", "z": float(z)})
        return found

    def _lost(self, match, pitch, end_x, lo, hi, p):
        found = []
        fps = match.fps
        direction = self._towards_end(end_x, pitch)
        min_gap = int(round(p.T_GAP_LOST * fps))
        for g_start, g_end in runs(~match.ball_visible[lo:hi]):
            g_start, g_end = lo + g_start, lo + g_end
            last = g_start - 1
            if last < 0 or g_end - g_start < min_gap or not match.ball_visible[last]:
                continue
            x, y = match.ball_x[last], match.ball_y[last]
            if abs(x - end_x) > pitch.length / 2.0:
                continue                                   # lost in the other half
            vx, vy = match.ball_vx[last], match.ball_vy[last]
            to_line = abs(end_x - x)
            if np.isfinite(vx) and vx * direction > 0:
                t = to_line / abs(vx)
                gap_s = (g_end - g_start) / fps
                if t <= min(p.T_SHOT_EXTRAP, gap_s):
                    y_cross = y + vy * t
                    margin = float(pitch.mouth_margin(y_cross, p.GOAL_TOL))
                    if margin > 0:
                        sigma = p.SIGMA_POS + p.SIGMA_EXTRAP_GROWTH * t
                        found.append({"frame": int(last), "source": "lost_shot",
                                      "z": float(margin / sigma)})
                        continue
            depth = pitch.box_depth - to_line
            lateral = pitch.box_half_width - abs(y - pitch.width / 2.0)
            if depth > 0 and lateral > 0:
                found.append({"frame": int(last), "source": "lost_box",
                              "z": float(z_score(min(depth, lateral), p.SIGMA_POS))})
        return found

    @staticmethod
    def _pick(cands, cluster_gap):
        """First frame of the latest cluster; source/z of the best candidate in that cluster."""
        if not cands:
            return None
        cands = sorted(cands, key=lambda c: c["frame"])
        cluster = [cands[-1]]
        for c in reversed(cands[:-1]):
            if cluster[0]["frame"] - c["frame"] <= cluster_gap:
                cluster.insert(0, c)
            else:
                break
        rank = {"in_goal": 0, "lost_shot": 1, "lost_box": 2}
        best = min(cluster, key=lambda c: (rank[c["source"]], -c["z"]))
        return {"frame": cluster[0]["frame"], "source": best["source"], "z": best["z"]}
