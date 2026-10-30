"""Derived events shared by several rules. For now: kick-offs and the attack direction.

A kick-off is recognised from the Laws of the Game (Law 8), from the PLAYERS only:
  - every player in their own half (the kicker excepted), the two teams on opposite sides
  - no opponent inside the centre circle (one tolerated: referee / tracker duplicate)
  - a player of the kicking team reaches the centre spot (often only just before the kick)
The ball is NOT required: the first test on BAR-ATH showed the tracker does not see the ball
resting on the spot (occluded by the kicker) and at times follows another object far from play.
When the ball is seen at the centre it is recorded as supporting evidence.

The same detector finds the start of each half and the restarts after a goal, and the first one
in each half gives the attack direction -- attack_dir is computed, never hard-coded.
"""

import numpy as np

from .preprocessing import HOME, AWAY, runs


def _team_sides(match, pitch, team, line_tol, ignore=None):
    """Per frame: visible players, players clearly left / right of halfway, majority side (-1/+1).

    ignore: (n, n_slots) bool -- players not counted for the sides (the kicker in the circle).
    """
    cols = match.team_of_slot == team
    x = match.px[:, cols]
    visible = ~np.isnan(x)
    counted = visible if ignore is None else visible & ~ignore[:, cols]
    halfway = pitch.length / 2.0
    left = ((x < halfway - line_tol) & counted).sum(axis=1)
    right = ((x > halfway + line_tol) & counted).sum(axis=1)
    return visible.sum(axis=1), left, right, np.where(left > right, -1, 1)


def kickoff_setup(match, pitch, p):
    """Per frame: (is_formation, home_side, kicking_team, ball_at_centre, intruders_h, intruders_a).

    Law 8 happens in two steps, and the detector follows them:
      1. formation -- both teams in their own halves (players inside the circle not counted for
         the sides), at most KO_MAX_INTRUDERS of ONE team inside the circle (the non-kicking side;
         the tolerance covers a referee or a duplicate assigned to a team by the tracker)
      2. kicker -- a player of one team reaches the spot, often only in the last second
    is_formation is step 1 only; kicking_team (HOME/AWAY/-1) is step 2 per frame, and the
    kick-off is attributed over the whole formation run (kickoffs()).
    """
    cx, cy = pitch.centre
    dist = np.nan_to_num(np.hypot(match.px - cx, match.py - cy), nan=np.inf)   # (n, n_slots)
    in_circle = dist < pitch.centre_radius
    inside_strict = dist < pitch.centre_radius - p.KO_CIRCLE_TOL
    at_spot = dist <= p.KO_KICKER_R

    n_h, left_h, right_h, side_h = _team_sides(match, pitch, HOME, p.KO_LINE_TOL, in_circle)
    n_a, left_a, right_a, side_a = _team_sides(match, pitch, AWAY, p.KO_LINE_TOL, in_circle)
    enough = (n_h >= p.KO_MIN_PLAYERS) & (n_a >= p.KO_MIN_PLAYERS)
    own_halves = ((np.minimum(left_h, right_h) <= p.KO_MAX_WRONG_SIDE)
                  & (np.minimum(left_a, right_a) <= p.KO_MAX_WRONG_SIDE)
                  & (side_h != side_a))

    home_cols, away_cols = match.team_of_slot == HOME, match.team_of_slot == AWAY
    intr_h = inside_strict[:, home_cols].sum(axis=1)
    intr_a = inside_strict[:, away_cols].sum(axis=1)
    formation = enough & own_halves & (np.minimum(intr_h, intr_a) <= p.KO_MAX_INTRUDERS)

    spot_h, spot_a = at_spot[:, home_cols].any(axis=1), at_spot[:, away_cols].any(axis=1)
    # kicker: the only team with a player at the spot. Opponents in the circle are checked once,
    # over the whole formation (kickoffs()), not frame by frame.
    kicking = np.where(spot_h & ~spot_a, HOME, np.where(spot_a & ~spot_h, AWAY, -1))

    ball_at_centre = np.nan_to_num(pitch.dist_to_centre(match.ball_x, match.ball_y),
                                   nan=np.inf) <= p.KO_CENTRE_R
    return formation, side_h, kicking, ball_at_centre, intr_h, intr_a


def kickoff_setup_mask(match, pitch, p):
    """(formation mask, home_side)."""
    out = kickoff_setup(match, pitch, p)
    return out[0], out[1]


def kickoffs(match, pitch, p, keep_rejected=False):
    """[{start, kick, home_side, half, kicker, kicker_s, ball_seen, st_tag, first_of_half}].

    A formation becomes a kick-off only if a kicker reached the spot and the OTHER team kept
    (median over the formation) at most KO_MAX_INTRUDERS players inside the circle. With
    keep_rejected=True the rejected formations are returned too, with a 'rejected' reason,
    for the diagnosis.

    start      first frame of the formation
    kick       first frame after it (the formation breaks when play starts)
    home_side  -1 if the home team stands in the left half (x < halfway), +1 otherwise
    kicker     team seen at the spot during the formation (majority of the frames where exactly
               one team is there), -1 if nobody reached the spot
    kicker_s   seconds with a kicker at the spot
    ball_seen  ball seen at the centre during the formation or within 1 s after the kick
    st_tag     Smart Tagging kick-off type overlapping [start - 5 s, kick + 5 s], '' if none
    rescued_by_st  formation and kicker found in the ATD but more than KO_MAX_INTRUDERS
               opponents in the circle (players on the edge + tracking error): accepted only
               because a Smart Tagging kick-off confirms it
    ball_after_m / ball_after_s  distance from the centre (and delay) of the first ball seen
               within KO_AFTER_WIN after the kick -- near the centre if the kick-off was played;
               far away suggests a formation that ended without a kick-off (e.g. goal overturned)
    first_of_half  first accepted kick-off of the half AND within KO_HALF_START_WIN of the
               half's first graded frame
    half       half of the kick frame; a kick up to 1 s before a half's first graded frame
               belongs to that half (the 1st half is graded from the kick-off itself, FAQ.md).
               Formations outside the halves (warm-up, line-ups) get half 0 and are dropped.
    """
    formation, side_h, kicking, ball_at_centre, intr_h, intr_a = kickoff_setup(match, pitch, p)
    fps = match.fps
    min_len = int(round(p.T_KO_SETUP * fps))
    gap = int(round(0.5 * fps))                       # bridge tracking flicker inside a formation
    st_kickoffs = [t for t in match.smart_tagging if t.get("type", "").startswith("kick off")]

    merged = []
    for start, end in runs(formation):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    found = []
    for start, end in merged:
        if end - start < min_len:
            continue
        kick = int(min(end, match.n - 1))
        half = _half_of_kick(match, kick)
        if half == 0:
            continue
        kickers = kicking[start:end][kicking[start:end] >= 0]
        lo, hi = start - int(5 * fps), kick + int(5 * fps)
        tags = [t["type"] for t in st_kickoffs if t["start_frame"] < hi and t["end_frame"] > lo]
        kicker = int(np.bincount(kickers).argmax()) if kickers.size else -1
        rejected = ""
        rescued = False
        if kicker < 0:
            rejected = "no_kicker"
        else:
            other_intruders = (intr_a if kicker == HOME else intr_h)[start:end]
            if np.median(other_intruders) > p.KO_MAX_INTRUDERS:
                if tags:                     # formation + kicker from the ATD, confirmed by the ST
                    rescued = True
                else:
                    rejected = "opponents_in_circle"
        if rejected and not keep_rejected:
            continue
        after = np.flatnonzero(match.ball_visible[kick:kick + int(p.KO_AFTER_WIN * fps)])
        ball_after_m = round(float(pitch.dist_to_centre(match.ball_x[kick + after[0]],
                                                        match.ball_y[kick + after[0]])), 1) \
            if after.size else None
        found.append({
            "start": int(start), "kick": kick,
            "home_side": int(np.median(side_h[start:end])),
            "half": half,
            "kicker": kicker,
            "kicker_s": round(kickers.size / fps, 2),
            "ball_seen": bool(ball_at_centre[start:min(match.n, kick + int(fps))].any()),
            "st_tag": tags[0] if tags else "",
            "rescued_by_st": rescued,
            "ball_after_m": ball_after_m,
            "ball_after_s": round(after[0] / fps, 2) if after.size else None,
            "rejected": rejected,
        })

    # The kick-off that starts a half: the first accepted one, and only if it comes within
    # KO_HALF_START_WIN of the half's first graded frame. A later first detection (the real one
    # missed) is NOT promoted -- it would hide a goal.
    half_start = {h: int(np.flatnonzero(match.half == h)[0]) for h in (1, 2)
                  if (match.half == h).any()}
    seen_halves = set()
    for ko in found:
        early = ko["half"] in half_start and \
            ko["kick"] - half_start[ko["half"]] <= p.KO_HALF_START_WIN * fps
        ko["first_of_half"] = not ko["rejected"] and early and ko["half"] not in seen_halves
        if not ko["rejected"]:
            seen_halves.add(ko["half"])
    return found


def _half_of_kick(match, frame, tolerance_s=1.0):
    if match.half[frame] > 0:
        return int(match.half[frame])
    ahead = match.half[frame:frame + int(tolerance_s * match.fps) + 1]
    nonzero = ahead[ahead > 0]
    return int(nonzero[0]) if nonzero.size else 0


def _nearest_team(match, frame):
    if frame >= match.n or not match.ball_visible[frame]:
        frame = max(0, frame - 1)
    if not match.ball_visible[frame]:
        return -1
    d = np.hypot(match.px[frame] - match.ball_x[frame], match.py[frame] - match.ball_y[frame])
    if np.all(np.isnan(d)):
        return -1
    return int(match.team_of_slot[int(np.nanargmin(d))])


def goalkeeper_side(match, pitch, half):
    """Own goal of each team in a half, from where its deepest player stands: -1 (x=0) / +1 (x=L).

    A goalkeeper spends the half in front of their own goal, so over a whole half the median
    distance of a team's deepest player from its own goal line is small, and from the other one
    is not. No threshold: the comparison of two medians decides. Returns {team: side} plus the
    two medians per team for the report; None for a team if its sides are not distinguishable.
    """
    frames = np.flatnonzero(match.half == half)
    out, detail = {}, {}
    for team in (HOME, AWAY):
        x = match.px[frames][:, match.team_of_slot == team]
        has = ~np.all(np.isnan(x), axis=1)
        if not has.any():
            out[team] = None
            continue
        to_left = np.nanmin(x[has], axis=1)                   # deepest player towards x = 0
        to_right = pitch.length - np.nanmax(x[has], axis=1)   # deepest player towards x = L
        med_left, med_right = float(np.median(to_left)), float(np.median(to_right))
        detail[team] = (round(med_left, 1), round(med_right, 1))
        out[team] = -1 if med_left < med_right else (1 if med_right < med_left else None)
    return out, detail


def attack_directions(match, pitch, p, kickoff_list=None):
    """{(team, half): +1 / -1} -- +1 means the team attacks the goal at x = length.

    Methods, in order of preference:
      kickoff     first kick-off of the half (teams in their own halves)
      goalkeeper  where each team's deepest player stands over the half (goalkeeper_side)
      centroid    team centroids in the first K_DIR seconds of the half (last resort)
    The method used, and the goalkeeper check for every half, are returned for the report.
    """
    kickoff_list = kickoff_list if kickoff_list is not None else kickoffs(match, pitch, p)
    attack_dir, method = {}, {}
    for half in (1, 2):
        gk, detail = goalkeeper_side(match, pitch, half)
        gk_home = None
        if gk.get(HOME) is not None and gk.get(AWAY) is not None and gk[HOME] != gk[AWAY]:
            gk_home = -gk[HOME]                      # own goal at x=0 -> attacks x=L (+1)
        first = next((ko for ko in kickoff_list if ko["first_of_half"] and ko["half"] == half),
                     None)
        if first is not None:
            home_attacks, used = -first["home_side"], "kickoff"   # standing left -> attacks +1
        elif gk_home is not None:
            home_attacks, used = gk_home, "goalkeeper"
        else:
            frames = np.flatnonzero(match.half == half)[:int(p.K_DIR * match.fps)]
            if frames.size == 0:
                continue
            cx = [np.nanmean(match.px[frames][:, match.team_of_slot == t]) for t in (HOME, AWAY)]
            home_attacks, used = (1 if cx[0] < cx[1] else -1), "centroid"
        method[half] = {"used": used, "goalkeeper_agrees": None if gk_home is None
                        else gk_home == home_attacks,
                        "deepest_player_median_m(left,right)": detail}
        attack_dir[(HOME, half)] = home_attacks
        attack_dir[(AWAY, half)] = -home_attacks
    return attack_dir, method
