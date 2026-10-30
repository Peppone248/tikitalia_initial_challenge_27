"""Events: facts with a start and an end, from thresholds applied to the features over time
(NOMENCLATURA 7). Missing in the team code (STATO_LAVORI 3) and implemented here:

    control_episodes   a player of one team "attached" to the ball (REGOLE 2.4)
    pass_chains        consecutive control episodes of the same team, receptions by another player
    restart_kicks      the ball starts moving after being held still (end of a D)

Every threshold comes from params (params_hmm.yaml / team params.yaml). Strengths use the
NoiseModel (noise.py): z = margin / sigma with sigma = MAD of the feature.
"""

import numpy as np

from tikitalia.preprocessing import HOME, AWAY, runs

TEAMS = (HOME, AWAY)


def merge_runs(run_list, max_gap):
    merged = []
    for start, end in run_list:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def in_pitch(match, pitch):
    """Ball visible and inside the pitch (or on the line)."""
    d = pitch.dist_outside(match.ball_x, match.ball_y)
    return match.ball_visible & (np.nan_to_num(d, nan=np.inf) <= 0.0)


def control_condition(match, feats, pitch, p, noise, team):
    """Per frame: (condition, z) of "a player of `team` controls the ball".

    nearest[team] <= R_CONTROL              (z on nearest_dist)
    nearest[other] > nearest[team]           (not contested; z on the gap, sigma*sqrt(2))
    rel_speed[team] <= V_REL_CONTROL         (ball follows the player; NaN rel_speed = unknown,
                                              the condition is not blocked by a missing speed)
    ball inside the pitch
    """
    other = 1 - team
    d = feats.nearest_dist[:, team]
    d_o = np.nan_to_num(feats.nearest_dist[:, other], nan=np.inf)
    rel = feats.rel_speed[:, team]
    z_d = noise.z(p.R_CONTROL - d, "nearest_dist")
    gap = np.minimum(d_o - d, 3.0 * p.R_CONTROL)          # a far opponent = saturated anyway
    z_g = gap / (noise.sigma("nearest_dist") * np.sqrt(2.0))
    z_r = np.where(np.isnan(rel), np.inf, noise.z(p.V_REL_CONTROL - np.nan_to_num(rel), "rel_speed"))
    cond = in_pitch(match, pitch) & (np.nan_to_num(z_d, nan=-1) > 0) & (z_g > 0) & (z_r > 0)
    z = np.minimum.reduce([np.nan_to_num(z_d, nan=-1.0), z_g, z_r])
    return cond, z


def control_episodes(match, feats, pitch, p, noise):
    """[{team, start, end, slot, z (per frame), dur_s, x0, y0, x1, y1}] sorted by start.

    Runs of the control condition, gaps up to CONTROL_MERGE_GAP bridged (flicker of the nearest
    player), kept if at least T_TOUCH long (a touch). Rules decide how long is "control".
    """
    fps = match.fps
    out = []
    for team in TEAMS:
        cond, z = control_condition(match, feats, pitch, p, noise, team)
        for start, end in merge_runs(runs(cond), int(round(p.CONTROL_MERGE_GAP * fps))):
            if end - start < max(1, int(round(p.T_TOUCH * fps))):
                continue
            slots = feats.nearest_slot[start:end, team]
            slots = slots[slots >= 0]
            slot = int(np.bincount(slots).argmax()) if slots.size else -1
            out.append({
                "team": team, "start": int(start), "end": int(end), "slot": slot,
                "z": np.where(cond[start:end], z[start:end], 0.0),
                "dur_s": (end - start) / fps,
                "x0": float(feats.nearest_x[start, team]), "y0": float(feats.nearest_y[start, team]),
                "x1": float(feats.nearest_x[end - 1, team]),
                "y1": float(feats.nearest_y[end - 1, team]),
            })
    out.sort(key=lambda e: e["start"])
    return out


def pass_chains(episodes, match, p):
    """Chains of control episodes of ONE team, each reception by another player.

    Consecutive episodes (in time, any team) e1 -> e2 with the same team, e2 starting within
    T_PASS_MAX after e1 ends, and the receiver elsewhere: different slot or e2 starts more than
    D_PASS_MIN from where e1 ended. An episode of the other team in between breaks the chain
    (it is the next one in time). Returns [{team, start, end, receptions, episodes}].
    """
    fps = match.fps
    max_gap = int(round(p.T_PASS_MAX * fps))
    chains, cur = [], None
    for prev, nxt in zip(episodes, episodes[1:] + [None]):
        if cur is None:
            cur = {"team": prev["team"], "start": prev["start"], "end": prev["end"],
                   "receptions": 0, "episodes": [prev]}
        linked = False
        if nxt is not None and nxt["team"] == prev["team"] and \
                0 <= nxt["start"] - prev["end"] <= max_gap and \
                match.half[prev["end"] - 1] == match.half[nxt["start"]]:
            moved = np.hypot(nxt["x0"] - prev["x1"], nxt["y0"] - prev["y1"])
            other_player = (nxt["slot"] != prev["slot"]) or (np.nan_to_num(moved) > p.D_PASS_MIN)
            if other_player:
                cur["receptions"] += 1
                cur["end"] = nxt["end"]
                cur["episodes"].append(nxt)
                linked = True
            elif nxt["start"] - prev["end"] <= int(round(p.CONTROL_MERGE_GAP * fps)) * 3:
                cur["end"] = nxt["end"]                     # same player again: same episode
                cur["episodes"].append(nxt)
                linked = True
        if not linked:
            chains.append(cur)
            cur = None
    return [c for c in chains if c["receptions"] > 0]


def restart_kicks(match, feats, p):
    """[{frame, kind, strength}] -- the ball starts moving after being held still.

    still_kick : ball still for >= T_STILL_RESTART, then visible and faster than V_BALL_STILL
                 within the next frame, and accelerating by more than ACC_KICK (estimated as a
                 high quantile of the ball acceleration, no labels)
    Kick-offs are added by the caller (they come from events of the team code).
    """
    fps = match.fps
    need = p.T_STILL_RESTART
    still = match.ball_still_for
    speed = np.nan_to_num(match.ball_speed, nan=0.0)
    acc = np.nan_to_num(feats.ball_acc, nan=0.0)
    out = []
    ends = np.flatnonzero((still[:-1] >= need) & (still[1:] == 0))
    look = max(2, int(round(0.2 * fps)))
    for f in ends:
        f1 = f + 1
        seg = slice(f1, min(match.n, f1 + look))
        if not match.ball_visible[seg].any():
            continue
        if speed[seg].max() <= p.V_BALL_STILL:
            continue
        peak_acc = acc[max(0, f - 1):min(match.n, f1 + look)].max()
        strength = float(np.clip(peak_acc / p.ACC_KICK, 0.0, 1.0))
        out.append({"frame": int(f1), "kind": "still_kick", "strength": strength,
                    "still_s": round(float(still[f]), 2)})
    return out


def estimate_acc_kick(match, feats, params):
    """ACC_KICK (origin: distribution) = ACC_KICK_QUANTILE of |a| on visible graded frames."""
    m = match.ball_visible & match.graded & np.isfinite(feats.ball_acc)
    value = float(np.quantile(feats.ball_acc[m], params.ACC_KICK_QUANTILE)) if m.any() else 20.0
    params.set_estimated("ACC_KICK", value)
    return value
