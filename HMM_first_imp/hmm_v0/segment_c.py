"""Approach C (v2): segment first, then assign the team. No NN, no HMM state machine.

    1. DEAD BALL     D mask from the D rules (or from the v1 HMM, dead.source: hmm), cleaned:
                     gaps between D spells shorter than close_gap_s are closed, D spells shorter
                     than min_dead_s dropped, alive stretches shorter than min_alive_s between two
                     D become D.
    2. BOUNDARIES    inside every alive stretch, possession may change ONLY at a candidate frame:
                     the start of a contact (control episode, any length >= T_TOUCH), of a
                     turnover (T-*), of a restart (R-*), or the first frame the ball is seen again
                     after a gap >= T_X_MIN. Changes are therefore dated by an observable event.
    3. ASSIGNMENT    every elementary segment (between two candidates) gets its summed team
                     evidence  S = sum_f e(f) / fps  (seconds of full-strength evidence; > 0 home):
                         e(f) = sum_r w_r * vote_r(f) * s_r(f)     vote = +1 H / -1 A
                              + w_near * nearest_team_sign(f) * clip(gap / near_gap_m, 0, 1)
                     and a 2-label dynamic programme chooses H/A for all segments of the stretch:
                         minimise  sum_i -S_i * sign(L_i)  +  lambda * #changes
                     lambda (change_penalty_s) = how many seconds of clear evidence a new team
                     needs to be credited with a possession: the one knob of the segment count.

Why: spellF1 judges segments, not frames. Boundaries come from events (contacts, restarts),
teams from evidence pooled over a whole segment, which is robust to the ball being unseen for
seconds (the evidence of the visible frames and of the slow signals carries the segment).
Everything is in config_c/segment.yaml.
"""

import numpy as np

from tikitalia.preprocessing import HOME, AWAY, runs
from . import votes as V

SIGN = {"H": 1.0, "A": -1.0}


def _clean_dead(dead, half, fps, cfg):
    """Close short gaps, drop short D spells, absorb short alive islands; per half."""
    d = dead.copy()
    close = int(round(cfg["close_gap_s"] * fps))
    for s, e in runs(~d):
        if e - s <= close and s > 0 and e < d.size and d[s - 1] and d[e] and half[s - 1] == half[e]:
            d[s:e] = True
    min_dead = int(round(cfg["min_dead_s"] * fps))
    for s, e in runs(d):
        if e - s < min_dead:
            d[s:e] = False
    min_alive = int(round(cfg["min_alive_s"] * fps))
    for s, e in runs(~d):
        if e - s < min_alive and s > 0 and e < d.size and d[s - 1] and d[e]:
            d[s:e] = True
    return d


def st_dead(match, context, cfg_st):
    """D spells anchored on the Smart Tagging set pieces (v3).

    A SET PIECES tag marks a restart that is about to happen: on BAR-ATH the first fast ball
    after the tag start comes +3..+10 s later for corners and mostly within +10 s for throw-ins
    (tracking only, REPORT_C_v3). So the ball is dead from before the tag start until the restart:
        start = tag start - pre_s
        end   = first sign of play after the tag start, within lead_max_s: the ball seen moving
                faster than kick_speed, or the start of a contact; none seen -> tag start +
                default_lead_s
    Works with the ball unseen, which is the case the rest of the pipeline misses."""
    fps, n = match.fps, match.n
    starts = np.array(sorted(ep["start"] for ep in context["control_episodes"]), dtype=int)
    fast = match.ball_visible & (np.nan_to_num(match.ball_speed) > cfg_st["kick_speed"])
    types = set(t.lower() for t in cfg_st["types"])
    out = np.zeros(n, bool)
    spans = []
    for t in match.smart_tagging:
        if t["code"].upper() != "SET PIECES" or (t.get("type") or "").lower() not in types:
            continue
        s = int(t["start_frame"])
        lo, hi = s, min(n, s + int(round(cfg_st["lead_max_s"] * fps)))
        if lo >= n:
            continue
        cands = []
        f = np.flatnonzero(fast[lo:hi])
        if f.size:
            cands.append(lo + int(f[0]))
        k = np.searchsorted(starts, lo)
        if k < starts.size and starts[k] < hi:
            cands.append(int(starts[k]))
        end = min(cands) if cands else min(n, s + int(round(cfg_st["default_lead_s"] * fps)))
        start = max(0, s - int(round(cfg_st["pre_s"] * fps)))
        back = cfg_st.get("back_to_last_action_s", 0)
        if back:
            # the ball went out just after the last action seen before the tag (end of a contact
            # or the ball seen fast), at most `back` s earlier; none seen -> tag start - pre_s
            lo_b = max(0, s - int(round(back * fps)))
            last = [ep["end"] for ep in context["control_episodes"] if lo_b <= ep["end"] <= s]
            fb = np.flatnonzero(fast[lo_b:s])
            if fb.size:
                last.append(lo_b + int(fb[-1]) + 1)
            if last:
                start = int(max(last))
        if end > start:
            out[start:end] = True
            spans.append((start, end))
    return out, spans


def dead_mask(match, rules, outputs, E, cfg_seg, hmm_labels=None, context=None):
    c = cfg_seg["dead"]
    if c["source"] == "hmm":
        raw = hmm_labels == "D"
    else:
        raw = E[:, V.CLASSES.index("D")] >= c["min_evidence"]
    st = c.get("smart_tagging", {})
    if st.get("enabled") and context is not None:
        raw = raw | st_dead(match, context, st)[0]
    return _clean_dead(raw & match.graded, match.half, match.fps, c)


def team_evidence(match, context, outputs, cfg_seg):
    """(n,) team evidence per frame, > 0 = home."""
    n, fps = match.n, match.fps
    w = cfg_seg["weights"]
    e = np.zeros(n)
    # contacts: every control episode, any length, signed by team, weighted by its strength
    for ep in context["control_episodes"]:
        s = np.clip(np.nan_to_num(ep["z"]) / 3.0, 0, 1)
        e[ep["start"]:ep["end"]] += w["contact"] * (1.0 if ep["team"] == HOME else -1.0) * s
    # rule votes
    for rid, wr in w["rules"].items():
        out = outputs.get(rid)
        if out is None or wr == 0:
            continue
        sign = np.where(out.label == "H", 1.0, np.where(out.label == "A", -1.0, 0.0))
        e += wr * sign * out.strength
    # nearest player, weighted by how uncontested (gap to the other team's nearest)
    f = context["features"]
    near = np.where(f.nearest_team == HOME, 1.0, np.where(f.nearest_team == AWAY, -1.0, 0.0))
    dmin = np.nanmin(np.where(np.isnan(f.nearest_dist), np.inf, f.nearest_dist), axis=1)
    ok = match.ball_visible & (dmin <= cfg_seg["near_max_m"])
    gap = np.clip(np.nan_to_num(f.nearest_gap) / cfg_seg["near_gap_m"], 0, 1)
    e += np.where(ok, w["nearest"] * near * gap, 0.0)
    return e


def candidates(match, context, outputs, cfg_seg):
    fps = match.fps
    pts = set(ep["start"] for ep in context["control_episodes"])
    for rid in cfg_seg["boundary_rules"]:
        out = outputs.get(rid)
        if out is not None:
            pts.update(int(ep["start"]) for ep in out.episodes if "start" in ep)
    need = int(round(cfg_seg["reappear_after_s"] * fps))
    for s, e in runs(~match.ball_visible):
        if e - s >= need and e < match.n:
            pts.add(int(e))
    return np.array(sorted(pts), dtype=int)


def assign(stretch_start, stretch_end, cands, ev, fps, lam, first_bias=0.0):
    """Labels H/A for [stretch_start, stretch_end): elementary segments cut at the candidates,
    2-label DP with change penalty lam (seconds of evidence)."""
    cuts = cands[(cands > stretch_start) & (cands < stretch_end)]
    bounds = np.concatenate(([stretch_start], cuts, [stretch_end]))
    csum = np.concatenate(([0.0], np.cumsum(ev[stretch_start:stretch_end]))) / fps
    S = csum[bounds[1:] - stretch_start] - csum[bounds[:-1] - stretch_start]
    m = S.size
    # cost of labelling segment i home / away
    cost = np.stack([-S, S], axis=1)                 # col 0 = H, col 1 = A
    cost[0] += np.array([-first_bias, first_bias])
    acc = cost[0].copy()
    back = np.zeros((m, 2), dtype=np.int8)
    for i in range(1, m):
        stay = acc
        switch = acc[::-1] + lam
        back[i] = np.where(stay <= switch, [0, 1], [1, 0])
        acc = np.minimum(stay, switch) + cost[i]
    lab = np.empty(m, dtype=np.int8)
    lab[-1] = int(np.argmin(acc))
    for i in range(m - 1, 0, -1):
        lab[i - 1] = back[i, lab[i]]
    out = np.empty(stretch_end - stretch_start, dtype="<U1")
    for i in range(m):
        out[bounds[i] - stretch_start:bounds[i + 1] - stretch_start] = "H" if lab[i] == 0 else "A"
    return out, m


# ------------------------------------------------------------------------------------------------
# v3: possession exactly as CHALLENGE.md words it
#
#   "A change of possession occurs when a player from the opposing team gains CONTROLLED
#    possession. While the ball is contested or loose, possession stays with the last team."
#   Proxies given: a player keeps the ball for a few seconds; a team completes passes.
#   Not a change: a touch without control, a deflection, an aerial duel without control.
#
#   -> a team GAINS possession at the start of one of its contacts that is CONFIRMED:
#        (a) the contact lasts >= T_CONTROL                         (keeps the ball), or
#        (b) the next contact in time is by a team-mate, within T_PASS_MAX (a completed pass);
#      or at a restart (R-TOUCH, R-KO: whoever restarts has the ball).
#   -> anything else (touch, deflection, duel, ball unseen) changes nothing: the last team keeps it.
#   -> goalkeeper: a contact of a team's deepest player inside its own penalty area is control
#      at once (hands); from it until the next contact after he releases the ball (the clearance
#      in flight included) possession is his team's and other contacts are ignored.
# ------------------------------------------------------------------------------------------------

def confirmed_gains(episodes, match, p, cfg_pos=None):
    """[(frame, team)] start of every confirmed contact (rule (a) or (b) above)."""
    fps = match.fps
    cfg_pos = cfg_pos or {}
    t_ctrl = int(round(cfg_pos.get("min_control_s", p.T_CONTROL) * fps))
    t_pass = int(round(cfg_pos.get("pass_link_s", p.T_PASS_MAX) * fps))
    out = []
    for i, ep in enumerate(episodes):
        ok = ep["end"] - ep["start"] >= t_ctrl
        if not ok and i + 1 < len(episodes):
            nxt = episodes[i + 1]
            ok = (nxt["team"] == ep["team"] and 0 <= nxt["start"] - ep["end"] <= t_pass
                  and match.half[nxt["start"]] == match.half[ep["start"]])
        if ok:
            out.append((ep["start"], ep["team"]))
    return out


def goalkeeper_windows(match, pitch, context, cfg_gk):
    """[(start, end, team)]: a team's deepest player controls the ball in his own box ->
    the window runs to the first contact that starts after he has released it."""
    eps = context["control_episodes"]
    attack = context["attack_dir"]
    merge = int(round(cfg_gk["merge_s"] * match.fps))
    L, W = pitch.length, pitch.width
    wins = []
    i = 0
    while i < len(eps):
        ep = eps[i]
        t, h = ep["team"], match.half[ep["start"]]
        ad = attack.get((t, h))
        if ad is None or ep["slot"] < 0 or not np.isfinite(ep["x0"]):
            i += 1
            continue
        goal_x = 0.0 if ad == 1 else L
        in_box = abs(ep["x0"] - goal_x) <= pitch.box_depth and \
            abs(ep["y0"] - W / 2.0) <= pitch.box_half_width
        mates = match.team_of_slot == t
        depth = np.abs(match.px[ep["start"], mates] - goal_x)
        if not in_box or not np.any(np.isfinite(depth)):
            i += 1
            continue
        deepest = np.flatnonzero(mates)[int(np.nanargmin(depth))]
        if deepest != ep["slot"]:
            i += 1
            continue
        hold_end, j = ep["end"], i + 1
        while j < len(eps) and eps[j]["start"] <= hold_end + merge:   # his hold, opponents
            if eps[j]["team"] == t and eps[j]["slot"] == ep["slot"]:   # pressing him ignored
                hold_end = max(hold_end, eps[j]["end"])
            j += 1
        if hold_end - ep["start"] < int(round(cfg_gk.get("min_hold_s", 0.0) * match.fps)):
            i += 1                                  # a touch at his feet, not a hold
            continue
        nxt = next((e for e in eps[j - 1:] if e["start"] >= hold_end), None)
        end = nxt["start"] if nxt is not None else match.n
        wins.append((ep["start"], end, t))
        while i < len(eps) and eps[i]["start"] < end:
            i += 1
    return wins


def possession_events(match, pitch, context, outputs, p, cfg_seg):
    """Sorted [(frame, team)] of possession gains, after the goalkeeper windows."""
    eps = context["control_episodes"]
    gains = confirmed_gains(eps, match, p, cfg_seg["possession"])
    for rid in cfg_seg["possession"]["restart_rules"]:
        out = outputs.get(rid)
        if out is None:
            continue
        for ep in out.episodes:
            s = int(ep.get("start", -1))
            if 0 <= s < match.n and out.label[s] in ("H", "A"):
                gains.append((s, HOME if out.label[s] == "H" else AWAY))
    wins = goalkeeper_windows(match, pitch, context, cfg_seg["goalkeeper"]) \
        if cfg_seg["goalkeeper"]["enabled"] else []
    if wins:
        keep = np.ones(len(gains), bool)
        for k, (f, _t) in enumerate(gains):
            keep[k] = not any(s <= f < e for s, e, _ in wins)
        gains = [g for g, k in zip(gains, keep) if k] + [(s, t) for s, _e, t in wins]
    gains.sort()
    return gains, wins


def assign_control(s, e, gains, fallback):
    """Labels for [s, e): the team of the last gain; before the first gain, the first gain's
    team (a restart has a clear possessor, CHALLENGE.md)."""
    lab = np.empty(e - s, dtype="<U1")
    inside = [(f, t) for f, t in gains if s <= f < e]
    if not inside:
        lab[:] = fallback
        return lab, 0
    cur = "H" if inside[0][1] == HOME else "A"
    pos = s
    changes = 0
    for f, t in inside:
        team = "H" if t == HOME else "A"
        lab[pos - s:f - s] = cur
        if team != cur:
            changes += 1
        cur, pos = team, f
    lab[pos - s:] = cur
    return lab, changes


def decide(match, context, rules, outputs, E, cfg_seg, hmm_labels=None, p=None, pitch=None):
    fps = match.fps
    dead = dead_mask(match, rules, outputs, E, cfg_seg, hmm_labels, context)
    labels = np.full(match.n, "X", dtype="<U1")
    labels[dead] = "D"
    mode = cfg_seg.get("possession", {}).get("mode", "evidence")
    info = {"possession_mode": mode, "dead_source": cfg_seg["dead"]["source"],
            "dead_share_%": round(100.0 * float(dead[match.graded].mean()), 2)}
    if mode == "control":
        gains, wins = possession_events(match, pitch, context, outputs, p, cfg_seg)
        n_stretch, n_changes, n_empty = 0, 0, 0
        ev_all = team_evidence(match, context, outputs, cfg_seg)
        for h in (1, 2):
            for s, e in runs((match.half == h) & ~dead):
                fb = "H" if ev_all[s:e].sum() >= 0 else "A"
                lab, c = assign_control(s, e, gains, fb)
                labels[s:e] = lab
                n_stretch += 1
                n_changes += c
                n_empty += not any(s <= f < e for f, _ in gains)
        info.update({"alive_stretches": n_stretch, "possession_gains": len(gains),
                     "changes_in_play": n_changes, "stretches_without_gain": n_empty,
                     "goalkeeper_windows": len(wins)})
        return labels, info
    ev = team_evidence(match, context, outputs, cfg_seg)
    cands = candidates(match, context, outputs, cfg_seg)
    gk = cfg_seg.get("goalkeeper", {})
    wins = goalkeeper_windows(match, pitch, context, gk) if gk.get("enabled") else []
    if wins:
        # the goalkeeper's team owns the window: one segment, strong evidence, no cut inside
        inside = np.zeros(match.n, bool)
        for s, e, t in wins:
            ev[s:e] = (1.0 if t == HOME else -1.0) * float(gk.get("weight", 2.0))
            inside[s + 1:e] = True
        cands = np.array(sorted(set(cands[~inside[np.clip(cands, 0, match.n - 1)]].tolist())
                                | {s for s, _e, _t in wins} | {e for _s, e, _t in wins
                                                               if e < match.n}), dtype=int)
    lam = float(cfg_seg["change_penalty_s"])
    n_seg, n_stretch = 0, 0
    for h in (1, 2):
        alive = (match.half == h) & ~dead
        for s, e in runs(alive):
            lab, m = assign(s, e, cands, ev, fps, lam)
            labels[s:e] = lab
            n_seg += m
            n_stretch += 1
    info.update({"alive_stretches": n_stretch, "elementary_segments": n_seg,
                 "candidates": int(cands.size), "change_penalty_s": lam,
                 "goalkeeper_windows": len(wins)})
    return labels, info
