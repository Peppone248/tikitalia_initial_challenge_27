"""Approach B, round 2: goalkeepers, placed ball, set-piece formations, earlier dead-ball start.

Every rule works on positions only (tracking or IDSSE in the same form): metres, origin at a
corner of the pitch, x along the length. All return frame masks; nothing reads labels.

    goalkeeper_team_fix   (1)  a goalkeeper-like fragment of a slot gets the team defending that goal
    placed_ball           (2)  ball seen still on the pitch for >= min_still_s, not held = dead
    decel_start           (3)  move the start of a dead span back over the collective deceleration
    formations            (4)  wall / corner / throw-in / goal kick (10) seen from the players
    formation_dead        (4)  formations -> dead spans, from the stoppage to the kick
"""

import warnings

import numpy as np

from tikitalia.preprocessing import runs

FPS = 25
PEN_DEPTH, PEN_HALF_W = 16.5, 20.16
GA_DEPTH, GA_HALF_W = 5.5, 9.16


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def player_speeds(px, py, fps=FPS):
    return np.hypot(np.gradient(px, axis=0), np.gradient(py, axis=0)) * fps


def ball_speed(bx, by, fps=FPS, smooth=5):
    from .baseline_b import moving_average
    x = moving_average(np.where(np.isnan(bx), 0, bx), smooth)
    y = moving_average(np.where(np.isnan(by), 0, by), smooth)
    v = np.hypot(np.gradient(x), np.gradient(y)) * fps
    seen = ~np.isnan(bx)
    ok = seen & np.r_[seen[1:], False] & np.r_[False, seen[:-1]]
    v[~ok] = np.nan
    return v


def defended_side(px, team, half):
    """{half: {team: 'left'|'right'}}: the side each team defends (its players' mean x)."""
    out = {}
    for h in (1, 2):
        f = half == h
        if not f.any():
            continue
        mx = [_nanmean(px[f][:, team == t], None) for t in (0, 1)]
        left0 = mx[0] < mx[1]
        out[h] = {0: "left" if left0 else "right", 1: "right" if left0 else "left"}
    return out


# -- (1) goalkeepers -------------------------------------------------------------------------------
def goalkeeper_team_fix(px, py, team, half, length, width, cfg):
    """A fragment of a slot (a visible run) that behaves like a goalkeeper of goal G -- inside G's
    penalty area for >= in_box_share of its frames and the deepest of all players towards G for
    >= deepest_share -- belongs to the team that defends G. When the slot's team says otherwise,
    the tracker has re-attached the slot to the other team's goalkeeper: the fragment is moved to
    a new column with the right team. Returns (px, py, team, list of fixes)."""
    px, py = px.copy(), py.copy()
    sides = defended_side(px, team, half)
    min_len = int(round(cfg["min_s"] * FPS))
    new_x, new_y, new_t, fixes = [], [], [], []
    for j in range(px.shape[1]):
        for s, e in runs(~np.isnan(px[:, j]) & (half > 0)):
            if e - s < min_len:
                continue
            h = int(np.bincount(half[s:e]).argmax())
            if h not in sides:
                continue
            x, y = px[s:e, j], py[s:e, j]
            for goal, xin in (("left", x), ("right", length - x)):
                inbox = (xin < PEN_DEPTH) & (np.abs(y - width / 2) < PEN_HALF_W)
                if inbox.mean() < cfg["in_box_share"]:
                    continue
                allx = px[s:e] if goal == "left" else length - px[s:e]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    deepest = xin <= np.nanmin(allx, axis=1) + 0.01
                if deepest.mean() < cfg["deepest_share"]:
                    continue
                owner = 0 if sides[h][0] == goal else 1
                if owner != team[j]:
                    col_x = np.full(px.shape[0], np.nan, np.float32)
                    col_y = np.full(px.shape[0], np.nan, np.float32)
                    col_x[s:e], col_y[s:e] = x, y
                    px[s:e, j] = np.nan
                    py[s:e, j] = np.nan
                    new_x.append(col_x)
                    new_y.append(col_y)
                    new_t.append(owner)
                    fixes.append((j, int(s), int(e), int(team[j]), owner))
                break
    if new_x:
        px = np.concatenate([px, np.stack(new_x, 1)], 1)
        py = np.concatenate([py, np.stack(new_y, 1)], 1)
        team = np.concatenate([np.asarray(team), np.array(new_t, dtype=np.asarray(team).dtype)])
    return px, py, team, fixes


# -- (2) placed ball -------------------------------------------------------------------------------
def placed_ball(bx, by, px, py, team, half, length, width, cfg):
    """The ball seen still (speed < v_still) for >= min_still_s inside the pitch is dead -- unless it
    is held: a player within hold_m of it for >= hold_share of the span inside his own penalty area
    (the goalkeeper with the ball in his hands: alive). The dead span starts where the ball was
    last seen moving before it stopped (at most back_s earlier: through a gap, the whistle came
    during the gap) and ends at the kick (first ball faster than kick_speed within lead_s)."""
    v = ball_speed(bx, by)
    still = (np.nan_to_num(v, nan=np.inf) < cfg["v_still"]) & (half > 0)
    inside = (bx > 0.5) & (bx < length - 0.5) & (by > 0.5) & (by < width - 0.5)
    still &= inside
    sides = defended_side(px, team, half)
    d = np.hypot(px - bx[:, None], py - by[:, None])
    near = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    jn = np.argmin(np.where(np.isnan(d), np.inf, d), axis=1)
    fast = np.nan_to_num(v) > cfg["kick_speed"]
    seen = ~np.isnan(bx)
    out = np.zeros(bx.size, bool)
    for s, e in runs(still):
        if e - s < int(round(cfg["min_still_s"] * FPS)):
            continue
        held = near[s:e] < cfg["hold_m"]
        if held.mean() >= cfg["hold_share"]:
            h = int(half[s])
            t = int(np.bincount(np.asarray(team)[jn[s:e][held]]).argmax())
            goal = sides.get(h, {}).get(t)
            xin = bx[s:e] if goal == "left" else length - bx[s:e]
            if np.nanmedian(xin) < PEN_DEPTH and np.nanmedian(np.abs(by[s:e] - width / 2)) < PEN_HALF_W:
                continue
        lo = max(0, s - int(round(cfg["back_s"] * FPS)))
        moving = np.flatnonzero(seen[lo:s] & ~still[lo:s] & (np.nan_to_num(v[lo:s]) > cfg["v_still"]))
        start = lo + int(moving[-1]) + 1 if moving.size else lo
        hi = min(bx.size, e + int(round(cfg["lead_s"] * FPS)))
        k = np.flatnonzero(fast[e:hi])
        end = e + int(k[0]) if k.size else e
        out[start:end] = True
    return out & (half > 0)


# -- (3) earlier start: the collective deceleration --------------------------------------------------
def decel_start(dead, mspeed, half, cfg, only=None):
    """Move the start of every dead span (or of the spans in `only`) back while the mean player
    speed was decreasing (players slowing down after the whistle), at most back_max_s, never
    across another span or a half boundary."""
    out = dead.copy()
    back = int(round(cfg["back_max_s"] * FPS))
    dv = np.gradient(mspeed)
    for s, e in runs(dead):
        if only is not None and not only[s:e].any():
            continue
        k = s
        while k > 0 and s - k < back and half[k - 1] == half[s] and not dead[k - 1] \
                and dv[k - 1] < -cfg["min_decel"]:
            k -= 1
        out[k:s] = True
    return out


# -- (4) set-piece formations ------------------------------------------------------------------------
def formations(px, py, team, half, length, width, cfg, bx=None, by=None):
    """Per-frame masks of four formations (positions only):
        wall      >= wall_n players of one team within wall_m of each other, all slower than slow_ms
        corner    a player within corner_m of a corner flag, slower than slow_ms
        throw_in  a player beyond a touchline by >= out_m (not near the corners)
        goal_kick a player in his own goal area, NO opponent in that penalty area and the ball SEEN
                  still inside the goal area (strict: with the ball unseen this formation is also
                  the goalkeeper holding the ball after a save, which is alive)
    """
    n, s = px.shape
    sp = player_speeds(px, py)
    slow = np.nan_to_num(sp, nan=np.inf) < cfg["slow_ms"]
    t = np.asarray(team)
    wall = np.zeros(n, bool)
    for tm in (0, 1):
        cols = np.flatnonzero(t == tm)
        X, Y, S = px[:, cols], py[:, cols], slow[:, cols]
        for a in range(0, n, 20000):
            x, y, sl = X[a:a + 20000], Y[a:a + 20000], S[a:a + 20000]
            dd = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
            close = (np.nan_to_num(dd, nan=np.inf) < cfg["wall_m"]) & sl[:, :, None] & sl[:, None, :]
            idx = np.arange(close.shape[1])
            close[:, idx, idx] = False
            wall[a:a + 20000] |= (((close.sum(2) + 1) >= cfg["wall_n"]) & sl).any(1)
    corners = [(0, 0), (0, width), (length, 0), (length, width)]
    corner = np.zeros(n, bool)
    for cx, cy in corners:
        corner |= ((np.hypot(px - cx, py - cy) < cfg["corner_m"]) & slow).any(1)
    out_m = cfg["out_m"]
    margin = cfg["corner_m"] + 3.0
    beyond = ((py < -out_m) | (py > width + out_m)) & (px > margin) & (px < length - margin)
    throw_in = np.nan_to_num(beyond, nan=False).any(1)
    sides = defended_side(px, t, half)
    goal_kick = np.zeros(n, bool)
    for h, sd in sides.items():
        f = half == h
        for tm in (0, 1):
            goal = sd[tm]
            X = px if goal == "left" else length - px
            in_ga = (X < GA_DEPTH) & (np.abs(py - width / 2) < GA_HALF_W)
            in_pa = (X < PEN_DEPTH) & (np.abs(py - width / 2) < PEN_HALF_W)
            own_ga = (in_ga[:, t == tm]).any(1)
            opp_pa = (in_pa[:, t != tm]).any(1)
            m = own_ga & ~opp_pa & f
            if bx is not None:
                bxx = bx if goal == "left" else length - bx
                bv = ball_speed(bx, by)
                ball_ok = (bxx < GA_DEPTH + 1) & (np.abs(by - width / 2) < GA_HALF_W + 1) \
                    & (np.nan_to_num(bv, nan=np.inf) < cfg.get("gk_ball_still_ms", 0.5))
                if not cfg.get("goal_kick_strict", True):
                    ball_ok |= np.isnan(bx)
                m &= ball_ok
            goal_kick |= m
    return {"wall": wall & (half > 0), "corner": corner & (half > 0),
            "throw_in": throw_in & (half > 0), "goal_kick": goal_kick & (half > 0)}


def formation_dead(mask, bx, by, mspeed, half, cfg):
    """Formation frames held for >= min_s -> dead from the stoppage to the kick. Start: back over
    the collective deceleration from the formation start (at most back_s); end: the first ball
    faster than kick_speed after the formation ends, within lead_s (else the formation end)."""
    v = ball_speed(bx, by)
    fast = np.nan_to_num(v) > cfg["kick_speed"]
    dv = np.gradient(mspeed)
    out = np.zeros(mask.size, bool)
    m = mask.copy()
    k = int(round(cfg.get("bridge_s", 1.0) * FPS))
    for s, e in runs(~m):
        if 0 < s and e < m.size and e - s < k and m[s - 1] and m[e]:
            m[s:e] = True
    for s, e in runs(m):
        if e - s < int(round(cfg["min_s"] * FPS)):
            continue
        a = s
        lim = max(0, s - int(round(cfg["back_s"] * FPS)))
        while a > lim and half[a - 1] == half[s] and not fast[a - 1] and \
                (dv[a - 1] < 0 or mspeed[a - 1] < cfg["walk_ms"]):
            a -= 1
        hi = min(m.size, e + int(round(cfg["lead_s"] * FPS)))
        kk = np.flatnonzero(fast[e:hi])
        end = e + int(kk[0]) if kk.size else e
        out[a:end] = True
    return out & (half > 0)


# -- the ball is out only when it leaves the pitch -----------------------------------------------------
def tag_starts(tags, types):
    """Start frames of the SET PIECES tags of the given types (lower case), from a Smart Tagging
    list as loaded (BAR-ATH: type in tags["Type"]; simulated IDSSE feed: type in "type")."""
    types = set(t.lower() for t in types)
    out = []
    for t in tags or []:
        typ = (t.get("type") or (t.get("tags") or {}).get("Type") or "").lower()
        if (t.get("code") or "").upper() == "SET PIECES" and typ in types:
            out.append(int(t["start_frame"]))
    return np.array(sorted(out), dtype=int)


def ball_out_start(dead, bx, by, half, length, width, cfg, fps=FPS, tags=None):
    """A dead span that the ball LEAVES the pitch in cannot start before the ball leaves it.

    Inside every dead span: if the ball is later seen outside the lines (by more than out_m), the
    frames before that first crossing where the ball is seen INSIDE the pitch (at least in_m from
    every line) and moving (faster than min_speed) are play: the dead span starts at the crossing.
    The rule of the game: the ball is out of play when it has wholly crossed a line. A span in which
    the ball is never seen outside (fouls, offsides) is left as it is. So is a span with a Smart
    Tagging restart of a type in skip_tag_types (free kick): there the stoppage is the whistle and
    the ball leaving the pitch is the ball kicked away after it."""
    skip = tag_starts(tags, cfg.get("skip_tag_types", ()))
    v = np.nan_to_num(ball_speed(bx, by), nan=0.0)
    seen = ~np.isnan(bx)
    inside = seen & (bx > cfg["in_m"]) & (bx < length - cfg["in_m"]) \
        & (by > cfg["in_m"]) & (by < width - cfg["in_m"])
    outside = seen & ((bx < -cfg["out_m"]) | (bx > length + cfg["out_m"])
                      | (by < -cfg["out_m"]) | (by > width + cfg["out_m"]))
    out = dead.copy()
    for s, e in runs(dead):
        o = np.flatnonzero(outside[s:e])
        if not o.size or ((skip >= s) & (skip < e)).any():
            continue
        cross = s + int(o[0])
        lo = s
        if cfg.get("continuous"):
            # only the ball seen without a break up to the crossing, at most approach_s before it,
            # and coming from at least min_approach_m inside the line it crosses
            a = cross
            lim = max(s, cross - int(round(cfg["approach_s"] * fps)))
            while a > lim and seen[a - 1]:
                a -= 1
            lo = a
            if lo >= cross:
                continue
            xs, ys = bx[lo:cross + 1], by[lo:cross + 1]
            dist = np.minimum.reduce([xs, length - xs, ys, width - ys])   # to the nearest line
            # the ball has been well inside the pitch on its way out (not just dribbled along the line)
            if not (dist.max() - dist[-1] > cfg.get("min_approach_m", 0.0)):
                continue
        playing = inside[lo:cross] & (v[lo:cross] > cfg["min_speed"])
        if not playing.any():
            continue
        first = lo + int(np.flatnonzero(playing)[0])
        last_play = lo + int(np.flatnonzero(playing)[-1]) + 1
        out[s if not cfg.get("continuous") else first:last_play] = False
    return out


# -- (R1, 01/11) the restart kick closes the dead span ---------------------------------------------
def tag_windows(tags, types):
    """(start, end) frames of the SET PIECES tags of the given types (lower case)."""
    types = set(t.lower() for t in types)
    out = []
    for t in tags or []:
        typ = (t.get("type") or (t.get("tags") or {}).get("Type") or "").lower()
        if (t.get("code") or "").upper() == "SET PIECES" and typ in types:
            out.append((int(t["start_frame"]), int(t["end_frame"])))
    return sorted(out)


def seen_speed(bx, by, m, fps=FPS):
    """(forward, backward) ball speed over m frames from the SEEN positions only (nan where any end
    is unseen). ball_speed smooths over gap edges and inflates there; a kick right after the tracker
    finds the ball again must be measured on the positions themselves."""
    n = bx.size
    fwd = np.full(n, np.nan)
    bwd = np.full(n, np.nan)
    if n > m:
        d = np.hypot(bx[m:] - bx[:-m], by[m:] - by[:-m]) / m * fps
        fwd[:-m] = d
        bwd[m:] = d
    return fwd, bwd


def restart_kicks(bx, by, length, width, cfg, fps=FPS):
    """(n,) bool: kicks from rest, dated to the start of the kick. A kick = the ball seen leaving
    faster than kick_speed (over measure_s), at most margin_m outside the lines, having been at rest
    within rest_s before: seen slower than still_speed (the kick is dated to the last such frame + 1;
    the tracker spreads a kick over ~0.8 s, BAR-ATH goal kick 37376-37396), or unseen (dated to the
    first seen frame: the tracker finds the ball already moving, corner at BAR-ATH 34827)."""
    m = max(1, int(round(cfg["measure_s"] * fps)))
    rest = max(1, int(round(cfg["rest_s"] * fps)))
    fwd, _ = seen_speed(bx, by, m, fps)
    seen = ~np.isnan(bx)
    near = seen & (bx > -cfg["margin_m"]) & (bx < length + cfg["margin_m"]) \
        & (by > -cfg["margin_m"]) & (by < width + cfg["margin_m"])
    fast = near & (np.nan_to_num(fwd, nan=0.0) > cfg["kick_speed"])
    slow = seen & (np.nan_to_num(fwd, nan=np.inf) < cfg["still_speed"])
    out = np.zeros(bx.size, bool)
    for s, e in runs(fast):
        lo = max(0, s - rest)
        r = np.flatnonzero(slow[lo:s])
        u = np.flatnonzero(~seen[lo:s])
        if r.size and (not u.size or r[-1] > u[-1]):
            out[lo + int(r[-1]) + 1] = True
        elif u.size:
            out[lo + int(u[-1]) + 1] = True
    return out


def restart_kick(dead, bx, by, half, length, width, cfg, tags, keep=None, fps=FPS, used=None):
    """A Smart Tagging restart (corner, goal kick, throw-in, free kick, kick-off) ends the dead span
    at its kick: the first kick from rest (restart_kicks) at or after the tag start, inside the tag
    window and at most extend_s after a dead frame.
      - after the kick the ball is in play: dead frames from the kick on become play until the first
        evidence of a new stoppage (ball seen outside the lines, ball unseen for stop_unseen_s, ball
        seen slower than still_speed for stop_still_s), at most max_after_s; frames another dead
        rule marks (`keep`: D-SLOW, placed ball, wall) stay dead;
      - before the kick: a dead span that ends at most extend_s before it, the ball unseen or slow
        in between, is extended to the kick (the ball is placed, the taker has not kicked yet).
    Returns the new dead mask."""
    out = np.asarray(dead, bool).copy()
    keep = np.zeros(out.size, bool) if keep is None else np.asarray(keep, bool)
    n = out.size
    kicks = restart_kicks(bx, by, length, width, cfg, fps)
    seen = ~np.isnan(bx)
    v = np.nan_to_num(ball_speed(bx, by), nan=0.0)
    outside = seen & ((bx < -cfg["out_m"]) | (bx > length + cfg["out_m"])
                      | (by < -cfg["out_m"]) | (by > width + cfg["out_m"]))
    ext = int(round(cfg["extend_s"] * fps))
    max_after = int(round(cfg["max_after_s"] * fps))
    stop_unseen = int(round(cfg["stop_unseen_s"] * fps))
    stop_still = int(round(cfg["stop_still_s"] * fps))
    dead0 = np.asarray(dead, bool)
    for ts, te in tag_windows(tags, cfg["types"]):
        lo, hi = max(0, ts), min(n, te)
        if hi <= lo or half[lo] == 0:
            continue
        # the kick: the first kick from rest in the window that has a dead frame within extend_s
        # before it (the stoppage it restarts)
        k = None
        for f in np.flatnonzero(kicks[lo:hi]) + lo:
            if dead0[max(0, f - ext):f + 1].any():
                k = int(f)
                break
        if k is None:
            continue
        if used is not None:
            used.append(k)
        # before: extend the stoppage to the kick
        last = np.flatnonzero(dead0[max(0, k - ext):k])
        if last.size:
            a = max(0, k - ext) + int(last[-1]) + 1
            if a < k and (~seen[a:k] | (v[a:k] < cfg["still_speed"])).all():
                out[a:k] = True
        # after: play until a new stoppage shows
        unseen_run = still_run = 0
        for f in range(k, min(n, k + max_after)):
            if half[f] != half[k] or outside[f]:
                break
            unseen_run = unseen_run + 1 if not seen[f] else 0
            still_run = still_run + 1 if (seen[f] and v[f] < cfg["still_speed"]) else 0
            if unseen_run >= stop_unseen or still_run >= stop_still:
                f0 = f - max(unseen_run, still_run) + 1
                out[f0:f + 1] = dead0[f0:f + 1]       # give back what was released in the run
                break
            if out[f] and not keep[f]:
                out[f] = False
    return out


# -- (R3, 01/11) the ball leaves the pitch where the tracker no longer sees it ------------------------
def ball_leaves_unseen(bx, by, half, length, width, cfg, fps=FPS, mspeed=None):
    """(n,) bool dead mask: a ball gap of at least min_unseen_s whose last seen ball was heading out
    -- faster than min_speed (seen positions over look_s), at most near_m from the line it heads to,
    crossing it within max_cross_s -- is dead from the estimated crossing until the ball is seen
    again (at most max_s). ball_out_start needs the ball SEEN outside; the automatic tracker often
    loses it on the line (BAR-ATH 36265: last seen 3 m from the goal line at 10 m/s, unseen 31 s).
    With max_player_speed (R3b): only if the players slow down after the crossing (median mean
    player speed over player_speed_s at most max_player_speed): a stoppage, not a ball lost in play.
    With reappear_m (R3c): only if the ball is seen again within reappear_m of the line it left by
    (BAR-ATH 43050: lost heading to the touchline, found 6 s later on it for the throw-in)."""
    n = bx.size
    out = np.zeros(n, bool)
    look = max(1, int(round(cfg["look_s"] * fps)))
    for s, e in runs(np.isnan(bx)):
        if s == 0 or e - s < int(round(cfg["min_unseen_s"] * fps)) or half[s - 1] == 0:
            continue
        a = max(0, s - 1 - look)
        seen = np.flatnonzero(~np.isnan(bx[a:s]))
        if seen.size < 2:
            continue
        k0, k1 = a + int(seen[0]), a + int(seen[-1])
        vx = (bx[k1] - bx[k0]) / (k1 - k0) * fps
        vy = (by[k1] - by[k0]) / (k1 - k0) * fps
        if np.hypot(vx, vy) < cfg["min_speed"]:
            continue
        x, y = bx[k1], by[k1]
        cands = []
        if vx > 0:
            cands.append((length - x, (length - x) / vx, "xL"))
        if vx < 0:
            cands.append((x, x / -vx, "x0"))
        if vy > 0:
            cands.append((width - y, (width - y) / vy, "yW"))
        if vy < 0:
            cands.append((y, y / -vy, "y0"))
        dist, t, line = min(cands, key=lambda c: c[1])
        if dist > cfg["near_m"] or t > cfg["max_cross_s"]:
            continue
        if cfg.get("reappear_m") is not None:
            # R3c: the ball is seen again next to the line it left by (the restart is taken there)
            if e >= n:
                continue
            gap_to_line = {"x0": bx[e], "xL": length - bx[e], "y0": by[e], "yW": width - by[e]}[line]
            if gap_to_line > cfg["reappear_m"]:      # beyond the line (outside the pitch) counts too
                continue
        c = k1 + int(np.ceil(t * fps))
        hi = min(e, c + int(round(cfg["max_s"] * fps)))
        if mspeed is not None and cfg.get("max_player_speed") is not None:
            w = mspeed[c:min(hi, c + int(round(cfg["player_speed_s"] * fps)))]
            if not w.size or np.nanmedian(w) > cfg["max_player_speed"]:
                continue                       # play goes on: the players do not slow down
        if c < hi:
            out[c:hi] = True
    return out


# -- (R4, 01/11) a D-SLOW stoppage starts when the ball was lost ------------------------------------------
def slow_from_ball_loss(slow, bx, half, cfg, fps=FPS):
    """(n,) bool: frames to add before each D-SLOW span. D-SLOW needs everybody walking, which comes
    seconds after the whistle (BAR-ATH foul at ~35190: D-SLOW from 35305). When the ball has been
    unseen without a break from at most back_s before the span's start, the stoppage starts where
    the ball was last seen."""
    n = bx.size
    add = np.zeros(n, bool)
    back = int(round(cfg["back_s"] * fps))
    seen = ~np.isnan(bx)
    for s, e in runs(np.asarray(slow, bool)):
        if s == 0:
            continue
        lo = max(0, s - back)
        sn = np.flatnonzero(seen[lo:s])
        g = lo + int(sn[-1]) + 1 if sn.size else None     # first unseen frame after the last seen ball
        if g is None or g >= s or half[g] != half[s]:
            continue
        add[g:s] = True
    return add


def dead_islands(dead, bx, half, mspeed, cfg, fps=FPS):
    """R6 (01/11): (n,) bool, alive islands to close. A stretch of play of at most max_s between two
    dead spans, in which the ball is never seen and the players stay slow (median mean speed at most
    max_player_speed), shows no play at all: it is the same stoppage (BAR-ATH 45619 and 50354: the
    players walk at 1.2-1.3 m/s, just above D-SLOW's 1.1, and a 5-7 s island of "possession" opens)."""
    add = np.zeros(dead.size, bool)
    seen = ~np.isnan(bx)
    for s, e in runs(~np.asarray(dead, bool) & (half > 0)):
        if s == 0 or e >= dead.size or not dead[s - 1] or not dead[e]:
            continue
        if e - s > int(round(cfg["max_s"] * fps)) or half[s - 1] != half[e] or seen[s:e].any():
            continue
        if np.nanmedian(mspeed[s:e]) > cfg["max_player_speed"]:
            continue
        add[s:e] = True
    return add


def players_running(dead, half, mspeed, cfg, fps=FPS):
    """R8 (REPORT_B 26), the opposite of D-SLOW: everybody running means play. The frames of a dead span in a
    run of at least min_s with the mean player speed at or above min_ms go back to play, ball seen or not:
    a restart the tracker misses must not leave a stoppage open. Returns the frames given back."""
    back = np.zeros(dead.size, bool)
    fast = np.asarray(dead, bool) & (half > 0) & (np.nan_to_num(mspeed, nan=0.0) >= cfg["min_ms"])
    for s, e in runs(fast):
        if e - s >= int(round(cfg["min_s"] * fps)):
            back[s:e] = True
    return back


def fast_ball_in_play(dead, bx, by, half, length, width, cfg, fps=FPS):
    """R9 (REPORT_B 26): a ball seen on the pitch, away from the lines, moving fast for at least min_s is in play
    (a ball kicked away after a whistle is shorter than that). Returns the frames given back."""
    v = np.nan_to_num(ball_speed(bx, by, fps), nan=0.0)
    with np.errstate(invalid="ignore"):
        inside = (bx >= cfg["in_m"]) & (bx <= length - cfg["in_m"]) & (by >= cfg["in_m"]) & (by <= width - cfg["in_m"])
    fast = np.asarray(dead, bool) & (half > 0) & inside & (v >= cfg["min_ms"])
    back = np.zeros(dead.size, bool)
    for s, e in runs(fast):
        if e - s >= int(round(cfg["min_s"] * fps)):
            back[s:e] = True
    return back


def long_unseen_dead(dead, bx, half, mspeed, cfg, fps=FPS):
    """R10 (REPORT_B 26): a live stretch with the ball unseen for at least min_s while the players stay slow (median
    mean speed at most max_ms) is not a possession nobody can see: it is a stoppage. Returns the frames added."""
    add = np.zeros(dead.size, bool)
    for s, e in runs(np.isnan(bx) & (half > 0)):
        if e - s < int(round(cfg["min_s"] * fps)):
            continue
        if half[s] == half[e - 1] and np.nanmedian(mspeed[s:e]) <= cfg["max_ms"]:
            add[s:e] = True
    return add & ~np.asarray(dead, bool)


def extra_dead(dead, layers, bx, by, half, length, width, cfg, tags, fps=FPS, mspeed=None):
    """The 01/11 dead-ball rules on top of the B v3 mask, in this order (each switched on in cfg):
    R4 slow_from_ball_loss, R3 ball_leaves_unseen, R1 restart_kick, R6 dead_islands. `layers`: the per-rule masks
    built so far ({'slow', 'placed', 'wall', ...}); the new ones are added to it. Returns the mask."""
    out = np.asarray(dead, bool).copy()
    c4 = cfg.get("slow_from_ball_loss") or {}
    if c4.get("enabled") and "slow" in layers:
        a4 = slow_from_ball_loss(layers["slow"], bx, half, c4, fps) & (half > 0)
        layers["slow_back"] = a4
        out |= a4
    c3 = cfg.get("ball_leaves_unseen") or {}
    if c3.get("enabled"):
        a3 = ball_leaves_unseen(bx, by, half, length, width, c3, fps, mspeed=mspeed) & (half > 0)
        layers["unseen_out"] = a3
        out |= a3
    c1 = cfg.get("restart_kick") or {}
    if c1.get("enabled"):
        keep = np.zeros(out.size, bool)
        for k in ("slow", "placed", "wall", "slow_back", "unseen_out"):
            if k in layers:
                keep |= layers[k]
        used = []
        new = restart_kick(out, bx, by, half, length, width, c1, tags, keep=keep, fps=fps, used=used)
        layers["restart_play"] = out & ~new
        layers["restart_extend"] = new & ~out
        km = np.zeros(out.size, bool)
        km[used] = True
        layers["restart_kicks"] = km
        out = new
    c6 = cfg.get("dead_islands") or {}
    if c6.get("enabled") and mspeed is not None:
        a6 = dead_islands(out, bx, half, mspeed, c6, fps)
        layers["island"] = a6
        out |= a6
    c10 = cfg.get("long_unseen") or {}
    if c10.get("enabled") and mspeed is not None:
        a10 = long_unseen_dead(out, bx, half, mspeed, c10, fps)
        layers["long_unseen"] = a10
        out |= a10
    c8 = cfg.get("players_running") or {}
    if c8.get("enabled") and mspeed is not None:
        b8 = players_running(out, half, mspeed, c8, fps)
        layers["running_play"] = b8
        out = out & ~b8
    c9 = cfg.get("fast_ball_play") or {}
    if c9.get("enabled"):
        b9 = fast_ball_in_play(out, bx, by, half, length, width, c9, fps)
        layers["fast_ball_play"] = b9
        out = out & ~b9
    return out


def whistle_dead(dead, blasts, bx, by, px, py, half, length, width, cfg, kick_cfg, fps=FPS):
    """R7 (REPORT_B 18): the referee's whistle (hmm_v0/whistle.py blasts, video frames).

    R7a a blast while the ball is alive starts a stoppage at the blast, if the players really slow
        down within slow_s (the lowest 1 s mean speed <= slow_ratio x the speed in the before_s
        before it, or <= slow_ms): dead from the blast to the next dead span of the rules if it
        starts within merge_s, else to the next kick from rest within max_s; neither -> no change.
        With after_dead_guard_s > 0, a blast at most that long after the end of a dead span of the rules is
        the restart whistle (a free kick, a kick-off) heard late, not a new stoppage: no change.
    R7b a blast inside a dead span that has already lasted restart_min_dead_s (not the whistle that
        stops play) announces the restart: the span ends at the first kick from rest within
        restart_s of the blast's end, or restart_s after it.
    Returns (mask, layers {"whistle_stop", "whistle_restart_cut"}, info)."""
    out = np.asarray(dead, bool).copy()
    n = out.size
    raw = np.nan_to_num(_nanmean(np.hypot(np.gradient(px, axis=0), np.gradient(py, axis=0)) * fps, 1))
    v1 = np.convolve(raw, np.ones(fps) / fps, "same")
    kicks = np.flatnonzero(restart_kicks(bx, by, length, width, kick_cfg, fps))
    slow_k, before_k = int(cfg["slow_s"] * fps), int(cfg["before_s"] * fps)
    merge_k, max_k = int(cfg["merge_s"] * fps), int(cfg["max_s"] * fps)
    rs_k, min_dead_k = int(cfg["restart_s"] * fps), int(cfg["restart_min_dead_s"] * fps)
    guard_k = int(cfg.get("after_dead_guard_s", 0) * fps)   # R7a guard (REPORT_B 33), 0 = off
    stop = np.zeros(n, bool)
    cut = np.zeros(n, bool)
    info = {"stop": 0, "stop_no_slowdown": 0, "stop_no_end": 0, "restart": 0, "restart_early": 0}
    ref = out.copy()                       # decisions on the rules' mask, not on each other's output
    for s, e in blasts:
        s, e = int(s), int(e)
        if s >= n or half[s] == 0:
            continue
        if not ref[s]:
            g0 = max(0, s - guard_k)
            if guard_k and np.any(ref[g0:s] & (half[g0:s] == half[s])):
                # just after the rules' stoppage ended: the whistle that restarts play, not a new stop
                info["stop_after_restart"] = info.get("stop_after_restart", 0) + 1
                continue
            v0 = raw[max(0, s - before_k):s].mean() if s > 0 else raw[0]
            vmin = v1[s:min(n, s + slow_k)].min()
            if not (vmin <= cfg["slow_ratio"] * v0 or vmin <= cfg["slow_ms"]):
                info["stop_no_slowdown"] += 1
                continue
            nxt = np.flatnonzero(ref[s:min(n, s + merge_k)] & (half[s:min(n, s + merge_k)] == half[s]))
            if nxt.size:
                end = s + int(nxt[0])
            else:
                k = kicks[(kicks > e) & (kicks <= s + max_k)]
                k = k[half[k] == half[s]] if k.size else k
                if not k.size:
                    info["stop_no_end"] += 1
                    continue
                end = int(k[0])
            stop[s:end] = True
            info["stop"] += 1
        elif cfg.get("restart", True):
            a = s
            while a > 0 and ref[a - 1]:
                a -= 1
            b = s
            while b < n and ref[b]:
                b += 1
            if s - a < min_dead_k:
                info["restart_early"] += 1
                continue
            k = kicks[(kicks >= s) & (kicks <= e + rs_k)]
            if not k.size and cfg.get("restart_need_kick"):
                info["restart_no_kick"] = info.get("restart_no_kick", 0) + 1
                continue
            new_end = int(k[0]) if k.size else e + rs_k
            if new_end < b:
                cut[new_end:b] = True
                info["restart"] += 1
    out = (out | stop) & ~cut
    return out, {"whistle_stop": stop & ~ref, "whistle_restart_cut": cut}, info
