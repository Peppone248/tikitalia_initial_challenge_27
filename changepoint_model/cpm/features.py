"""Per-frame features from the tracking alone, computed period by period.

Two groups, by their behaviour when home and away are exchanged (and the attack direction with
them):
    odd   f(swapped) = -f   -> who has the ball (H vs A); a model with no intercept on these is
                               exactly antisymmetric, so "home has the ball more" cannot be learnt
    even  f(swapped) =  f   -> alive vs dead, and the boundary classifiers

Conventions: 25 fps unless the match says otherwise; a team's "forward" axis is its own attack
direction, so every team-level quantity is comparable between the two teams and between periods.
Windows are centred unless the name says `past` / `next`; the model is run offline on a recorded
match, so looking forward is legitimate (the scorer sees the whole file).

The constants are a priori (football distances and speeds): control radius 1.5 m, controlled ball
below 8 m/s, walking below 1.1 m/s, contact = direction change > 30 deg or speed change > 2 m/s
over 3 frames (Vidal-Codina et al. 2022), windows 0.5 / 1 / 2 / 4 s, decays 0.5 / 2 / 8 s.
"""

import numpy as np

CONTROL_M = 1.5
CONTROL_BALL_SPEED = 8.0
WALK_SPEED = 1.1
CONTACT_DEG = 30.0
CONTACT_DV = 2.0
WINDOWS_S = (0.5, 1.0, 2.0, 4.0)
DECAYS_S = (0.5, 2.0, 8.0)
PRESSURE_M = (1.0, 3.0, 8.0)
UNSEEN_CAP_S = 60.0
MAX_PLAYER_SPEED = 12.0
MAX_BALL_SPEED = 40.0
SMOOTH_POS = 5


def _nanmean_axis1(a):
    ok = ~np.isnan(a)
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(ok, a, 0.0).sum(axis=1) / n
    out[n == 0] = np.nan
    return out


def _box(x, w):
    """Centred moving sum over 2w+1 frames (zeros outside), on axis 0."""
    if w <= 0:
        return x.copy()
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], axis=0), axis=0)
    T = x.shape[0]
    hi = np.minimum(np.arange(T) + w + 1, T)
    lo = np.maximum(np.arange(T) - w, 0)
    return c[hi] - c[lo]


def _box_past(x, w):
    """Sum over the w frames before t (t excluded)."""
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], axis=0), axis=0)
    T = x.shape[0]
    lo = np.maximum(np.arange(T) - w, 0)
    return c[np.arange(T)] - c[lo]


def _box_next(x, w):
    """Sum over the w frames after t (t excluded)."""
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], axis=0), axis=0)
    T = x.shape[0]
    hi = np.minimum(np.arange(T) + 1 + w, T)
    return c[hi] - c[np.arange(T) + 1]


def _smooth(x, w):
    """Centred moving average ignoring NaN (NaN where nothing in the window)."""
    ok = ~np.isnan(x)
    num = _box(np.where(ok, x, 0.0), w)
    den = _box(ok.astype(float), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def _since(mask):
    """Frames since the last True (0 at a True frame), large where none before."""
    T = len(mask)
    idx = np.where(mask, np.arange(T), -1)
    last = np.maximum.accumulate(idx)
    out = np.arange(T) - last
    out[last < 0] = T
    return out


def _until(mask):
    return _since(mask[::-1])[::-1]


def _last_value(values, mask):
    """Carry the last value where mask is True forward (NaN before the first)."""
    T = len(mask)
    idx = np.where(mask, np.arange(T), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, values[np.maximum(last, 0)], np.nan)
    return out


def _next_value(values, mask):
    return _last_value(values[::-1], mask[::-1])[::-1]


def _decayed(sign, mask, dt, tau_frames):
    """sign of the last event, decayed by exp(-dt / tau)."""
    s = _last_value(sign, mask)
    s = np.where(np.isnan(s), 0.0, s)
    return s * np.exp(-np.minimum(dt, 10 * tau_frames) / tau_frames)


def _team_stats(px, py, fwd_sign, fps):
    """Collective quantities of one team, in its own attack frame. Arrays (T,)."""
    T, P = px.shape
    fx = px * fwd_sign                      # forward coordinate
    n = np.sum(~np.isnan(px), axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = _nanmean_axis1(fx)
        cy = _nanmean_axis1(py)
        sx = np.nanstd(fx, axis=1)
        sy = np.nanstd(py, axis=1)
    enough = n >= 3
    srt = np.sort(np.where(np.isnan(fx), np.inf, fx), axis=1)
    deepest = srt[:, 0]
    second = srt[:, 1] if P > 1 else srt[:, 0]
    srt_desc = np.sort(np.where(np.isnan(fx), -np.inf, fx), axis=1)
    highest = srt_desc[:, -1]
    for arr in (deepest, second, highest):
        arr[~np.isfinite(arr)] = np.nan
    # velocities
    sm_x = _smooth(fx, SMOOTH_POS // 2)
    sm_y = _smooth(py, SMOOTH_POS // 2)
    vx = np.full_like(sm_x, np.nan)
    vy = np.full_like(sm_y, np.nan)
    vx[1:-1] = (sm_x[2:] - sm_x[:-2]) * fps / 2.0
    vy[1:-1] = (sm_y[2:] - sm_y[:-2]) * fps / 2.0
    speed = np.sqrt(vx ** 2 + vy ** 2)
    speed[speed > MAX_PLAYER_SPEED] = np.nan
    vfwd = _nanmean_axis1(np.where(np.isnan(speed), np.nan, vx))
    mspeed = _nanmean_axis1(speed)
    out = dict(n=n, cx=cx, cy=cy, spread=sx + sy, length=highest - deepest,
               width=np.nanmax(np.where(np.isnan(py), -np.inf, py), axis=1)
               - np.nanmin(np.where(np.isnan(py), np.inf, py), axis=1),
               deep=deepest, second=second, high=highest, vfwd=vfwd, mspeed=mspeed)
    out["width"][~np.isfinite(out["width"])] = np.nan
    for k in out:
        if k != "n":
            out[k] = np.where(enough | (k in ("mspeed",)), out[k], np.nan)
    return out, speed


def frame_features(m):
    """Return (even (T, Ne), odd (T, No), even_names, odd_names). Zero outside the periods."""
    T = m.T
    fps = m.fps
    even_cols, odd_cols = {}, {}
    even_out, odd_out = None, None
    for p, s, e in m.period_slices():
        ev, od = _period_features(m, s, e, m.attack[p], fps)
        if even_out is None:
            even_cols = list(ev.keys())
            odd_cols = list(od.keys())
            even_out = np.zeros((T, len(even_cols)), dtype=np.float32)
            odd_out = np.zeros((T, len(odd_cols)), dtype=np.float32)
        for j, k in enumerate(even_cols):
            even_out[s:e, j] = ev[k]
        for j, k in enumerate(odd_cols):
            odd_out[s:e, j] = od[k]
    even_out = np.nan_to_num(even_out, nan=0.0, posinf=0.0, neginf=0.0)
    odd_out = np.nan_to_num(odd_out, nan=0.0, posinf=0.0, neginf=0.0)
    return even_out, odd_out, even_cols, odd_cols


def _period_features(m, s, e, attack, fps):
    px, py = m.px[s:e].astype(np.float64), m.py[s:e].astype(np.float64)
    ball = m.ball[s:e]
    T = e - s
    L, W = m.pitch
    home = m.team == 0
    ev, od = {}, {}
    W_ = {w: int(round(w * fps)) for w in WINDOWS_S}
    TAU = {t: t * fps for t in DECAYS_S}

    # ---- ball ---------------------------------------------------------------------------------
    seen = ~np.isnan(ball[:, 0])
    bx, by = ball[:, 0], ball[:, 1]
    since = np.minimum(_since(seen), UNSEEN_CAP_S * fps)
    until = np.minimum(_until(seen), UNSEEN_CAP_S * fps)
    ev["ball_seen"] = seen.astype(float)
    ev["ball_since_log"] = np.log1p(since / fps)
    ev["ball_until_log"] = np.log1p(until / fps)
    ev["ball_gap_log"] = np.log1p((since + until) / fps) * (~seen)
    vbx = np.full(T, np.nan)
    vby = np.full(T, np.nan)
    ok3 = seen.copy()
    ok3[1:-1] &= seen[:-2] & seen[2:]
    vbx[1:-1] = (bx[2:] - bx[:-2]) * fps / 2.0
    vby[1:-1] = (by[2:] - by[:-2]) * fps / 2.0
    bspeed = np.sqrt(vbx ** 2 + vby ** 2)
    bspeed[~ok3] = np.nan
    bspeed[bspeed > MAX_BALL_SPEED] = np.nan
    bsp0 = np.nan_to_num(bspeed, nan=0.0)
    ev["ball_speed"] = bsp0
    ev["ball_speed_1s"] = np.nan_to_num(_smooth(bspeed, W_[1.0]), nan=0.0)
    ev["ball_slow"] = ((bspeed < 0.5) & seen).astype(float)
    ev["ball_slow_2s"] = _box(ev["ball_slow"], W_[2.0]) / (2 * W_[2.0] + 1)
    out_m = np.maximum(np.abs(bx) - L / 2, np.abs(by) - W / 2)
    ev["ball_out"] = np.nan_to_num((out_m > 0.5).astype(float) * seen, nan=0.0)
    ev["ball_line_dist"] = np.nan_to_num(np.minimum(-out_m, 20.0) * seen / 20.0, nan=0.0)
    last_out = _last_value(ev["ball_out"], seen)
    ev["ball_last_seen_out"] = np.nan_to_num(last_out, nan=0.0)
    last_line = _last_value(np.nan_to_num(np.minimum(-out_m, 20.0), nan=20.0), seen)
    ev["ball_last_seen_line_dist"] = np.nan_to_num(last_line, nan=20.0) / 20.0
    last_speed = _last_value(bsp0, seen & ok3)
    ev["ball_last_seen_speed"] = np.nan_to_num(last_speed, nan=0.0)
    ev["ball_abs_x"] = np.nan_to_num(np.abs(_last_value(bx, seen)), nan=0.0) / (L / 2)
    ev["ball_abs_y"] = np.nan_to_num(np.abs(_last_value(by, seen)), nan=0.0) / (W / 2)

    # ---- players ------------------------------------------------------------------------------
    hs, hspeed = _team_stats(px[:, home], py[:, home], attack, fps)
    as_, aspeed = _team_stats(px[:, ~home], py[:, ~home], -attack, fps)
    speed_all = np.concatenate([hspeed, aspeed], axis=1)
    mspeed = _nanmean_axis1(speed_all)
    ev["n_tracked"] = (hs["n"] + as_["n"]) / 22.0
    ev["n_diff_abs"] = np.abs(hs["n"] - as_["n"]) / 11.0
    ev["mean_speed"] = np.nan_to_num(mspeed, nan=0.0)
    for w in (1.0, 2.0, 4.0):
        ev["mean_speed_{}s".format(w)] = np.nan_to_num(_smooth(mspeed, W_[w]), nan=0.0)
    ev["walking_frac"] = np.nan_to_num(_nanmean_axis1((speed_all < WALK_SPEED).astype(float)
                                                      * np.where(np.isnan(speed_all), np.nan, 1)),
                                       nan=0.0)
    ev["walking_frac_2s"] = np.nan_to_num(_smooth(ev["walking_frac"], W_[2.0]), nan=0.0)
    ev["max_speed"] = np.nan_to_num(np.nanmax(np.where(np.isnan(speed_all), -1, speed_all), axis=1),
                                    nan=0.0).clip(0)
    ev["speed_change_1s"] = _box_next(np.nan_to_num(mspeed), W_[1.0]) / W_[1.0] \
        - _box_past(np.nan_to_num(mspeed), W_[1.0]) / W_[1.0]
    ev["speed_change_2s"] = _box_next(np.nan_to_num(mspeed), W_[2.0]) / W_[2.0] \
        - _box_past(np.nan_to_num(mspeed), W_[2.0]) / W_[2.0]
    # centroids in the pitch frame (not attack frame) for even features
    hcx = hs["cx"] * attack
    acx = as_["cx"] * (-attack)
    ev["centroid_dist"] = np.nan_to_num(np.sqrt((hcx - acx) ** 2 + (hs["cy"] - as_["cy"]) ** 2),
                                        nan=0.0) / 20.0
    ev["spread_sum"] = np.nan_to_num(hs["spread"] + as_["spread"], nan=0.0) / 30.0
    ev["spread_absdiff"] = np.nan_to_num(np.abs(hs["spread"] - as_["spread"]), nan=0.0) / 10.0
    ev["centroid_abs_x"] = np.nan_to_num(np.abs((hcx + acx) / 2), nan=0.0) / (L / 2)
    ev["players_in_box_any"] = _in_box_count(px, py, L, W) / 10.0

    # ---- ball - players -----------------------------------------------------------------------
    dx = px - bx[:, None]
    dy = py - by[:, None]
    dist = np.sqrt(dx ** 2 + dy ** 2)          # NaN where ball or player unseen
    dist_f = np.where(np.isnan(dist), np.inf, dist)
    dh = np.min(dist_f[:, home], axis=1)
    da = np.min(dist_f[:, ~home], axis=1)
    dmin = np.minimum(dh, da)
    ev["nearest_dist"] = np.where(np.isfinite(dmin), np.minimum(dmin, 20.0), 20.0) / 20.0 * 1.0
    ev["nearest_dist"][~seen] = 1.0
    srt = np.sort(dist_f, axis=1)
    second = srt[:, 1] if srt.shape[1] > 1 else srt[:, 0]
    ev["second_dist"] = np.where(np.isfinite(second), np.minimum(second, 20.0), 20.0) / 20.0
    ev["duel"] = ((dh < 1.5) & (da < 1.5)).astype(float)
    ev["duel_1s"] = _box(ev["duel"], W_[1.0]) / (2 * W_[1.0] + 1)
    controlled = seen & (dmin < CONTROL_M) & (np.nan_to_num(bspeed, nan=0.0) < CONTROL_BALL_SPEED)
    vote = np.where(controlled & (dh < da), 1.0, np.where(controlled & (da < dh), -1.0, 0.0))
    vote[~np.isfinite(dmin)] = 0.0
    ev["controlled"] = controlled.astype(float)
    ev["controlled_1s"] = _box(ev["controlled"], W_[1.0]) / (2 * W_[1.0] + 1)
    ev["controlled_4s"] = _box(ev["controlled"], W_[4.0]) / (2 * W_[4.0] + 1)
    od["vote"] = vote
    for w in WINDOWS_S:
        od["vote_{}s".format(w)] = _box(vote, W_[w]) / (2 * W_[w] + 1)
    for w in (1.0, 2.0, 4.0):
        od["vote_past_{}s".format(w)] = _box_past(vote, W_[w]) / W_[w]
        od["vote_next_{}s".format(w)] = _box_next(vote, W_[w]) / W_[w]
        ev["vote_flip_{}s".format(w)] = -(od["vote_past_{}s".format(w)]
                                          * od["vote_next_{}s".format(w)]).clip(max=0)
        ev["vote_absdiff_{}s".format(w)] = np.abs(od["vote_next_{}s".format(w)]
                                                  - od["vote_past_{}s".format(w)])
    has_vote = vote != 0
    dt_last = _since(has_vote)
    dt_next = _until(has_vote)
    for tau in DECAYS_S:
        od["vote_last_{}s".format(tau)] = _decayed(vote, has_vote, dt_last, TAU[tau])
        od["vote_next_dec_{}s".format(tau)] = _decayed(vote[::-1], has_vote[::-1],
                                                     _since(has_vote[::-1]), TAU[tau])[::-1]
        ev["vote_lastnext_flip_{}s".format(tau)] = -(od["vote_last_{}s".format(tau)]
                                                     * od["vote_next_dec_{}s".format(tau)]).clip(max=0)
    ev["vote_since_log"] = np.log1p(np.minimum(dt_last, 60 * fps) / fps)
    ev["vote_until_log"] = np.log1p(np.minimum(dt_next, 60 * fps) / fps)
    # pressure at three scales
    for r in PRESSURE_M:
        wgt = np.exp(-np.where(np.isnan(dist), np.inf, dist) / r)
        ph = np.sum(wgt[:, home], axis=1)
        pa = np.sum(wgt[:, ~home], axis=1)
        od["pressure_{}m".format(r)] = ph - pa
        ev["pressure_sum_{}m".format(r)] = ph + pa
        od["pressure_{}m_1s".format(r)] = _box(ph - pa, W_[1.0]) / (2 * W_[1.0] + 1)
    # nearest team on a slow ball
    slow = seen & (np.nan_to_num(bspeed, nan=99) < 3.0)
    with np.errstate(invalid="ignore"):
        od["nearest_team_slow"] = np.where(slow & np.isfinite(dmin), np.sign(da - dh), 0.0)
    od["nearest_team_slow_2s"] = _box(od["nearest_team_slow"], W_[2.0]) / (2 * W_[2.0] + 1)
    # contacts (Vidal-Codina)
    contact = np.zeros(T, dtype=bool)
    if T > 2:
        v0x, v0y = bx[1:-1] - bx[:-2], by[1:-1] - by[:-2]
        v1x, v1y = bx[2:] - bx[1:-1], by[2:] - by[1:-1]
        n0 = np.sqrt(v0x ** 2 + v0y ** 2)
        n1 = np.sqrt(v1x ** 2 + v1y ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosang = (v0x * v1x + v0y * v1y) / (n0 * n1)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        dv = np.abs(n1 - n0) * fps
        c = (ok3[1:-1]) & ((ang > CONTACT_DEG) | (dv > CONTACT_DV)) & (n0 * fps > 0.5)
        contact[1:-1] = np.nan_to_num(c, nan=False)
    contact &= np.isfinite(dmin) & (dmin < CONTROL_M)
    csign = np.where(contact & (dh < da), 1.0, np.where(contact & (da < dh), -1.0, 0.0))
    contact &= csign != 0
    od["contact"] = csign
    ev["contact_any"] = contact.astype(float)
    for w in (1.0, 2.0):
        od["contact_{}s".format(w)] = _box(csign, W_[w])
        ev["contact_n_{}s".format(w)] = _box(contact.astype(float), W_[w])
        ev["contact_flip_{}s".format(w)] = -(_box_past(csign, W_[w]) * _box_next(csign, W_[w])).clip(max=0)
    dtc = _since(contact)
    for tau in (0.5, 2.0):
        od["contact_last_{}s".format(tau)] = _decayed(csign, contact, dtc, TAU[tau])
        od["contact_next_{}s".format(tau)] = _decayed(csign[::-1], contact[::-1],
                                                    _since(contact[::-1]), TAU[tau])[::-1]
    # ball position in home's attack frame
    bfx = bx * attack
    od["ball_fx"] = np.nan_to_num(bfx, nan=0.0) / (L / 2)
    od["ball_fx_last"] = np.nan_to_num(_last_value(bfx, seen), nan=0.0) / (L / 2)
    od["ball_fx_next"] = np.nan_to_num(_next_value(bfx, seen), nan=0.0) / (L / 2)
    vbfx = np.nan_to_num(vbx * attack, nan=0.0)
    od["ball_vfwd"] = vbfx / 10.0
    od["ball_vfwd_last"] = np.nan_to_num(_last_value(vbfx, seen & ok3), nan=0.0) / 10.0

    # ---- collective shape: home minus away, each in its attack frame -------------------------
    def diff(k, scale):
        return np.nan_to_num(hs[k] - as_[k], nan=0.0) / scale

    shape = {"spread": diff("spread", 10.0), "length": diff("length", 20.0),
             "width": diff("width", 20.0), "vfwd": diff("vfwd", 2.0), "centroid": diff("cx", 20.0),
             "defline": diff("second", 20.0), "deepest": diff("deep", 20.0),
             "attline": diff("high", 20.0), "mspeed": diff("mspeed", 1.0)}
    for k, v in shape.items():
        od["shape_" + k] = v
        od["shape_{}_3s".format(k)] = _box(v, int(1.5 * fps)) / (2 * int(1.5 * fps) + 1)
    for k in ("vfwd", "centroid", "defline", "attline", "spread"):
        v = shape[k]
        for w in (1.0, 2.0):
            after = _box_next(v, W_[w]) / W_[w]
            before = _box_past(v, W_[w]) / W_[w]
            od["trans_{}_{}s".format(k, w)] = after - before
            ev["trans_abs_{}_{}s".format(k, w)] = np.abs(after - before)
    # home and away centroid relative to the ball along the attack axis (who is behind the ball)
    bfx_last = np.nan_to_num(_last_value(bfx, seen), nan=0.0)
    od["centroid_vs_ball"] = np.nan_to_num((hs["cx"] - bfx_last) - (as_["cx"] + bfx_last), nan=0.0) / 20.0
    # keeper-related: deepest player within 2 m of ball in his own box
    hdeep_box = (hs["deep"] < -L / 2 + 16.5)
    adeep_box = (as_["deep"] < -L / 2 + 16.5)
    ev["deep_in_box_any"] = np.nan_to_num(hdeep_box.astype(float) + adeep_box.astype(float), nan=0.0)
    return ev, od


def _in_box_count(px, py, L, W):
    inb = (np.abs(px) > L / 2 - 16.5) & (np.abs(py) < 20.16)
    return np.sum(inb, axis=1).astype(float)
