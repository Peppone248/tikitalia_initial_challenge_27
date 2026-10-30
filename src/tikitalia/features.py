"""Per-frame features: MEASUREMENTS shared by every rule. No thresholds, no labels.

Layering (docs/MOTORE.md 2):
    Match (clean positions) -> features (derived measurements) -> events -> rules -> engine

What belongs here: quantities a rule compares with a threshold ("distance of the nearest home
player from the ball"). What does NOT: the comparison itself ("home player in control") -- that
is an event or a rule. The only constants used are computation windows and validity guards,
all in params.yaml.

Every array is indexed by video frame. Shapes: (n,) per frame, (n, 2) per team (column HOME=0,
AWAY=1), (n, 3) per team + all players, (n, n_slots) per player slot. NaN = not measurable on
that frame (ball not seen, too few players tracked, half not graded for team-relative values).

Groups, in the order they are computed:
  1. player kinematics      velocities of every slot, tracker swaps removed
  2. ball extras            acceleration, seconds since the ball was last seen
  3. ball <-> players       distances, nearest player per team, how the ball moves with him,
                            who is chasing it, pressure on the nearest player
  4. team shape             centroid, spread, players in each half / box, forward speed,
                            all relative to the attack direction of that team
  5. collective tempo       median player speed, its recent reference, the slowdown ratio
  6. noise                  sigma of each family of features, for the strength z-scores (4.1)

Only numpy and the standard library, like preprocessing.py.
"""

import warnings

import numpy as np

from .preprocessing import AWAY, HOME, robust_sigma, runs, smooth_and_differentiate

TEAMS = (HOME, AWAY)
ALL = 2                                    # third column of the (n, 3) tempo arrays


class Features(object):
    """Container of the per-frame features. Built by compute_features(); rules read attributes.

    Player kinematics
      pvx, pvy, pspeed        (n, n_slots)  m/s, NaN where the slot is missing or the step is
                                            faster than V_PLAYER_MAX (tracker swap / jump)
    Ball extras
      ball_acc                (n,)          m/s^2, magnitude of the ball acceleration
      ball_missing_for        (n,)          s since the ball was last seen (0 when visible)
    Ball <-> players
      dist_ball               (n, n_slots)  m, every player to the ball (float32)
      nearest_dist            (n, 2)        m, nearest player of each team to the ball
      nearest_slot            (n, 2)        slot index of that player, -1 if none
      nearest_x, nearest_y    (n, 2)        m, his position
      nearest_vx, nearest_vy  (n, 2)        m/s, his velocity
      nearest_team            (n,)          HOME / AWAY / -1: team of the nearest player overall
      nearest_gap             (n,)          m, (other team's nearest) - (nearest overall), >= 0
      rel_speed               (n, 2)        m/s, |ball velocity - nearest player velocity|
      align_cos               (n, 2)        cosine between ball and nearest player velocities
      closing_speed           (n, 2)        m/s, rate at which the nearest player's distance to the
                                            ball shrinks (> 0 closing, ~0 while dribbling)
      chase_speed             (n, 2)        m/s, fastest run AT the ball within the team (own
                                            velocity towards the ball, whatever the ball does)
      pressure_dist           (n, 2)        m, nearest player of the team -> nearest opponent
    Team shape (relative to attack_dir; NaN on frames not in a graded half)
      n_visible               (n, 2)        tracked players
      centroid_fwd            (n, 2)        m, centroid along the attack direction, 0 = halfway,
                                            > 0 = in the opponent half
      centroid_y              (n, 2)        m
      spread                  (n, 2)        m, sd_x + sd_y of the team
      n_opp_half              (n, 2)        players in the opponent half
      n_opp_box, n_own_box    (n, 2)        players in the opponent / own penalty box
      v_forward               (n, 2)        m/s, mean velocity of the team towards the goal it attacks
    Collective tempo
      speed_med               (n, 3)        m/s, median player speed: HOME, AWAY, all
      speed_ref               (n, 3)        m/s, rolling median of speed_med over the last REF_WIN s
      slowdown                (n, 3)        speed_med / speed_ref (< 1 = slower than usual)
    Noise (NOMENCLATURA 2)
      sigma                   dict          {"player_pos", "ball_pos", "dist_ball_player",
                                             "player_speed"} in m or m/s
    """

    def __init__(self, n, n_slots):
        self.n, self.n_slots = n, n_slots
        self.sigma = {}
        self.diagnostics = {}

    def n_within(self, team, radius):
        """(n,) players of `team` within `radius` m of the ball. The radius comes from the rule."""
        cols = self._cols[team]
        return np.sum(self.dist_ball[:, cols] <= radius, axis=1)

    def team_of(self, slot):
        """Team of a slot index array (-1 stays -1)."""
        slot = np.asarray(slot)
        return np.where(slot >= 0, self._team_of_slot[np.clip(slot, 0, None)], -1)


# -- helpers ---------------------------------------------------------------------------------------
def _quiet(func, *args, **kwargs):
    """numpy nan-reductions warn on all-NaN slices; here NaN is the intended answer."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(*args, **kwargs)


def _take(values, slot):
    """values[f, slot[f]] with slot = -1 -> NaN."""
    safe = np.clip(slot, 0, None)
    out = np.take_along_axis(values, safe[:, None], axis=1)[:, 0].astype(float)
    out[slot < 0] = np.nan
    return out


def _nanargmin_rows(values):
    """(argmin per row, min per row) ignoring NaN; (-1, NaN) on all-NaN rows."""
    filled = np.where(np.isnan(values), np.inf, values)
    idx = np.argmin(filled, axis=1)
    best = filled[np.arange(len(idx)), idx]
    missing = ~np.isfinite(best)
    return np.where(missing, -1, idx), np.where(missing, np.nan, best)


def _rolling_trailing_median(per_second, window):
    """Median over the last `window` values (current one included), ignoring NaN."""
    window = max(int(window), 1)
    padded = np.concatenate((np.full(window - 1, np.nan), per_second))
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return _quiet(np.nanmedian, view, axis=1)


# -- 1. player kinematics --------------------------------------------------------------------------
def player_kinematics(px, py, fps, window, v_max):
    """Smoothed velocities of every slot, as for the ball (preprocessing.smooth_and_differentiate).

    A tracked slot that moves faster than v_max between frames is a swap of identities or a jump
    of the tracker, not a sprint: its velocity is set to NaN there (the position is kept).
    Returns (vx, vy, speed, residuals, share_capped).
    """
    n, k = px.shape
    vx = np.full((n, k), np.nan, dtype=np.float32)
    vy = np.full((n, k), np.nan, dtype=np.float32)
    residuals = []
    for j in range(k):
        x, y = px[:, j].astype(float), py[:, j].astype(float)
        if np.all(np.isnan(x)):
            continue
        _, _, vxj, vyj, res = smooth_and_differentiate(x, y, fps, window)
        vx[:, j], vy[:, j] = vxj, vyj
        if res.size:
            residuals.append(res)
    speed = np.hypot(vx, vy)
    measured = ~np.isnan(speed)
    too_fast = measured & (speed > v_max)
    vx[too_fast] = vy[too_fast] = speed[too_fast] = np.nan
    share = float(too_fast.sum() / measured.sum()) if measured.any() else 0.0
    residuals = np.concatenate(residuals) if residuals else np.array([])
    return vx, vy, speed, residuals, share


# -- 2. ball extras --------------------------------------------------------------------------------
def ball_acceleration(vx, vy, fps):
    """|d v / dt| on each visible stretch (central differences), NaN elsewhere."""
    acc = np.full(len(vx), np.nan)
    for start, end in runs(~np.isnan(vx)):
        if end - start >= 3:
            ax = np.gradient(vx[start:end]) * fps
            ay = np.gradient(vy[start:end]) * fps
            acc[start:end] = np.hypot(ax, ay)
    return acc


def seconds_missing(visible, fps):
    """Seconds since the ball was last seen; 0 when visible, NaN before the first sighting."""
    n = len(visible)
    idx = np.where(visible, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = (np.arange(n) - last) / float(fps)
    out[last < 0] = np.nan
    return out


# -- 3. ball <-> players ---------------------------------------------------------------------------
def ball_player(feats, match, cols):
    bx, by = match.ball_x[:, None], match.ball_y[:, None]
    dx = (bx - match.px).astype(np.float32)               # player -> ball
    dy = (by - match.py).astype(np.float32)
    dist = np.hypot(dx, dy)
    feats.dist_ball = dist

    n = match.n
    shape = (n, 2)
    feats.nearest_dist = np.full(shape, np.nan)
    feats.nearest_slot = np.full(shape, -1, dtype=np.int16)
    for name in ("nearest_x", "nearest_y", "nearest_vx", "nearest_vy", "rel_speed",
                 "align_cos", "closing_speed", "chase_speed", "pressure_dist"):
        setattr(feats, name, np.full(shape, np.nan))

    # two different "towards the ball" speeds, both projected on u = (ball - p) / |ball - p|:
    #   effort  v_player . u            -- is he running at the ball? (a chaser, even if slower)
    #   closing (v_player - v_ball) . u -- is the distance shrinking? (0 for a dribbler)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = dx / dist, dy / dist
        effort = feats.pvx * ux + feats.pvy * uy
        closing = (feats.pvx - match.ball_vx[:, None]) * ux + \
                  (feats.pvy - match.ball_vy[:, None]) * uy

    for t in TEAMS:
        c = cols[t]
        local, best = _nanargmin_rows(dist[:, c])
        slot = np.where(local >= 0, c[np.clip(local, 0, None)], -1)
        feats.nearest_slot[:, t] = slot
        feats.nearest_dist[:, t] = best
        feats.nearest_x[:, t] = _take(match.px, slot)
        feats.nearest_y[:, t] = _take(match.py, slot)
        feats.nearest_vx[:, t] = _take(feats.pvx, slot)
        feats.nearest_vy[:, t] = _take(feats.pvy, slot)
        feats.closing_speed[:, t] = _take(closing, slot)
        feats.chase_speed[:, t] = _quiet(np.nanmax, effort[:, c], axis=1)

        dvx = match.ball_vx - feats.nearest_vx[:, t]
        dvy = match.ball_vy - feats.nearest_vy[:, t]
        feats.rel_speed[:, t] = np.hypot(dvx, dvy)
        nb = np.hypot(match.ball_vx, match.ball_vy)
        npl = np.hypot(feats.nearest_vx[:, t], feats.nearest_vy[:, t])
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (match.ball_vx * feats.nearest_vx[:, t] +
                   match.ball_vy * feats.nearest_vy[:, t]) / (nb * npl)
        cos[(nb == 0) | (npl == 0)] = np.nan
        feats.align_cos[:, t] = np.clip(cos, -1.0, 1.0)

        # pressure: the only player-to-player distance a rule needs -- nearest player of the
        # team (ball-carrier candidate) to the nearest opponent
        o = cols[1 - t]
        cx, cy = feats.nearest_x[:, t][:, None], feats.nearest_y[:, t][:, None]
        d_opp = np.hypot(match.px[:, o] - cx, match.py[:, o] - cy)
        feats.pressure_dist[:, t] = _quiet(np.nanmin, d_opp, axis=1)

    dH, dA = feats.nearest_dist[:, HOME], feats.nearest_dist[:, AWAY]
    feats.nearest_team = np.where(np.isnan(dH) & np.isnan(dA), -1,
                                  np.where(np.nan_to_num(dH, nan=np.inf) <=
                                           np.nan_to_num(dA, nan=np.inf), HOME, AWAY)
                                  ).astype(np.int8)
    feats.nearest_gap = np.abs(dH - dA)                    # NaN if one team has nobody tracked


# -- 4. team shape ---------------------------------------------------------------------------------
def direction_per_frame(half, attack_dir, team):
    """(n,) +1 / -1 attack direction of `team` on each frame; NaN outside the graded halves."""
    out = np.full(len(half), np.nan)
    for h in (1, 2):
        d = attack_dir.get((team, h))
        if d is not None:
            out[half == h] = d
    return out


def team_shape(feats, match, pitch, cols, attack_dir):
    L, W = pitch.length, pitch.width
    shape = (match.n, 2)
    for name in ("centroid_fwd", "centroid_y", "spread", "v_forward"):
        setattr(feats, name, np.full(shape, np.nan))
    feats.n_visible = np.zeros(shape, dtype=np.int16)
    for name in ("n_opp_half", "n_opp_box", "n_own_box"):
        setattr(feats, name, np.zeros(shape, dtype=np.int16))

    for t in TEAMS:
        c = cols[t]
        x, y = match.px[:, c], match.py[:, c]
        seen = ~np.isnan(x)
        feats.n_visible[:, t] = seen.sum(axis=1)
        d = direction_per_frame(match.half, attack_dir, t)[:, None]
        fwd = (x - L / 2.0) * d                            # > 0 = opponent half of team t
        in_box_y = np.abs(y - W / 2.0) <= pitch.box_half_width
        box_line = L / 2.0 - pitch.box_depth

        feats.centroid_fwd[:, t] = _quiet(np.nanmean, fwd, axis=1)
        feats.centroid_y[:, t] = _quiet(np.nanmean, y, axis=1)
        feats.spread[:, t] = _quiet(np.nanstd, x, axis=1) + _quiet(np.nanstd, y, axis=1)
        feats.n_opp_half[:, t] = np.sum(fwd > 0, axis=1)
        feats.n_opp_box[:, t] = np.sum((fwd >= box_line) & in_box_y, axis=1)
        feats.n_own_box[:, t] = np.sum((fwd <= -box_line) & in_box_y, axis=1)
        feats.v_forward[:, t] = _quiet(np.nanmean, feats.pvx[:, c] * d, axis=1)


# -- 5. collective tempo ---------------------------------------------------------------------------
def collective_tempo(feats, match, cols, ref_win_s, min_players):
    """Median player speed per team and overall, against its recent reference.

    The reference is the median of the per-second medians over the last REF_WIN seconds (trailing:
    "how fast were they playing just before?"). A known limit: inside a stoppage longer than
    REF_WIN the reference itself drops and the ratio drifts back towards 1 -- rules that open a D
    on a slowdown must look at its START, not at how long the ratio stays low.
    """
    n, fps = match.n, int(round(match.fps))
    feats.speed_med = np.full((n, 3), np.nan)
    groups = {HOME: cols[HOME], AWAY: cols[AWAY], ALL: np.arange(feats.pspeed.shape[1])}
    for g, c in groups.items():
        sp = feats.pspeed[:, c]
        med = _quiet(np.nanmedian, sp, axis=1)
        need = min_players if g != ALL else 2 * min_players
        med[np.sum(~np.isnan(sp), axis=1) < need] = np.nan
        feats.speed_med[:, g] = med

    n_sec = int(np.ceil(n / float(fps)))
    padded = np.full((n_sec * fps, 3), np.nan)
    padded[:n] = feats.speed_med
    per_second = _quiet(np.nanmedian, padded.reshape(n_sec, fps, 3), axis=1)
    feats.speed_ref = np.full((n, 3), np.nan)
    for g in range(3):
        ref = _rolling_trailing_median(per_second[:, g], ref_win_s)
        feats.speed_ref[:, g] = np.repeat(ref, fps)[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        feats.slowdown = feats.speed_med / feats.speed_ref
    feats.slowdown[~np.isfinite(feats.slowdown)] = np.nan


# -- entry point -----------------------------------------------------------------------------------
def compute_features(match, pitch, params, attack_dir):
    """All features of a prepared Match. attack_dir: {(team, half): +1/-1} from events."""
    n, k = match.px.shape
    feats = Features(n, k)
    cols = {t: np.flatnonzero(match.team_of_slot == t) for t in TEAMS}
    feats._cols, feats._team_of_slot = cols, match.team_of_slot

    # 1. players
    vx, vy, speed, residuals, capped = player_kinematics(
        match.px, match.py, match.fps, params.SMOOTH_WIN, params.V_PLAYER_MAX)
    feats.pvx, feats.pvy, feats.pspeed = vx, vy, speed

    # 6. noise first: the player sigma is an estimated parameter other code may read
    sigma_p = robust_sigma(residuals)
    sigma_p = max(sigma_p, params.SIGMA_POS_FLOOR) if np.isfinite(sigma_p) \
        else params.SIGMA_POS_FLOOR
    params.set_estimated("SIGMA_PLAYER_POS", sigma_p)
    sigma_b = params.SIGMA_POS
    # velocity = central difference of a W-frame moving average: the two averages differ by
    # 4 independent samples, so per component sigma_v = sigma_pos * fps / W (white jitter)
    w = max(int(params.SMOOTH_WIN), 1)
    feats.sigma = {
        "player_pos": sigma_p,
        "ball_pos": sigma_b,
        "dist_ball_player": float(np.hypot(sigma_b, sigma_p)),
        "player_speed": float(sigma_p * match.fps / w),
        "ball_speed": float(sigma_b * match.fps / w),
    }

    # 2. ball extras
    feats.ball_acc = ball_acceleration(match.ball_vx, match.ball_vy, match.fps)
    feats.ball_missing_for = seconds_missing(match.ball_visible, match.fps)

    # 3-5
    ball_player(feats, match, cols)
    team_shape(feats, match, pitch, cols, attack_dir)
    collective_tempo(feats, match, cols, params.REF_WIN, params.TEMPO_MIN_PLAYERS)

    feats.diagnostics = {"player_steps_too_fast": capped, "sigma_player_pos_m": sigma_p}
    match.diagnostics.update(feats.diagnostics)
    return feats
