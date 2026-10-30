"""The broadcast recording, aligned to the tactical timeline by an algorithm (no frame typed by hand).

DATA.md: the broadcast video has its own timeline (164,488 frames), cuts, replays and close-ups,
and no mapping to the tactical frames is given. Both ATDs are in pitch coordinates, so the mapping
can be found by matching what the two trackers see:

    1. COARSE   for broadcast windows of 20 s every 60 s, every offset t - b (step 2 frames) is scored
                by the mean ball distance (capped at 5 m) between the two trackers; the best offset
                of each window is kept when it beats every offset > 10 s away by a margin.
    2. MODES    the offsets found cluster on a few values (the TV timeline is continuous inside a
                half: replays are shown INSTEAD of live play, they do not shift the clock); a mode
                is kept if >= min_share of the windows vote for it.
    3. SWITCH   with several modes in time order, the frame where the mapping changes is the one
                that minimises the player-matching cost of the two sides (one switch per change).
    4. FINE     each mode is refined to the frame: offsets +/- 12 around it, scored by the median
                distance of every broadcast player to the nearest same-team tactical player.
    5. LIVE     a broadcast frame shows LIVE wide play when its homography is valid and its players
                match the tactical ones (median distance < live_max_m with >= live_min_players);
                otherwise it is a cut: close-up, crowd, replay, graphic.

On the tactical timeline this gives, per frame:
    covered   the broadcast was on air at that moment
    live      the broadcast was showing the live play from a wide camera
    cut       covered and not live (close-up, replay, ...)
    ball_x/y  the broadcast tracker's ball, where live (metres)

TV directors cut away from the wide camera when the ball is dead (fouls, goals, the ball out,
substitutions) and show replays during stoppages: `cut` is a strong prior on the dead ball,
including fouls and offsides, which the Smart Tagging does not contain.
"""

import os

import numpy as np

from tikitalia.preprocessing import parse_raw_atd, team_of_each_track, runs
from . import paths

FPS = 25

DEFAULTS = {
    "window_s": 20, "every_s": 60, "sample_step": 10, "offset_step": 2, "cap_m": 5.0,
    "min_pairs": 12, "margin_m": 0.8, "mode_tol": 6, "min_share": 0.1,
    "fine_halfwidth": 12, "fine_every": 25,
    "live_max_m": 1.5, "live_min_players": 4,
}


def load_broadcast(pitch, cache=None, game="BAR-ATH"):
    """Arrays of the broadcast ATD (metres) + homography validity; cached as .npz (gitignored)."""
    from games import asset_paths
    p = asset_paths(game=game, source="broadcast")
    cache = cache or paths.resolve("../outputs/cache/broadcast_atd_arrays.npz")
    team, _names = team_of_each_track(p["atd_meta"])
    if os.path.isfile(cache):
        z = np.load(cache)
        bx, by, px, py = z["bx"], z["by"], z["px"], z["py"]
    else:
        with open(p["atd_raw"]) as h:
            n = sum(1 for _ in h)
        bx, by, px, py = parse_raw_atd(p["atd_raw"], n, team.size, pitch)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, bx=bx, by=by, px=px, py=py, team=team)
    valid = np.zeros(bx.size, bool)
    hom = p.get("homography")
    if hom and os.path.isfile(hom):
        import csv
        with open(hom) as h:
            r = csv.reader(h)
            head = next(r)
            iv, ifr = head.index("valid"), head.index("frame")
            for row in r:
                f = int(row[ifr])
                if 0 <= f < valid.size:
                    valid[f] = row[iv] == "1"
    return {"bx": bx, "by": by, "px": px, "py": py, "team": team, "valid": valid}


def _player_cost(B, T, team_b, team_t, bi, ti):
    """Median over the broadcast players of the distance to the nearest same-team tactical player;
    also the number of broadcast players. NaN where nothing to compare."""
    out, cnt = [], np.zeros(bi.size, int)
    for team in (0, 1):
        bx, by = B["px"][bi][:, team_b == team], B["py"][bi][:, team_b == team]
        tx, ty = T["px"][ti][:, team_t == team], T["py"][ti][:, team_t == team]
        d = np.hypot(bx[:, :, None] - tx[:, None, :], by[:, :, None] - ty[:, None, :])
        d = np.where(np.isnan(d), np.inf, d).min(axis=2)
        d[np.isnan(bx) | np.isinf(d)] = np.nan
        out.append(d)
        cnt += (~np.isnan(bx)).sum(axis=1)
    d = np.concatenate(out, axis=1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(d, axis=1)
    return med, cnt


def coarse_offsets(B, T, c):
    nB, nT = B["bx"].size, T["bx"].size
    Tb = np.stack([T["bx"], T["by"]], 1).astype(np.float32)
    pad = np.full((nT + 2 * nB + 10, 2), np.nan, np.float32)
    pad[nB:nB + nT] = Tb
    offs = np.arange(-nB, nT, c["offset_step"])
    W, every = int(c["window_s"] * FPS), int(c["every_s"] * FPS)
    found = []
    for s in range(0, nB - W, every):
        k = np.arange(s, s + W, c["sample_step"])
        k = k[~np.isnan(B["bx"][k])]
        if k.size < c["min_pairs"]:
            continue
        idx = np.clip(k[:, None] + offs[None, :] + nB, 0, pad.shape[0] - 1)
        d = np.hypot(pad[idx, 0] - B["bx"][k][:, None], pad[idx, 1] - B["by"][k][:, None])
        n = (~np.isnan(d)).sum(0)
        with np.errstate(all="ignore"):
            cost = np.nansum(np.minimum(d, c["cap_m"]), 0) / np.maximum(n, 1)
        cost[n < c["min_pairs"]] = np.inf
        j = int(np.argmin(cost))
        far = np.abs(offs - offs[j]) > 10 * FPS
        if np.isfinite(cost[j]) and np.min(cost[far]) - cost[j] >= c["margin_m"]:
            found.append((s + W // 2, int(offs[j])))
    return found


def modes_of(found, c):
    if not found:
        return []
    o = np.array([f[1] for f in found])
    modes = []
    for v in np.unique(o):
        near = np.abs(o - v) <= c["mode_tol"]
        if near.mean() >= c["min_share"]:
            m = int(np.round(np.median(o[near])))
            if not any(abs(m - x) <= c["mode_tol"] for x in modes):
                modes.append(m)
    # order the modes by where (in broadcast time) they are found
    first = {m: np.median([b for b, x in found if abs(x - m) <= c["mode_tol"]]) for m in modes}
    return sorted(modes, key=lambda m: first[m])


def refine(B, T, team_b, team_t, lo, hi, base, c):
    bi = np.arange(lo, hi, c["fine_every"])
    best = None
    for o in range(base - c["fine_halfwidth"], base + c["fine_halfwidth"] + 1):
        ti = bi + o
        ok = (ti >= 0) & (ti < T["bx"].size)
        med, _ = _player_cost(B, T, team_b, team_t, bi[ok], ti[ok])
        score = np.nanmedian(med) if np.any(~np.isnan(med)) else np.inf
        if best is None or score < best[1]:
            best = (o, score)
    return best


def align(B, T, team_t, cfg=None):
    """{"segments": [(b_lo, b_hi, offset)], "per_frame_cost", "live" (broadcast frames), ...}"""
    c = dict(DEFAULTS, **(cfg or {}))
    team_b = B["team"]
    nB, nT = B["bx"].size, T["bx"].size
    found = coarse_offsets(B, T, c)
    modes = modes_of(found, c)
    if not modes:
        return None
    # switches between consecutive modes: minimise the summed player cost (frames every 25)
    bounds = [0]
    probe = np.arange(0, nB, 25)
    costs = []
    for m in modes:
        ti = probe + m
        ok = (ti >= 0) & (ti < nT)
        med = np.full(probe.size, np.nan)
        med[ok], _ = _player_cost(B, T, team_b, team_t, probe[ok], ti[ok])
        costs.append(np.where(np.isnan(med), 0.0, np.minimum(med, 10.0)))
    for i in range(len(modes) - 1):
        a, b = costs[i], costs[i + 1]
        lo = np.searchsorted(probe, bounds[-1])
        ca = np.concatenate(([0], np.cumsum(a[lo:])))
        cb = np.concatenate(([0], np.cumsum(b[lo:][::-1])))[::-1]
        k = int(np.argmin(ca + cb))
        bounds.append(int(probe[min(lo + k, probe.size - 1)]))
    bounds.append(nB)
    segments = []
    for i, m in enumerate(modes):
        lo, hi = bounds[i], bounds[i + 1]
        o, score = refine(B, T, team_b, team_t, lo, hi, m, c)
        segments.append((lo, hi, o, float(score)))
    # per broadcast frame: tactical frame, matching cost, live
    tframe = np.full(nB, -1, int)
    for lo, hi, o, _s in segments:
        tframe[lo:hi] = np.arange(lo, hi) + o
    tframe[(tframe < 0) | (tframe >= nT)] = -1
    allb = np.flatnonzero(tframe >= 0)
    cost = np.full(nB, np.nan)
    nplay = np.zeros(nB, int)
    for s in range(0, allb.size, 20000):
        bi = allb[s:s + 20000]
        cost[bi], nplay[bi] = _player_cost(B, T, team_b, team_t, bi, tframe[bi])
    live = B["valid"] & (tframe >= 0) & (nplay >= c["live_min_players"]) \
        & (np.nan_to_num(cost, nan=np.inf) < c["live_max_m"])
    return {"segments": segments, "coarse": found, "modes": modes, "tframe": tframe,
            "cost": cost, "n_players": nplay, "live": live, "cfg": c}


def to_tactical(B, al, nT):
    """Arrays on the TACTICAL timeline: covered, live, cut, ball_x, ball_y (NaN where not live)."""
    covered = np.zeros(nT, bool)
    live = np.zeros(nT, bool)
    bx = np.full(nT, np.nan)
    by = np.full(nT, np.nan)
    tf = al["tframe"]
    ok = tf >= 0
    covered[tf[ok]] = True
    lv = ok & al["live"]
    live[tf[lv]] = True
    bx[tf[lv]] = B["bx"][lv]
    by[tf[lv]] = B["by"][lv]
    return {"covered": covered, "live": live, "cut": covered & ~live, "ball_x": bx, "ball_y": by}


def cut_spans(tac, half, fps=FPS, min_s=2.0, bridge_s=1.0):
    """Stretches where the broadcast was on a cut (not live) for >= min_s, short live islands
    (< bridge_s) inside a cut bridged; per graded half."""
    cut = tac["cut"] & (half > 0)
    k = int(round(bridge_s * fps))
    for s, e in runs(~cut):
        if 0 < s and e < cut.size and e - s < k and cut[s - 1] and cut[e]:
            cut[s:e] = True
    return [(s, e) for s, e in runs(cut) if e - s >= int(round(min_s * fps))]


def bc_dead(tac, half, player_speed, ball_speed, cfg, fps=FPS):
    """D-BC (option, OFF by default): inside a broadcast cut of >= min_s, the frames where the
    tactical tracking shows no play (mean player speed < v_max_ms and no ball seen faster than
    kick_speed), cleaned like the other D spans (gaps < close_s joined, spans < min_dead_s dropped).
    The cut says "the director left the live play", the tracking says "and nobody is playing".
    Not validated on IDSSE (no broadcast there): see REPORT_B.md for what was measured."""
    from .segment_c import _clean_dead
    m = np.zeros(half.size, bool)
    for s, e in cut_spans(tac, half, fps, cfg["min_s"], cfg["bridge_s"]):
        m[s:e] = True
    cand = m & (player_speed < cfg["v_max_ms"]) \
        & ~(np.nan_to_num(ball_speed) > cfg["kick_speed"]) & (half > 0)
    return _clean_dead(cand, half, fps, {"close_gap_s": cfg["close_s"],
                                         "min_dead_s": cfg["min_dead_s"], "min_alive_s": 0})
