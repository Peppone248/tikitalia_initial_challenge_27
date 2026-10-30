"""Validation on other games: IDSSE (DFL) matches -> Match -> pipeline -> scored with the official
evaluation/possession_metrics.py. Config: config/validation.yaml. CLI: scripts/run_validation.py.

Labels (the IDSSE tracking itself, no person involved):
    BallStatus 0                 -> D
    BallStatus 1, BallPossession 1 -> H (home), 2 -> A (away), 0 -> X (not scored)
Frames of periods 1 and 2 are graded, like the two halves of BAR-ATH.

Degradation to BAR-ATH-like tracking uses only BAR-ATH TRACKING statistics (lengths of the ball
gaps / visible stretches, player-track fragments), never its ground truth.
"""

import copy
import os
import sys

import numpy as np
import yaml

from . import paths
from tikitalia.geometry import Pitch
from tikitalia.preprocessing import Match, prepare_ball, runs
from tikitalia.config import load_geometry

sys.path.insert(0, os.path.join(paths.REPO_ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25


def load_config(path=None):
    with open(path or os.path.join(paths.CONFIG_DIR, "validation.yaml"), encoding="utf-8") as h:
        return yaml.safe_load(h)


# -- data ------------------------------------------------------------------------------------------
def load_idsse(path):
    z = np.load(path)
    L = float(z["pitch_x"]) or 105.0
    W = float(z["pitch_y"]) or 68.0
    shift_x = L / 2.0 if np.nanmin(z["ball_x"]) < -1.0 else 0.0     # DFL: origin at the centre
    shift_y = W / 2.0 if np.nanmin(z["ball_y"]) < -1.0 else 0.0
    d = {k: z[k] for k in z.files}
    d["ball_x"] = d["ball_x"] + shift_x
    d["ball_y"] = d["ball_y"] + shift_y
    d["px"] = (d["px"] + shift_x).astype(np.float32)
    d["py"] = (d["py"] + shift_y).astype(np.float32)
    d["length"], d["width"] = L, W
    return d


def labels_of(d):
    lab = np.full(d["period"].size, "X", dtype="<U1")
    graded = np.isin(d["period"], (1, 2))
    alive, poss = d["ball_alive"], d["ball_possession"]
    lab[graded & ~alive] = "D"
    lab[graded & alive & (poss == 1)] = "H"
    lab[graded & alive & (poss == 2)] = "A"
    return lab


def controlled_labels(lab, min_s):
    """CHALLENGE.md applied mechanically to the DFL labels: "a simple touch of the ball without
    control" / "a deflection that does not intercept it" is not a change of possession. A spell of
    team T shorter than min_s, between two spells of the OTHER team (no dead ball in between),
    is relabelled to that other team. min_s = T_CONTROL (a priori, the same constant the rules use),
    never chosen on any ground truth."""
    out = lab.copy()
    k = int(round(min_s * FPS))
    changed = True
    while changed:
        changed = False
        idx = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
        ends = np.r_[idx[1:], out.size]
        for i in range(1, len(idx) - 1):
            s, e = idx[i], ends[i]
            mid, before, after = out[s], out[idx[i - 1]], out[idx[i + 1]]
            if mid in ("H", "A") and before == after and before in ("H", "A") and before != mid \
                    and e - s < k:
                out[s:e] = before
                changed = True
    return out


# -- degradation -----------------------------------------------------------------------------------
def barath_stats(cfg_deg):
    """Run lengths (frames) of the BAR-ATH ball visible / missing stretches and slot fragments,
    on graded frames. Tracking statistics only."""
    path = paths.resolve(cfg_deg["barath_cache"])
    if not os.path.isfile(path):
        return None
    from games import half_video_blocks
    c = np.load(path)
    bx, px = c["bx"], c["px"]
    g = np.zeros(bx.size, dtype=bool)
    for _h, a, b in half_video_blocks("BAR-ATH"):
        g[a:b + 1] = True
    vis = ~np.isnan(bx)
    frag = []
    for j in range(px.shape[1]):
        frag += [e - s for s, e in runs(~np.isnan(px[:, j]) & g)]
    return {"visible": np.array([e - s for s, e in runs(vis & g)]),
            "missing": np.array([e - s for s, e in runs(~vis & g)]),
            "fragments": np.array(frag)}


def _sample(rng, pool, quantiles_s):
    if pool is not None and pool.size:
        return int(rng.choice(pool))
    q = np.array(quantiles_s) * FPS                                  # piecewise-linear quantiles
    return int(max(1, np.interp(rng.random(), [0.1, 0.25, 0.5, 0.75, 0.9], q)))


def degrade(d, cfg_deg, stats, match_seed):
    rng = np.random.default_rng([int(cfg_deg["seed"]), match_seed])
    n = d["period"].size
    bx, by = d["ball_x"].copy(), d["ball_y"].copy()
    cb = cfg_deg["ball"]
    use = cb.get("resample_barath_runs", True) and stats is not None
    keep = np.zeros(n, dtype=bool)
    f, visible = 0, bool(rng.random() < 0.6)
    while f < n:
        if visible:
            k = _sample(rng, stats["visible"] if use else None, cb["fallback_visible_s"])
            keep[f:f + k] = True
        else:
            k = _sample(rng, stats["missing"] if use else None, cb["fallback_missing_s"])
        f += k
        visible = not visible
    z = np.nan_to_num(d["ball_z"], nan=0.0)
    keep &= z <= cb["aerial_z_m"]
    L, W = d["length"], d["width"]
    out = np.maximum.reduce([-bx, bx - L, -by, by - W])
    keep &= np.nan_to_num(out, nan=np.inf) <= cb["max_outside_m"]
    bx[~keep] = np.nan
    by[~keep] = np.nan

    cp = cfg_deg["players"]
    px, py = d["px"].copy(), d["py"].copy()
    frag = stats["fragments"] if (stats is not None and cp.get("resample_barath_fragments", True)) \
        else None
    share = float(cp["missing_share"])
    for j in range(px.shape[1]):
        f = int(rng.integers(0, FPS * 30))
        while f < n:
            k = int(rng.choice(frag)) if frag is not None and frag.size \
                else int(rng.exponential(cp["fallback_fragment_s"] * FPS)) + 1
            f += k
            gap = int(rng.exponential(max(1.0, k * share / (1.0 - share)))) + 1
            px[f:f + gap, j] = np.nan
            py[f:f + gap, j] = np.nan
            f += gap
    out = dict(d)
    out.update(ball_x=bx, ball_y=by, px=px, py=py)
    return out


# -- Match + pitch -----------------------------------------------------------------------------------
def pitch_for(d):
    geo = copy.deepcopy(load_geometry(paths.TEAM_CONFIG_DIR))
    geo["pitch"]["length"], geo["pitch"]["width"] = float(d["length"]), float(d["width"])
    return Pitch(geo)


def match_from(d, params):
    half = np.where(np.isin(d["period"], (1, 2)), d["period"], 0).astype(np.int8)
    m = Match(FPS, d["ball_x"], d["ball_y"], d["px"], d["py"], d["player_team"], half,
              smart_tagging=[], team_names=("home", "away"))
    return prepare_ball(m, params)


def ball_missing_share(d):
    g = np.isin(d["period"], (1, 2))
    return float(np.isnan(d["ball_x"][g]).mean())


# -- scoring ---------------------------------------------------------------------------------------
def score(gt, pred, bootstrap_n=0):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=bootstrap_n)
    keep = ("spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high", "macro_f1",
            "bootstrap_low", "bootstrap_high", "alive_dead_f1", "team_acc_gt_alive",
            "n_gt_spells", "n_pred_spells", "gt_mean_spell_duration_s", "mean_spell_duration_s")
    out = {k: r.get(k) for k in keep}
    pc = r.get("per_class")
    if isinstance(pc, dict) and "D" in pc:
        out["D_f1"] = pc["D"].get("f1") if isinstance(pc["D"], dict) else None
        out["D_precision"] = pc["D"].get("precision")
        out["D_recall"] = pc["D"].get("recall")
    elif isinstance(pc, (list, tuple)) and len(pc) >= 3:
        out["D_f1"] = pc[2][2] if isinstance(pc[2], (list, tuple)) else None
    return out, r


def pooled(gts, preds, bootstrap_n):
    """All matches back to back, separated by one X frame (the scorer never matches across it)."""
    sep = np.array(["X"])
    gt = np.concatenate([np.concatenate([g, sep]) for g in gts])
    pr = np.concatenate([np.concatenate([p, sep]) for p in preds])
    return score(gt, pr, bootstrap_n)


# -- the official baseline (starter/baseline.py), re-implemented on the arrays ---------------------
def baseline_labels(d, window):
    """Nearest tracked player's team where the ball is seen, D where it is not, then a centred
    majority vote over +/- window frames (ties to home) -- the same algorithm as
    starter/baseline.py, so it can be scored on the validation matches like our versions."""
    n = d["period"].size
    bx, by = d["ball_x"], d["ball_y"]
    dist = np.hypot(d["px"] - bx[:, None], d["py"] - by[:, None])
    seen = ~np.isnan(bx) & ~np.all(np.isnan(dist), axis=1)
    j = np.argmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    codes = np.full(n, -1, dtype=np.int8)
    codes[seen] = d["player_team"][j[seen]]
    home_c = np.concatenate(([0], np.cumsum(codes == 0)))
    away_c = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    home, away = home_c[hi] - home_c[lo], away_c[hi] - away_c[lo]
    lab = np.full(n, "D", dtype="<U1")
    ok = (home + away) > 0
    lab[ok] = np.where(home[ok] >= away[ok], "H", "A")
    lab[~np.isin(d["period"], (1, 2))] = "X"
    return lab


# -- a Smart Tagging-like feed for IDSSE (v3) --------------------------------------------------------
def restart_type(x, y, length, width):
    """Restart type from where the ball is when play resumes (metres, origin at a corner)."""
    hx, hy = length / 2.0, width / 2.0
    x, y = x - hx, y - hy
    if abs(x) < 1.5 and abs(y) < 1.5:
        return "kick off"
    if abs(abs(x) - hx) < 3.0 and abs(abs(y) - hy) < 3.0:
        return "corner kick"
    if abs(abs(y) - hy) < 2.0:
        return "throw-in"
    if abs(abs(x) - hx) < 6.5 and abs(y) < 10.0:
        return "goal kick"
    return "free kick"


def simulate_smart_tagging(d, cfg_sim, match_seed):
    """SET PIECES tags as BAR-ATH's Smart Tagging would give them, from the true restarts of a
    CLEAN IDSSE match (BallStatus 0 -> 1): each restart kept with the per-type recall of the
    BAR-ATH tagging, tag start = restart - lead (lead resampled from BAR-ATH), 35 s long.
    Uses the DFL labels of another game only to place the tags, like the degradation does."""
    rng = np.random.RandomState(int(cfg_sim["seed"]) + 1000 * match_seed)
    alive = d["ball_alive"].astype(bool)
    period = d["period"]
    L, W = float(d["length"]), float(d["width"])
    leads = np.asarray(cfg_sim["lead_s"], float)
    out = []
    for h in (1, 2):
        idx = np.flatnonzero(period == h)
        a = alive[idx]
        if a.size and a[0]:                     # the half starts with the kick-off itself
            start = int(idx[0]) - int(round(rng.uniform(*cfg_sim["kickoff_lead_s"]) * FPS))
            out.append({"code": "SET PIECES", "type": "kick off (start)", "team": None,
                        "tags": {}, "start_frame": start, "end_frame": start + 5 * FPS})
        for s, e in runs(~a):
            if e >= a.size:
                continue
            k = int(idx[e])
            typ = restart_type(float(d["ball_x"][k]), float(d["ball_y"][k]), L, W)
            if typ == "kick off":
                # BAR-ATH tags every kick-off: a 5 s window, the kick 3.8-4.9 s after its start;
                # a kick-off that is not the half's first follows a goal (GOALS tag before it)
                first = s == 0
                start = k - int(round(rng.uniform(*cfg_sim["kickoff_lead_s"]) * FPS))
                out.append({"code": "SET PIECES", "type": "kick off (start)" if first
                            else "kick off (after goal)", "team": None, "tags": {},
                            "start_frame": start, "end_frame": start + 5 * FPS})
                if not first:
                    g0 = max(int(idx[0]), k - int(round(cfg_sim["goal_tag_before_s"] * FPS)))
                    out.append({"code": "GOALS", "type": "", "team": None, "tags": {},
                                "start_frame": g0, "end_frame": start + FPS})
                continue
            if s == 0 or rng.rand() >= cfg_sim["recall"].get(typ, 0.0):
                continue
            start = max(int(idx[0]), k - int(round(rng.choice(leads) * FPS)))
            out.append({"code": "SET PIECES", "type": typ, "team": None, "tags": {},
                        "start_frame": start,
                        "end_frame": start + int(round(cfg_sim["length_s"] * FPS))})
    out.sort(key=lambda t: t["start_frame"])
    return out
