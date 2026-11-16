"""Fit, predict and decode: the whole model in one place.

    prepared = prepare(match, scenario, seed)           # degrade + features + targets
    models   = fit([prepared_i for i in train])          # team, dead, change, stop, restart heads
    probs    = predict(models, prepared_j)               # per-frame probabilities
    labels   = decode(probs, prepared_j.period, cfg)     # boundary-driven Viterbi

Everything a prediction needs after training is in `models` (JSON for the logistic parts, a
pickle for the gradient-boosted heads) and in the a priori constants of the modules.
"""

import pickle

import numpy as np

from .match import H, A, D, X
from .degrade import degrade
from .features import frame_features
from .models import TeamModel, EvenModel, boundary_targets, even_inputs, BOUNDARY_K
from . import decode as dec


class Prepared:
    def __init__(self, m, even, odd, even_names, odd_names):
        self.match = m
        self.even = even
        self.odd = odd
        self.even_names = even_names
        self.odd_names = odd_names
        self.period = m.period
        self.labels = m.labels
        self.graded = m.graded() & ((m.labels != X) if m.labels is not None else True)


def prepare(m, scenario="clean", seed=2027):
    """Degrade (unless the source is already automatic tracking: PFF is used as given) and
    compute the features."""
    if m.extra.get("source", "").lower().startswith("pff"):
        scenario = "clean"
    dm = degrade(m, scenario, seed)
    even, odd, en, on = frame_features(dm)
    return Prepared(dm, even, odd, en, on)


DEFAULT_CFG = dict(k=BOUNDARY_K, gamma=1.0, gamma_sr=None, w=1.0, backend="hgb", team_l2=1.0,
                   dead_l2=1.0, subsample=3, min_frames=None, hazard="boundary", stack=True,
                   heads="frame", seq=None)

STACK_WINDOWS_S = (0.5, 1.0, 2.0, 4.0)


def stacked_inputs(P, team_model, fps=25):
    """Even inputs of the second stage: the even features, |odd|, and the trajectory of the
    team model's own output (how much P(home) moves across the frame, and how sure it is)."""
    from .features import _box_past, _box_next, _box
    base = even_inputs(P.even, P.odd)
    if team_model is None:
        return base
    ph = team_model.predict_proba(P.odd)
    cols = [np.abs(ph - 0.5) * 2]                      # confidence (even)
    for w in STACK_WINDOWS_S:
        wf = int(round(w * fps))
        before = _box_past(ph, wf) / wf
        after = _box_next(ph, wf) / wf
        cols.append(np.abs(after - before))             # a change moves P(home)
        cols.append(-((before - 0.5) * (after - 0.5)).clip(max=0) * 4)   # sign flip
        cols.append(np.abs(_box(ph, wf) / (2 * wf + 1) - 0.5) * 2)       # local confidence
    return np.concatenate([base, np.stack(cols, axis=1).astype(np.float32)], axis=1)


def fit(prepared_list, cfg=None, verbose=False, log=print):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    if cfg["heads"] == "seq":
        from .seq import train_seq
        seq_cfg = dict(cfg["seq"] or {})
        seq_cfg.setdefault("k", cfg["k"])
        sd = train_seq(prepared_list, seq_cfg, verbose=verbose, log=log)
        return {"cfg": cfg, "seq": sd, "prior": np.array([1 / 3.0, 1 / 3.0, 1 / 3.0]),
                "even_names": prepared_list[0].even_names, "odd_names": prepared_list[0].odd_names}
    step = cfg["subsample"]
    odd_tr, yh_tr = [], []
    ev_tr, yd_tr, ych, yst, yre = [], [], [], [], []
    for P in prepared_list:
        g = P.graded
        lab = P.labels
        alive = g & (lab < D)
        idx_a = np.flatnonzero(alive)[::step]
        odd_tr.append(P.odd[idx_a])
        yh_tr.append((lab[idx_a] == H).astype(float))
    odd_tr = np.concatenate(odd_tr)
    yh_tr = np.concatenate(yh_tr)
    models = {"cfg": cfg}
    # the team: the symmetric logistic. Swap augmentation is implicit (no intercept, odd inputs).
    models["team"] = TeamModel(cfg["team_l2"]).fit(odd_tr, yh_tr, prepared_list[0].odd_names)
    team_for_stack = models["team"] if cfg["stack"] else None
    for P in prepared_list:
        g = P.graded
        lab = P.labels
        idx_g = np.flatnonzero(g)[::step]
        ev_tr.append(stacked_inputs(P, team_for_stack)[idx_g])
        yd_tr.append(lab[idx_g] == D)
        ch, st, re = boundary_targets(lab, P.period, cfg["k"])
        ych.append(ch[idx_g])
        yst.append(st[idx_g])
        yre.append(re[idx_g])
    Xe = np.concatenate(ev_tr)
    yd = np.concatenate(yd_tr)
    be = cfg["backend"]
    models["dead"] = EvenModel(be, l2=cfg["dead_l2"]).fit(Xe, yd)
    models["change"] = EvenModel(be, l2=cfg["dead_l2"]).fit(Xe, np.concatenate(ych))
    models["stop"] = EvenModel(be, l2=cfg["dead_l2"]).fit(Xe, np.concatenate(yst))
    models["restart"] = EvenModel(be, l2=cfg["dead_l2"]).fit(Xe, np.concatenate(yre))
    # class priors over the graded frames
    labs = np.concatenate([P.labels[P.graded] for P in prepared_list])
    prior = np.array([np.mean(labs == H), np.mean(labs == A), np.mean(labs == D)])
    models["prior"] = prior
    models["even_names"] = prepared_list[0].even_names
    models["odd_names"] = prepared_list[0].odd_names
    return models


def predict(models, P):
    if models["cfg"].get("heads") == "seq":
        from .seq import predict_seq
        return predict_seq(models["seq"], P)
    if models["cfg"].get("heads") == "hybrid":
        # v4: P(home) from the GRU, dead ball and boundaries from the boosted trees
        from .seq import predict_seq
        out = predict(models["trees"], P)
        out["p_home"] = predict_seq(models["seq"], P)["p_home"]
        return out
    Xe = stacked_inputs(P, models["team"] if models["cfg"]["stack"] else None)
    out = {"p_home": models["team"].predict_proba(P.odd),
           "p_dead": models["dead"].predict_proba(Xe),
           "p_change": models["change"].predict_proba(Xe),
           "p_stop": models["stop"].predict_proba(Xe),
           "p_restart": models["restart"].predict_proba(Xe)}
    return out


def decode(probs, period, models, gamma=None, w=None, hazard=None, min_frames=None, fps=25,
           gamma_sr=None):
    cfg = models["cfg"]
    gamma = cfg["gamma"] if gamma is None else gamma
    gamma_sr = cfg["gamma_sr"] if gamma_sr is None else gamma_sr
    gamma_sr = gamma if gamma_sr is None else gamma_sr
    w = cfg["w"] if w is None else w
    hazard = cfg["hazard"] if hazard is None else hazard
    T = len(period)
    logE = dec.emissions(probs["p_home"], probs["p_dead"], models["prior"], w=w)
    if hazard == "constant":
        hc, hs, hr = dec.constant_hazards(T, fps=fps)
        hc, hs, hr = hc * gamma, hs * gamma, hr * gamma
    else:
        hc, _, _ = dec.hazards_from_boundaries(probs["p_change"], probs["p_stop"],
                                              probs["p_restart"], k=cfg["k"], gamma=gamma)
        _, hs, hr = dec.hazards_from_boundaries(probs["p_change"], probs["p_stop"],
                                                probs["p_restart"], k=cfg["k"], gamma=gamma_sr)
    return dec.decode_match(logE, hc, hs, hr, period, min_frames or cfg["min_frames"], fps=fps)


def save_models(models, path):
    with open(path, "wb") as fh:
        pickle.dump(models, fh, protocol=4)


def load_models(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
