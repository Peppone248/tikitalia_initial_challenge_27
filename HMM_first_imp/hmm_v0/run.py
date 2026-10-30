"""The whole pipeline, one function per layer (MOTORE.md 2):

    dati grezzi -> Match -> features -> events -> regole -> MOTORE -> CSV

Nothing here reads any ground truth. `run()` returns everything for inspection; the CLI is
scripts/make_submission.py.
"""

import csv
import json
import os
import time

import numpy as np

from . import paths
from .config import Config
from . import diagnostics, engine, events, exact, frame_features, nn, noise, pseudo_labels, \
    segment_c, votes
from .rules import build_rules
from .submission import validate, write_csv

from tikitalia.events import attack_directions, kickoffs, kickoff_setup_mask  # noqa: E402
from tikitalia.features import compute_features  # noqa: E402
from tikitalia.geometry import Pitch  # noqa: E402
from tikitalia.preprocessing import load_match  # noqa: E402


def _log(msg, t0=[None]):
    if t0[0] is None:
        t0[0] = time.time()
    print("[{:6.1f}s] {}".format(time.time() - t0[0], msg), flush=True)


def build_context(match, pitch, cfg):
    """Layers 2-3: features (team code), noise, events."""
    p = cfg.params
    ko = kickoffs(match, pitch, p)
    attack_dir, method = attack_directions(match, pitch, p, ko)
    feats = compute_features(match, pitch, p, attack_dir)
    nz = noise.NoiseModel(noise.sources_from(match, feats, pitch), cfg.hmm["noise"]["features"],
                          cfg.hmm["noise"]["mode"], cfg.hmm["noise"]["window"])
    events.estimate_acc_kick(match, feats, p)
    episodes = events.control_episodes(match, feats, pitch, p, nz)
    chains = events.pass_chains(episodes, match, p)
    restarts = events.restart_kicks(match, feats, p)
    restarts += [{"frame": int(k["kick"]), "kind": "kickoff", "strength": 1.0}
                 for k in ko if not k.get("rejected")]
    restarts.sort(key=lambda r: r["frame"])
    return {"kickoffs": ko, "attack_dir": attack_dir, "attack_method": method,
            "features": feats, "noise": nz, "control_episodes": episodes,
            "pass_chains": chains, "restarts": restarts, "stage_outputs": {}}


def apply_rules(match, pitch, context, cfg):
    """Layer 4: every enabled rule, stage 1 then stage 2 (which may read stage 1)."""
    rules = build_rules(cfg.rules, cfg.params)
    outputs = {}
    for r in rules:
        outputs[r.id] = r.fires(match, pitch, context)
        if r.stage == 1:
            context["stage_outputs"][r.id] = outputs[r.id]
    return rules, outputs


def run_match(match, pitch, cfg, log=lambda m: None):
    """Layers 2-6 on a prepared Match: features, events, rules, votes, NN, HMM. No file I/O.
    Used for BAR-ATH (run) and for the validation matches (validation/)."""
    log("2/7 features + noise (mode {}) + events".format(cfg.hmm["noise"]["mode"]))
    context = build_context(match, pitch, cfg)

    log("3/7 rules")
    rules, outputs = apply_rules(match, pitch, context, cfg)

    log("4/7 votes (w = q * s) -> evidence")
    E, dead, weights = votes.collect(rules, outputs, match.n, cfg.hmm["votes"])
    restart_frames = votes.restart_frames(outputs, rules, context)

    engine_kind = cfg.pipeline.get("engine", "hmm")
    need_hmm = engine_kind == "hmm" or cfg.segment.get("dead", {}).get("source") == "hmm"
    target = P = prior = macro = hmm = None
    target_counts, nn_info, names, eng_info = {}, {}, [], {}
    if need_hmm:
        labels, macro, hmm, eng_info, target, target_counts, nn_info, names, P, prior = \
            _hmm_path(match, pitch, cfg, context, outputs, E, restart_frames, log)
    if engine_kind == "segment":
        log("6/7 MOTORE (approach C): dead ball -> boundaries at events -> team per segment")
        hmm_labels = labels if need_hmm else None
        labels, seg_info = segment_c.decide(match, context, rules, outputs, E, cfg.segment,
                                            hmm_labels, p=cfg.params, pitch=pitch)
        eng_info = {"approach": "C (segment then assign)", "segment": seg_info,
                    "hmm_used_for_dead": bool(need_hmm)}
    return _finish(match, cfg, context, rules, outputs, E, dead, weights, target, target_counts,
                   P, prior, nn_info, names, labels, macro, hmm, eng_info)


def _hmm_path(match, pitch, cfg, context, outputs, E, restart_frames, log):
    log("5/7 NN: pseudo-labels from the rules -> train -> P(H, A, D, X)")
    target, target_counts = pseudo_labels.make(E, match.graded, cfg.hmm["pseudo_labels"])
    ko_mask, _ = kickoff_setup_mask(match, pitch, cfg.params)
    poss_codes = cfg.rules.get("S-PHASE", {}).get("options", {}).get("possession_codes", [])
    def_codes = cfg.rules.get("S-PHASE", {}).get("options", {}).get("defending_codes", [])
    sp_mask, st_team = frame_features.st_masks(match, poss_codes, def_codes)
    if not cfg.hmm["nn"].get("st_set_piece_feature", True):      # v3 option
        sp_mask = np.zeros_like(sp_mask)
    X, names, swap = frame_features.build(match, context["features"], pitch, cfg.params,
                                          ko_mask.astype(np.float32), sp_mask, st_team)
    if cfg.hmm["nn"].get("exact_arithmetic", True):
        X = exact.quantize(X, nn.INPUT_BITS)          # ulp differences of the features vanish
    model, prior, nn_info = nn.train(X, swap, target, cfg.hmm["nn"], log=log)
    P = nn.predict(model, X, int(cfg.hmm["nn"]["context_frames"]), swap=swap,
                   decimals=cfg.hmm["nn"].get("output_decimals"))

    log("6/7 MOTORE: Bakis HMM + Viterbi per half")
    change_pts = votes.change_points(
        outputs, (cfg.hmm["hmm"].get("team_change", {}) or {}).get("sources", []))
    labels, macro, hmm, eng_info = engine.decide(match, P, prior, E, restart_frames,
                                                 cfg.hmm["hmm"], log=log, change_pts=change_pts)
    return labels, macro, hmm, eng_info, target, target_counts, nn_info, names, P, prior


def _finish(match, cfg, context, rules, outputs, E, dead, weights, target, target_counts, P,
            prior, nn_info, names, labels, macro, hmm, eng_info):
    rule_rows = votes.summary(rules, outputs, weights, dead, match.graded)
    summary = {
        "match": {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in match.diagnostics.items()},
        "engine_kind": cfg.pipeline.get("engine", "hmm"),
        "noise_mode": cfg.hmm["noise"]["mode"], "noise": context["noise"].summary(),
        "acc_kick_m_s2": round(cfg.params.ACC_KICK, 2),
        "events": {"control_episodes": len(context["control_episodes"]),
                   "control_episodes_>=T_CONTROL": sum(
                       1 for e in context["control_episodes"]
                       if e["dur_s"] >= cfg.params.T_CONTROL),
                   "pass_chains": len(context["pass_chains"]),
                   "restart_kicks": sum(1 for r in context["restarts"]
                                        if r["kind"] == "still_kick"),
                   "kickoffs": sum(1 for r in context["restarts"] if r["kind"] == "kickoff")},
        "rules": rule_rows,
        "pseudo_labels": target_counts,
        "nn": nn_info, "nn_inputs": names,
        "engine": eng_info,
        "output": diagnostics.output_summary(labels, match, match.fps),
        "fragmentation": diagnostics.fragmentation(
            labels, match, outputs, ["P-DRIB", "P-FREE", "P-PASS", "T-INTERCEPT", "T-REV",
                                     "R-TOUCH", "R-KO"], match.fps),
        "anchor_agreement": diagnostics.anchor_agreement(
            labels, rules, outputs, weights, match.graded, cfg.hmm["votes"]["conf_anchor"]),
        "overridden_team_params": cfg.overridden_team_params,
    }
    return {"context": context, "rules": rules, "outputs": outputs, "E": E, "dead": dead,
            "weights": weights, "target": target, "P": P, "prior": prior, "labels": labels,
            "macro": macro, "hmm": hmm, "summary": summary}


def run(config_dir=None, overrides=None, out_dir=None, quiet=False):
    log = (lambda m: None) if quiet else _log
    cfg = Config(config_dir, overrides)
    pc = cfg.pipeline
    out_dir = paths.resolve(out_dir or pc["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    log("1/7 Match: loading {} ...".format(pc["game"]))
    pitch = Pitch.from_config(paths.TEAM_CONFIG_DIR)
    match = load_match(pitch, cfg.params, pc["game"], paths.resolve(pc.get("data_dir")),
                       paths.resolve(pc.get("cache_path")))

    res = run_match(match, pitch, cfg, log)
    labels = res["labels"]

    log("7/7 CSV + validation")
    csv_path = os.path.join(out_dir, "{}.csv".format(pc["team_name"]))
    write_csv(labels, csv_path)
    ok, val_out = validate(csv_path)
    if pc.get("copy_to_evaluation_results"):
        write_csv(labels, os.path.join(paths.REPO_ROOT, "evaluation", "results",
                                       "{}_hmm_v0.csv".format(pc["team_name"])))
    summary = dict({"csv": csv_path, "validation_ok": ok}, **res["summary"])
    res["summary"] = summary
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2, default=_jsonable)
    with open(os.path.join(out_dir, "run_config_snapshot.json"), "w", encoding="utf-8") as h:
        json.dump(cfg.snapshot(), h, indent=2, default=_jsonable)
    if pc.get("write_episodes", True):
        _write_episodes(res["outputs"], os.path.join(out_dir, "episodes"))
    log("done: {} (validation {})".format(csv_path, "OK" if ok else "FAILED"))
    res.update({"cfg": cfg, "match": match, "pitch": pitch, "validation": val_out,
                "csv": csv_path})
    return res


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_episodes(outputs, folder):
    """Episode lists per rule (frame numbers): local debugging only, under outputs/ (gitignored)."""
    os.makedirs(folder, exist_ok=True)
    for rid, out in outputs.items():
        eps = [e for e in out.episodes if "start" in e]
        if not eps:
            continue
        keys = sorted({k for e in eps for k in e})
        with open(os.path.join(folder, rid + ".csv"), "w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=keys)
            w.writeheader()
            for e in eps:
                w.writerow({k: _jsonable(v) if isinstance(v, (np.generic, np.ndarray)) else v
                            for k, v in e.items()})
