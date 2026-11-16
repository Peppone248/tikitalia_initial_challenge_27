"""Label the provided match from its ATD: the submission file.

    python changepoint_model/scripts/predict_atd.py --models changepoint_model/models/cpm_v4_hybrid.pkl \
        --whistle HMM_first_imp/outputs/whistles/BAR-ATH_whistles_lit.json --out outputs/cpm_v42/tikitalia.csv

Reads the tactical ATD through games.asset_paths() (data/ or MTAH_DATA_DIR), assigns the two
halves from games.half_video_blocks() (the graded frames), computes the features, runs the
classifiers and the boundary-driven Viterbi, writes frame,label for all 199319 video frames (X
outside the halves) and prints shares, possession count and the SHA-256 of the file. Two runs give
the same file: everything is deterministic (numpy, fixed models, no threads in the decoder). The
numerical libraries are pinned to one thread below, before numpy is imported, so that every sum is
done in the same order on every machine (RULES.md 3: the same file, not approximately).

Any EPTS pair can be labelled instead with --meta / --raw (periods from the metadata), e.g. a
Metrica sample game, to check the pipeline end to end on a match with labels.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import warnings

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

# The boosted heads were pickled with scikit-learn 1.9.1; with 1.7.2 the probabilities were checked
# to be identical to the bit (Metrica game 3, same SHA-256), so the version warning is silenced.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:  # pragma: no cover
    pass
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", message="Mean of empty slice")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
sys.path.insert(0, PKG)
sys.path.insert(0, REPO)

from cpm.loaders.epts import load_epts  # noqa: E402
from cpm import pipeline  # noqa: E402
from cpm.match import H, A, D, X, spells  # noqa: E402

LETTERS = np.array(["H", "A", "D", "X"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(PKG, "models", "cpm_v1.pkl"))
    ap.add_argument("--out", default="outputs/cpm_v1/tikitalia.csv")
    ap.add_argument("--meta", help="EPTS metadata XML (default: the provided tactical ATD)")
    ap.add_argument("--raw", help="EPTS raw text (default: the provided tactical ATD)")
    ap.add_argument("--cache", help="npz cache of the parsed ATD (optional, speeds up re-runs)")
    ap.add_argument("--no-kickoff", action="store_true",
                    help="do not set D before each period's kick-off (Law 8, cpm/kickoff.py)")
    ap.add_argument("--whistle", default=None,
                    help="blasts JSON (HMM_first_imp/scripts/detect_whistles.py): applies R7a on top "
                         "(off by default; not measurable on the external matches, a team decision)")
    ap.add_argument("--broadcast", default=os.path.join(REPO, "HMM_first_imp", "outputs", "broadcast",
                                                        "broadcast_on_tactical.npz"),
                    help="label-free check of R7a: broadcast on a cut on the frames it adds")
    ap.add_argument("--n-frames", type=int, default=None,
                    help="rows of the submission (default: the video length from games.py)")
    args = ap.parse_args()
    t0 = time.time()
    if args.meta and args.raw:
        m = load_epts(args.meta, args.raw, cache=args.cache)
        n_frames = args.n_frames or int(m.frame_index.max() + 1)
    else:
        from games import asset_paths, half_video_blocks, GAME, source_frames
        paths = asset_paths()
        blocks = half_video_blocks(GAME)

        def period_of_frame(frame_id):
            v = frame_id - 1
            for half, first, last in blocks:
                if first <= v <= last:
                    return half
            return 0
        m = load_epts(paths["atd_meta"], paths["atd_raw"], name=GAME,
                      period_of_frame=period_of_frame, cache=args.cache)
        n_frames = args.n_frames or source_frames()
    print("loaded {}: {} frames, {} player slots, ball seen {:.1%}, attack {} ({:.0f} s)".format(
        m.name, m.T, m.n_players, np.mean(~np.isnan(m.ball[:, 0])), m.attack, time.time() - t0),
        flush=True)
    models = pipeline.load_models(args.models)
    P = pipeline.prepare(m, "clean")
    probs = pipeline.predict(models, P)
    labels = pipeline.decode(probs, P.period, models, fps=m.fps)
    kickoff_info = None
    if not args.no_kickoff:
        from cpm.kickoff import apply_kickoff
        tags = None
        if not (args.meta and args.raw):
            try:
                sys.path.insert(0, os.path.join(REPO, "starter"))
                import load_smart_tagging
                tags = load_smart_tagging.load()
            except Exception as exc:  # noqa: BLE001
                print("Smart Tagging not read ({}): kick-off from the tracking only".format(exc))
        labels, kickoff_info = apply_kickoff(labels, m, tags)
        print("kick-off (Law 8):", json.dumps(kickoff_info))
    whistle_info = None
    if args.whistle:
        from cpm.rule_layers import whistle_stop
        with open(args.whistle) as fh:
            blasts = json.load(fh)["blasts"]
        before = labels.copy()
        labels, whistle_info, added = whistle_stop(labels, m, blasts)
        g = m.graded() & (before != X)
        seen = ~np.isnan(m.ball[:, 0])
        with np.errstate(invalid="ignore"):
            ps = np.hypot(np.gradient(m.px, axis=0), np.gradient(m.py, axis=0)) * m.fps
            ps[ps > 12] = np.nan
            pmean = np.nanmean(ps, axis=1)

        def stats(mask):
            if not mask.any():
                return None
            return {"frames": int(mask.sum()), "player_speed": round(float(np.nanmean(pmean[mask])), 2),
                    "ball_seen": round(float(seen[mask].mean()), 3)}
        whistle_info["label_free_check"] = {"frames_added_by_R7a": stats(added),
                                            "v4_alive": stats(g & (before < D) & ~added),
                                            "v4_dead": stats(g & (before == D))}
        if args.broadcast and os.path.isfile(args.broadcast):
            cut = np.load(args.broadcast)["cut"].astype(bool)
            idx = np.minimum(m.frame_index, cut.size - 1)
            c = cut[idx] & (m.frame_index < cut.size)
            whistle_info["broadcast_cut_share"] = {
                "frames_added_by_R7a": round(float(c[added].mean()), 3) if added.any() else None,
                "v4_alive": round(float(c[g & (before < D)].mean()), 3),
                "v4_dead": round(float(c[g & (before == D)].mean()), 3)}
        print("whistle R7a:", json.dumps(whistle_info))
    out = np.full(n_frames, X, dtype=np.int8)
    rows = m.frame_index
    ok = (rows >= 0) & (rows < n_frames)
    out[rows[ok]] = labels[ok]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(out))
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    graded = out != X
    n = graded.sum()
    sp = [s for s in spells(out, (out != X).astype(int)) if s[0] in (H, A)]
    summary = {"file": args.out, "sha256": sha, "n_rows": int(n_frames), "graded": int(n),
               "share_H": float(np.mean(out[graded] == H)), "share_A": float(np.mean(out[graded] == A)),
               "share_D": float(np.mean(out[graded] == D)), "possessions": len(sp),
               "mean_possession_s": float(np.mean([s[2] - s[1] for s in sp]) / m.fps) if sp else 0.0,
               "dead_spells": sum(1 for s in spells(out, (out != X).astype(int)) if s[0] == D),
               "kickoff": kickoff_info,
               "whistle": whistle_info,
               "models": os.path.basename(args.models), "gamma": models["cfg"]["gamma"],
               "w": models["cfg"]["w"], "seconds": time.time() - t0}
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    if m.labels is not None:
        from cpm.evaluate import score
        gt = m.labels.copy()
        gt[~m.graded()] = X
        print("score against the match's own labels:", json.dumps(score(gt, labels), default=float))


if __name__ == "__main__":
    main()
