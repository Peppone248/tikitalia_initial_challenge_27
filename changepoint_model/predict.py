"""Label the provided match: writes the submission file.

    python changepoint_model/predict.py                       # -> outputs/tikitalia.csv

Steps: read the tactical tracking (games.asset_paths(): data/ or MTAH_DATA_DIR), compute the 134
frame features, run the trained heads (models/cpm_v4_hybrid.pkl), decode H/A/D with the
boundary-driven Viterbi, then two rules from the Laws of the Game: no play before each kick-off
(Law 8, from the Smart Tagging kick-off tag) and play stops at the referee's whistle (Law 5, from
the audio of the tactical video). Writes frame,label for every video frame (X outside the two
halves) and prints shares, possession count and the SHA-256 of the file.

The audio is extracted once with ffmpeg next to the video (BAR-ATH_tactical_audio.flac).
Any EPTS pair with labels can be labelled instead with --meta / --raw (e.g. a Metrica sample
game): the score against its own labels is printed.

Deterministic: fixed models, numpy inference, and the numerical libraries pinned to one thread
below (before numpy is imported) so that every sum runs in the same order on every machine.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

try:  # the heads were pickled with scikit-learn 1.9.1; 1.7.2 gives the same probabilities to the bit
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:  # pragma: no cover
    pass
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", message="Mean of empty slice")

PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(PKG)
sys.path.insert(0, PKG)
sys.path.insert(0, REPO)

from cpm import pipeline  # noqa: E402
from cpm.kickoff import apply_kickoff  # noqa: E402
from cpm.loaders.epts import load_epts  # noqa: E402
from cpm.match import H, A, D, X, spells  # noqa: E402
from cpm import whistle  # noqa: E402

LETTERS = np.array(["H", "A", "D", "X"])
MODEL = os.path.join(PKG, "models", "cpm_v4_hybrid.pkl")


def load_provided_match(paths, raw=None):
    from games import half_video_blocks, GAME
    blocks = half_video_blocks(GAME)

    def period_of_frame(frame_id):
        v = frame_id - 1
        for half, first, last in blocks:
            if first <= v <= last:
                return half
        return 0
    return load_epts(paths["atd_meta"], raw or paths["atd_raw"], name=GAME, period_of_frame=period_of_frame)


def audio_file(video, audio):
    """the match audio as 16 kHz mono FLAC, extracted once with ffmpeg."""
    audio = audio or os.path.join(os.path.dirname(video), "BAR-ATH_tactical_audio.flac")
    if not os.path.isfile(audio):
        print("extracting the audio -> {}".format(audio), flush=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "flac", audio], check=True)
    return audio


def write_csv(path, labels, frame_index, n_frames):
    out = np.full(n_frames, X, dtype=np.int8)
    ok = (frame_index >= 0) & (frame_index < n_frames)
    out[frame_index[ok]] = labels[ok]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join("outputs", "tikitalia.csv"))
    ap.add_argument("--models", default=MODEL)
    ap.add_argument("--audio", help="audio of the tactical video (default: extracted next to the video)")
    ap.add_argument("--no-kickoff", action="store_true", help="skip the kick-off rule (Law 8)")
    ap.add_argument("--no-whistle", action="store_true", help="skip the whistle rule (Law 5)")
    ap.add_argument("--atd-raw", help="another tracking file of the provided match, same format and metadata")
    ap.add_argument("--meta", help="EPTS metadata XML of another match (with --raw)")
    ap.add_argument("--raw", help="EPTS raw tracking of another match (with --meta)")
    args = ap.parse_args()
    t0 = time.time()

    provided = not (args.meta and args.raw)
    if provided:
        from games import asset_paths, source_frames
        paths = asset_paths()
        m = load_provided_match(paths, args.atd_raw)
        n_frames = source_frames()
    else:
        m = load_epts(args.meta, args.raw)
        n_frames = int(m.frame_index.max() + 1)
    print("loaded {}: {} frames, ball seen {:.1%}, attack {}".format(
        m.name, m.T, np.mean(~np.isnan(m.ball[:, 0])), m.attack), flush=True)

    models = pipeline.load_models(args.models)
    P = pipeline.prepare(m, "clean")
    labels = pipeline.decode(pipeline.predict(models, P), P.period, models, fps=m.fps)

    kickoff_info = whistle_info = None
    if provided and not args.no_kickoff:
        sys.path.insert(0, os.path.join(REPO, "starter"))
        import load_smart_tagging
        labels, kickoff_info = apply_kickoff(labels, m, load_smart_tagging.load())
        print("kick-off (Law 8):", json.dumps(kickoff_info), flush=True)
    if provided and not args.no_whistle:
        pcm = whistle.read_audio(audio_file(paths["video"], args.audio))
        blasts = whistle.detect_blasts(pcm, n_frames)
        labels, whistle_info, _ = whistle.whistle_stop(labels, m, blasts)
        print("whistle (Law 5):", json.dumps(whistle_info), flush=True)

    out = write_csv(args.out, labels, m.frame_index, n_frames)
    graded = out != X
    sp = [s for s in spells(out, graded.astype(int)) if s[0] in (H, A)]
    summary = {"file": args.out, "sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest(),
               "rows": int(n_frames), "graded": int(graded.sum()),
               "share_H": round(float(np.mean(out[graded] == H)), 4),
               "share_A": round(float(np.mean(out[graded] == A)), 4),
               "share_D": round(float(np.mean(out[graded] == D)), 4),
               "possessions": len(sp), "kickoff": kickoff_info, "whistle": whistle_info,
               "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=1))
    if m.labels is not None:
        from cpm.evaluate import score
        gt = m.labels.copy()
        gt[~m.graded()] = X
        print("score against the match's own labels:", json.dumps(score(gt, labels), default=float))


if __name__ == "__main__":
    main()
