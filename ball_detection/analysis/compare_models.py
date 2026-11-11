"""
Comparison between the models, before and after post-processing, and of the ensemble (no
inference: it reads detections.csv, track.csv and creates ball.csv).

Without ground truth, behaviour and agreement are measured:

  A. Raw detections of each model (detections.csv)
     det%          frames with at least one detection
     on_pitch%     frames with at least one detection inside the pitch
     off_pitch%    share of the detections that fall outside the pitch (almost always false positives)
     imposs_jumps% consecutive frames in which the best on-pitch detection moves faster than
                   40 m/s (physically impossible: at least one of the two is wrong)
  B. Post-processed tracks of each model (track.csv, post_process.py)
     % chosen / recovered / synthetic detections / ball present, segments, impossible jumps
  C. Agreement between pairs of models, on the raw detections and on the tracks
     both%         frames in which both have a detection (on the pitch / in the track)
     agreeing%     frames in which they are within 15 px
     agreement%    agreeing / both
  D. Ensemble (ball_consensus.py): % confirmed / tracked / recovered / synthetic / ball present

Output: table on screen + runs/ball_pipeline/<video>/comparison.md (comparison_<models>.md if
--models is not the default one), ball.csv of the ensembles.

Usage (from the project root):
  python -m analysis.compare_models
  python -m analysis.compare_models --models tikitalia_tiles roboflow
"""
import argparse
import itertools
import math
from collections import Counter

import numpy as np

from utils.config import ALL_MODELS, DEFAULT_MODELS, set_video, ball_path
from utils.io import load_detections
from utils.pitch import Homographies
from inference.ball_consensus import add_params, consensus, summarize, write_csv, load_tracks, STATUSES


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def in_pitch(by_frame, models, hom, frames):
    """frame -> {model: [(x, y, conf)] inside the pitch, sorted by confidence}; and the off-pitch count."""
    res, outside = {}, Counter()
    for f in frames:
        d = {}
        for m, x, y, c in by_frame.get(f, []):
            if m not in models:
                continue
            if hom.has(f) and hom.on_pitch(f, x, y):
                d.setdefault(m, []).append((x, y, c))
            else:
                outside[m] += 1
        for m in d:
            d[m].sort(key=lambda t: -t[2])
        res[f] = d
    return res, outside


def jumps(positions, max_step):
    """(pairs of consecutive frames, how many with a displacement > max_step); positions: frame -> metres."""
    pairs = bad = 0
    for f, p in positions.items():
        q = positions.get(f + 1)
        if q is not None:
            pairs += 1
            bad += np.linalg.norm(q - p) > max_step
    return pairs, bad


def raw_table(models, by_frame, inp, outside, hom, frames, max_step):
    n = len(frames)
    rows = ["| model | det% | on_pitch% | off_pitch% | imposs_jumps% |", "|---|---|---|---|---|"]
    for m in models:
        det = sum(1 for f in frames if any(d[0] == m for d in by_frame.get(f, [])))
        inside = sum(1 for f in frames if m in inp[f])
        tot_det = sum(1 for f in frames for d in by_frame.get(f, []) if d[0] == m)
        pos = {f: hom.to_pitch(f, *inp[f][m][0][:2]) for f in frames if m in inp[f]}
        pairs, bad = jumps(pos, max_step)
        rows.append(f"| {m} | {pct(det, n)} | {pct(inside, n)} | {pct(outside[m], tot_det)} | {pct(bad, pairs)} |")
    return rows


def track_table(models, tracks, frames, max_step):
    n = len(frames)
    rows = ["| model | chosen detections | recovered | synthetic | **ball present** | segments | imposs_jumps% |",
            "|---|---|---|---|---|---|---|"]
    for m in models:
        t = tracks[m].loc[frames.start:frames.stop - 1]
        c = t["status"].value_counts()
        present = int((t["status"] != "none").sum())
        p = t[t["status"] != "none"]
        pos = {int(f): np.array([u, v]) for f, u, v in zip(p.index, p["pitch_x_m"], p["pitch_y_m"])}
        pairs, bad = jumps(pos, max_step)
        nseg = t.loc[t["status"] == "detected", "segment"].nunique()
        rows.append(f"| {m} | {pct(c.get('detected', 0), n)} | {pct(c.get('recovered', 0), n)} | "
                    f"{pct(c.get('interpolated', 0), n)} | **{pct(present, n)}** | {nseg} | {pct(bad, pairs)} |")
    return rows


def pair_table(models, inp, tracks, frames, agree_px):
    n = len(frames)
    rows = ["| pair | data | both% | agreeing% | agreement% |", "|---|---|---|---|---|"]
    for a, b in itertools.combinations(models, 2):
        both = agree = 0
        for f in frames:
            if a in inp[f] and b in inp[f]:
                both += 1
                agree += any(math.hypot(p[0] - q[0], p[1] - q[1]) <= agree_px
                             for p in inp[f][a] for q in inp[f][b])
        rows.append(f"| {a} + {b} | raw detections | {pct(both, n)} | {pct(agree, n)} | {pct(agree, both)} |")
        ta = tracks[a].loc[frames.start:frames.stop - 1]
        tb = tracks[b].loc[frames.start:frames.stop - 1]
        mask = (ta["status"] == "detected") & (tb["status"] == "detected")
        d = np.hypot(ta.loc[mask, "x"] - tb.loc[mask, "x"], ta.loc[mask, "y"] - tb.loc[mask, "y"])
        both, agree = int(mask.sum()), int((d <= agree_px).sum())
        rows.append(f"| {a} + {b} | tracks | {pct(both, n)} | {pct(agree, n)} | {pct(agree, both)} |")
    return rows


def ensemble_table(models, tracks, hom, frames, args):
    n = len(frames)
    rows = ["| ensemble | " + " | ".join(STATUSES) + " | **ball present** | none |",
            "|---|" + "---|" * (len(STATUSES) + 2)]
    results = {}
    for k in range(args.min_votes, len(models) + 1):
        for combo in itertools.combinations(models, k):
            out, _ = consensus({m: tracks[m] for m in combo}, hom, frames, args)
            write_csv(ball_path(combo), out, hom, frames)
            c = summarize(out, n)
            results[combo] = out
            rows.append(f"| {' + '.join(combo)} | " + " | ".join(pct(c[s], n) for s in STATUSES) +
                        f" | **{pct(c['present'], n)}** | {pct(c['none'], n)} |")
    return rows, results


def main():
    p = argparse.ArgumentParser(description="Comparison between models, tracks and ensembles")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    add_params(p)
    args = p.parse_args()
    models = [m for m in ALL_MODELS if m in args.models]
    out = set_video(args.video)

    by_frame, raw_frames = load_detections(models)
    tracks, frames = load_tracks(models)
    frames = range(max(frames.start, raw_frames.start), min(frames.stop, raw_frames.stop))
    hom = Homographies(args.homography)
    max_step = args.max_speed / args.fps
    print(f"[INFO] models {models}, frames {frames.start}-{frames.stop - 1} ({len(frames)})")

    inp, outside = in_pitch(by_frame, models, hom, frames)
    report = [f"# Model comparison: frames {frames.start}-{frames.stop - 1} ({len(frames)} frames)", "",
              "## A. Raw detections", ""] + raw_table(models, by_frame, inp, outside, hom, frames, max_step)
    report += ["", "## B. Post-processed tracks", ""] + track_table(models, tracks, frames, max_step)
    report += ["", "## C. Agreement between models", ""] + pair_table(models, inp, tracks, frames, args.agree_px)
    ens_rows, _ = ensemble_table(models, tracks, hom, frames, args)
    report += ["", "## D. Ensemble", ""] + ens_rows
    text = "\n".join(report)
    print("\n" + text)
    # the default models write comparison.md; other combinations write their own file
    report_name = "comparison.md" if models == DEFAULT_MODELS else f"comparison_{'+'.join(models)}.md"
    (out / report_name).write_text(text + "\n", encoding="utf-8")
    print(f"\n[INFO] Saved {out / report_name} and the ensemble's ball.csv")


if __name__ == "__main__":
    main()
