"""
Pipeline step 2: post-processing of the detections of ONE model -> clean ball track.

Input:  runs/ball_pipeline/<model>/detections.csv (detect_ball.py: up to 3 candidates per frame)
Output: runs/ball_pipeline/<model>/track.csv, ONE row per frame:
    frame                 index in the video
    x, y                  position in pixels (empty if no ball)
    pitch_x_m, pitch_y_m  position on the pitch in metres
    status                detected | recovered | interpolated | none
    conf                  confidence of the chosen detection (detected / recovered)
    segment               id of the continuous track segment (-1 = point between two different segments)

Algorithm:
  1. Candidates: the detections inside the pitch (homography, 3 m margin), converted to metres.
  2. Choosing the ball (there is ONE ball): for every frame one of the candidates or "none" is
     chosen, optimising the WHOLE sequence together (Viterbi algorithm). The score of a
     sequence is the sum, over the chosen detections, of:
       + logit(confidence)          evidence from the detector (negative below 0.5)
       + continuity bonus           if the detection continues the previous track...
       - linearity penalty          ...the higher the further it is from the position predicted
                                    at constant velocity (max --kick-cap: a kick or a deflection
                                    change direction abruptly, but remain possible)
     A step is accepted as a continuation only if it is physically reachable
     (≤ --max-speed). A track survives --coast-s seconds without detections; beyond that it restarts.
     So, among several candidates in the same frame, the one consistent with the ball's motion
     wins, and a low-confidence candidate that sits exactly on the trajectory is kept.
  3. Segments with fewer than --min-dets detections are discarded (a real ball is seen for more
     than an instant).
  4. Gaps: filled with fill_gaps (utils/pitch.py): up to 0.5 s always, with a spline in metres
     on the pitch; up to 2 s only if the ball is slow (≤ 6 m/s), with a straight line.
     A discarded candidate that falls on the estimated trajectory is recovered ('recovered').

Parameters:
  --cont-bonus 2.0  a detection that continues a plausible trajectory is worth +2 logit:
                    even at conf 0.15 (logit -1.7, the minimum threshold) it is kept if it is on
                    the trajectory; a weak, isolated detection is not
  --sigma-m 0.5     m: typical deviation from the predicted position (detection and homography noise)
  --kick-cap 1.0    maximum penalty for a change of direction: smaller than the bonus, so a kick
                    does not break the track
  --coast-s 0.5     a track stays "alive" for up to half a second without detections
  --min-dets 3      a segment must have at least 3 detections (0.12 s)

Usage (from the project root):
  python -m inference.post_process                  # the default models
  python -m inference.post_process --model tikitalia_model
"""
import argparse
import csv
import math
from collections import Counter

import numpy as np
import pandas as pd

from utils.config import ALL_MODELS, DEFAULT_MODELS, detections_path, track_path, set_video
from utils.io import processed_range
from utils.pitch import Homographies, add_common_params, fill_gaps


def add_track_params(p):
    p.add_argument("--cont-bonus", type=float, default=2.0)
    p.add_argument("--sigma-m", type=float, default=0.5)
    p.add_argument("--kick-cap", type=float, default=1.0)
    p.add_argument("--coast-s", type=float, default=0.5)
    p.add_argument("--min-dets", type=int, default=3)


def logit(c):
    c = min(max(c, 1e-3), 1 - 1e-3)
    return max(-4.0, min(4.0, math.log(c / (1 - c))))


def load_candidates(model, hom):
    """frame -> [(x, y, conf, pitch_x, pitch_y)] inside the pitch; and the range of processed frames."""
    df = pd.read_csv(detections_path(model))
    a, b = processed_range(model)
    cands = {}
    for f, x, y, c in zip(df["frame"].to_numpy(), df["x"].to_numpy(), df["y"].to_numpy(), df["conf"].to_numpy()):
        f = int(f)
        if hom.has(f) and hom.on_pitch(f, x, y):
            u, v = hom.to_pitch(f, x, y)
            cands.setdefault(f, []).append((float(x), float(y), float(c), float(u), float(v)))
    return cands, range(a, b + 1)


def select_path(cands, frames, args):
    """Viterbi. State per frame: 0 = no ball, i = candidate i-1.
    Each state remembers the position of the last detection, the velocity and the frames without
    detections (standard approximation: the velocity is that of the best path reaching the state).
    Returns frame -> (candidate index, new_segment)."""
    max_step = args.max_speed / args.fps
    coast = round(args.coast_s * args.fps)
    two_s2 = 2 * args.sigma_m ** 2
    # state: (score, last position (u, v) or None, velocity (du, dv) or None, frames without detections)
    prev = [(0.0, None, None, 0)]
    back = []
    for f in frames:
        best_i = max(range(len(prev)), key=lambda i: prev[i][0])
        s0, last0, v0, gap0 = prev[best_i]
        if last0 is not None and gap0 + 1 > coast:
            last0, v0, gap0 = None, None, -1
        cur = [(s0, last0, v0, gap0 + 1 if last0 is not None else 0)]
        bk = [(best_i, False)]
        for x, y, c, u, v in cands.get(f, []):
            em = logit(c)
            best = (prev[best_i][0] + em, best_i, True, None)        # start of a new segment
            for i, (s, last, vel, gap) in enumerate(prev):
                if last is None:
                    continue
                k = gap + 1
                du, dv = u - last[0], v - last[1]
                if math.hypot(du, dv) > max_step * k + args.noise_m:  # unreachable
                    continue
                if vel is None:
                    pen = 0.0
                else:
                    e = math.hypot(du - vel[0] * k, dv - vel[1] * k) / k
                    pen = min(e * e / two_s2, args.kick_cap)
                sc = s + em + args.cont_bonus - pen
                if sc > best[0]:
                    best = (sc, i, False, (last, vel, k))
            sc, i, new, ctx = best
            if new:
                nvel = None
            else:
                last, vel, k = ctx
                step = ((u - last[0]) / k, (v - last[1]) / k)
                nvel = step if vel is None else (0.5 * step[0] + 0.5 * vel[0], 0.5 * step[1] + 0.5 * vel[1])
            cur.append((sc, (u, v), nvel, 0))
            bk.append((i, new))
        back.append(bk)
        prev = cur
    i = max(range(len(prev)), key=lambda j: prev[j][0])
    path = {}
    for t in range(len(frames) - 1, -1, -1):
        pi, new = back[t][i]
        if i > 0:
            path[frames[t]] = (i - 1, new)
        i = pi
    return path


def segments_of(path):
    """frame -> segment id. A segment restarts when Viterbi chooses 'new segment'."""
    seg, sid = {}, -1
    for f in sorted(path):
        if path[f][1] or sid < 0:
            sid += 1
        seg[f] = sid
    return seg


def track_model(model, hom, args):
    cands, frames = load_candidates(model, hom)
    path = select_path(cands, frames, args)
    seg = segments_of(path)
    sizes = Counter(seg.values())
    kept = {f: cands[f][path[f][0]] for f in path if sizes[seg[f]] >= args.min_dets}
    stats = {"frames": len(frames), "frames_with_candidates": len(cands), "chosen_detections": len(path),
             "segments": len(sizes), "short_segments_discarded": sum(1 for n in sizes.values() if n < args.min_dets)}

    unused = {f: [("cand", x, y, c) for (x, y, c, _, _) in cs
                  if f not in kept or kept[f][:2] != (x, y)] for f, cs in cands.items()}
    filled = fill_gaps({f: d[:2] for f, d in kept.items()}, hom, args, unused)

    out = {}
    for f, (x, y, c, _, _) in kept.items():
        out[f] = (x, y, "detected", c, seg[f])
    known = sorted(kept)
    for f, (x, y, status, src) in filled.items():
        k = np.searchsorted(known, f)
        sa, sb = seg[known[k - 1]], seg[known[k]]
        conf = next((c for (_, cx, cy, c) in unused.get(f, []) if (cx, cy) == (x, y)), "")
        out[f] = (x, y, status, conf, sa if sa == sb else -1)
    return out, frames, stats


def write_track(path, out, hom, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["frame", "x", "y", "pitch_x_m", "pitch_y_m", "status", "conf", "segment"])
        for f in frames:
            if f in out:
                x, y, s, c, sg = out[f]
                px, py = hom.to_pitch(f, x, y)
                w.writerow([f, f"{x:.1f}", f"{y:.1f}", f"{px:.2f}", f"{py:.2f}", s,
                            f"{c:.3f}" if c != "" else "", sg])
            else:
                w.writerow([f, "", "", "", "", "none", "", ""])


def main():
    p = argparse.ArgumentParser(description="Post-processing of the detections of a model")
    p.add_argument("--model", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    add_common_params(p)
    add_track_params(p)
    args = p.parse_args()
    set_video(args.video)
    hom = Homographies(args.homography)
    for m in args.model:
        out, frames, stats = track_model(m, hom, args)
        write_track(track_path(m), out, hom, frames)
        n = len(frames)
        cnt = Counter(v[2] for v in out.values())
        print(f"\n[{m}] frames {frames.start}-{frames.stop - 1}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        for s in ["detected", "recovered", "interpolated"]:
            print(f"  {s}: {cnt[s]} ({100 * cnt[s] / n:.1f}%)")
        print(f"  ball present: {len(out)} ({100 * len(out) / n:.1f}%)")
        print(f"[INFO] Saved {track_path(m)}")


if __name__ == "__main__":
    main()
