"""
Pipeline step 3: final ball position, by consensus between the models' tracks.

Input:  the post-processed tracks (runs/ball_pipeline/<model>/track.csv, from post_process.py)
        + per-frame homography of the video.
Output: runs/ball_pipeline/ensemble_<models>/ball.csv, ONE row per frame:
    frame         index in the given video
    x, y          position in pixels (empty if no ball)
    pitch_x_m, pitch_y_m   position on the pitch in metres (projection on the ground)
    status        confirmed | tracked | recovered | interpolated | none
    source        agreeing models (confirmed), model (tracked / recovered),
                  'spline' or 'linear' (interpolated: SYNTHETIC point)

Algorithm:
  1. Per-frame confirmation: among ALL the candidates inside the pitch (detections.csv, up to 3
     per model) those of at least --min-votes models within --agree-px -> 'confirmed', at the
     mean position. All candidates are used, not only the one chosen by the track, so a
     confirmation does not depend on the post-processing choices. A confirmation with no other
     reachable confirmation within ±--isolation-s seconds is discarded (isolated false positive).
  2. Per-segment confirmation: a continuous segment of a model's track ('segment' column)
     that goes through at least one confirmed frame (within --agree-px) is "trusted": both models saw it,
     so it is the ball. Its detections in the other frames count as 'tracked', even if the
     other model sees nothing there. Segments that are never confirmed are ignored: this is how
     the false positives of a single model are discarded.
     If two trusted segments disagree in the same frame, the one closer to the last known
     position is kept. A 'tracked' point must be reachable from the last known position at a
     speed ≤ --max-speed.
  3. Gaps: fill_gaps (utils/pitch.py), as in the tracks: up to 0.5 s always (spline),
     up to 2 s if the ball is slow (straight line). A real detection of a model that falls on the
     estimated trajectory is recovered ('recovered'); otherwise a synthetic point ('interpolated').

Parameters:
  --min-votes 2      two detectors trained on different data rarely fail at the same spot
  --agree-px 15      two detectors on the same ball differ by a few pixels
  --isolation-s 0.2  a real ball is detected for more than an instant
  plus the common physical ones of utils.pitch.add_common_params (40 m/s, 0.5 s, 2 s, 6 m/s)

Usage (from the project root):
  python -m inference.ball_consensus                                   # ensemble of the default models
  python -m inference.ball_consensus --models tikitalia_tiles roboflow
"""
import argparse
import csv
import math
from collections import defaultdict

import numpy as np

from utils.config import ALL_MODELS, DEFAULT_MODELS, ball_path, set_video
from utils.io import load_track, load_detections
from utils.pitch import Homographies, add_common_params, fill_gaps

STATUSES = ["confirmed", "tracked", "recovered", "interpolated"]


def add_params(p):
    """Ensemble parameters (shared with compare_models.py)."""
    add_common_params(p)
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--agree-px", type=float, default=15)
    p.add_argument("--isolation-s", type=float, default=0.2)


def load_tracks(models):
    """model -> track (per-frame DataFrame) and the common frame range."""
    tracks = {m: load_track(m) for m in models}
    lo = max(t.index.min() for t in tracks.values())
    hi = min(t.index.max() for t in tracks.values())
    return tracks, range(int(lo), int(hi) + 1)


def real_detections(tracks):
    """model -> {frame: (x, y, conf, segment)} for the real detections only ('detected')."""
    det = {}
    for m, t in tracks.items():
        d = t[t["status"] == "detected"]
        det[m] = {int(f): (float(x), float(y), float(c), int(s))
                  for f, x, y, c, s in zip(d.index, d["x"], d["y"], d["conf"], d["segment"])}
    return det


def vote(cands, agree_px, min_votes):
    """cands: [(model, x, y, conf)]. Returns (x, y, models) of the best group, or None."""
    best, best_key = None, None
    for _, x0, y0, _ in cands:
        top = {}
        for m, x, y, c in cands:
            if math.hypot(x - x0, y - y0) <= agree_px and (m not in top or c > top[m][2]):
                top[m] = (x, y, c)
        if len(top) < min_votes:
            continue
        key = (len(top), sum(v[2] for v in top.values()))
        if best_key is None or key > best_key:
            best_key = key
            best = (float(np.mean([v[0] for v in top.values()])),
                    float(np.mean([v[1] for v in top.values()])), [m for m in ALL_MODELS if m in top])
    return best


def consensus(tracks, hom, frames, args):
    """Returns out[frame] = (x, y, status, source) and some statistics."""
    models = list(tracks)
    det = real_detections(tracks)
    raw, _ = load_detections(models)
    stats = defaultdict(int)

    out = {}                                                        # 1. per-frame confirmation
    trusted = {m: set() for m in models}
    for f in frames:
        cands = [d for d in raw.get(f, []) if hom.has(f) and hom.on_pitch(f, d[1], d[2])]
        v = vote(cands, args.agree_px, args.min_votes)
        if v:
            out[f] = (v[0], v[1], "confirmed", "+".join(v[2]))

    max_step = args.max_speed / args.fps                            # isolated confirmations: discarded
    win = max(1, round(args.isolation_s * args.fps))
    pos = {f: hom.to_pitch(f, *out[f][:2]) for f in out}
    for f in list(out):
        if not any(f2 in pos and np.linalg.norm(pos[f2] - pos[f]) <= max_step * abs(f2 - f) + args.noise_m
                   for f2 in range(f - win, f + win + 1) if f2 != f):
            del out[f]
            stats["isolated_confirmed_discarded"] += 1
    for f, (x, y, _, _) in out.items():
        for m in models:
            if f in det[m] and math.hypot(det[m][f][0] - x, det[m][f][1] - y) <= args.agree_px:
                trusted[m].add(det[m][f][3])

                                                                    # 2. per-segment confirmation
    last = None                                                     # (frame, position in metres)
    for f in frames:
        if f in out:
            if hom.has(f):
                last = (f, hom.to_pitch(f, *out[f][:2]))
            continue
        opts = [(m, det[m][f]) for m in models if f in det[m] and det[m][f][3] in trusted[m]]
        if not opts or not hom.has(f):
            continue
        pos = [(m, d, hom.to_pitch(f, d[0], d[1])) for m, d in opts]
        if last is not None:
            reach = max_step * (f - last[0]) + args.noise_m
            dist = [(np.linalg.norm(p - last[1]), m, d, p) for m, d, p in pos]
            ok = [t for t in dist if t[0] <= reach]
            stats["unreachable_tracked_discarded"] += len(dist) - len(ok)
            if not ok:
                continue
            _, m, d, p = min(ok, key=lambda t: t[0])
        elif len(pos) == 1:
            m, d, p = pos[0]
        else:
            stats["conflicting_tracked_discarded"] += 1
            continue
        out[f] = (d[0], d[1], "tracked", m)
        last = (f, p)

    others = defaultdict(list)                                      # 3. gaps
    for m in models:
        for f, (x, y, _, _) in det[m].items():
            if f not in out:
                others[f].append((m, x, y))
    out.update(fill_gaps({f: v[:2] for f, v in out.items()}, hom, args, others))
    stats["trusted_segments"] = sum(len(s) for s in trusted.values())
    return out, stats


def summarize(out, n):
    """Counts per outcome."""
    count = defaultdict(int)
    for v in out.values():
        count[v[2]] += 1
    count["none"] = n - len(out)
    count["present"] = len(out)
    return count


def write_csv(path, out, hom, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["frame", "x", "y", "pitch_x_m", "pitch_y_m", "status", "source"])
        for f in frames:
            if f in out:
                x, y, s, src = out[f]
                px, py = hom.to_pitch(f, x, y)
                w.writerow([f, f"{x:.1f}", f"{y:.1f}", f"{px:.2f}", f"{py:.2f}", s, src])
            else:
                w.writerow([f, "", "", "", "", "none", ""])


def main():
    p = argparse.ArgumentParser(description="Consensus between the models' tracks for the ball position")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    add_params(p)
    args = p.parse_args()
    if len(args.models) < args.min_votes:
        raise SystemExit(f"[ERROR] at least {args.min_votes} models are needed")

    set_video(args.video)
    tracks, frames = load_tracks(args.models)
    print(f"[INFO] models {args.models}, frames {frames.start}-{frames.stop - 1}")
    hom = Homographies(args.homography)
    out, stats = consensus(tracks, hom, frames, args)
    path = ball_path(args.models)
    write_csv(path, out, hom, frames)

    n, c = len(frames), summarize(out, len(frames))
    print("\n| outcome | frames | % |\n|---|---|---|")
    for s in STATUSES + ["none"]:
        print(f"| {s} | {c[s]} | {100 * c[s] / n:.1f}% |")
    print(f"| **ball present** | {c['present']} | **{100 * c['present'] / n:.1f}%** |")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"\n[INFO] Saved {path}")


if __name__ == "__main__":
    main()
