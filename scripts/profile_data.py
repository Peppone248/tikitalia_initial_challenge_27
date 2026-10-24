"""Aggregate profile of the BAR-ATH data, to ground design decisions.

    python scripts/profile_data.py              # prints a report (~1-2 min, reads the raw ATD)
    python scripts/profile_data.py --out cache/profile.txt

Prints ONLY aggregate numbers (rates, counts, percentiles) over the graded frames -- no
frame-level rows, coordinates or extracts of the data. Read the output before sharing it.
It does not read any ground truth.
"""
import argparse
import csv
import io
import math
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter")]

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]


def pct(a, q):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def runs(mask):
    """(start, length) of each run of True in a boolean array."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


def parse_xy(group):
    """First two numbers of an 'x,y,...' group, NaN if missing."""
    if not group or group[0] == "N":
        return math.nan, math.nan
    a = group.split(",")
    try:
        return float(a[0]), float(a[1])
    except (ValueError, IndexError):
        return math.nan, math.nan


def read_atd(raw_path, n_frames, n_slots):
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < n_frames:
                continue
            ball[f] = parse_xy(parts[2])
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                players[f, slot] = parse_xy(group)
    return ball, players


def section(out, title):
    out.write("\n== {} ==\n".format(title))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="also write the report to this file")
    args = ap.parse_args()

    meta_path, raw_path, homog_path = require(["atd_meta", "atd_raw", "homography"])
    n = source_frames("tactical")
    graded = np.zeros(n, dtype=bool)
    half_of = np.zeros(n, dtype=np.int8)
    for half, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
        half_of[first:last + 1] = half
    G = int(graded.sum())

    team = team_of_each_track(meta_path)          # 0 home, 1 away, per slot
    print("reading ATD (1-2 min) ...", file=sys.stderr)
    ball, players = read_atd(raw_path, n, len(team))

    out = io.StringIO()
    out.write("BAR-ATH aggregate data profile (graded frames only: {} frames, {:.1f} min)\n"
              .format(G, G / FPS / 60))

    # ---------------------------------------------------------------- ball visibility
    bx = ball[:, 0] * LENGTH
    by = ball[:, 1] * WIDTH
    present = ~np.isnan(ball[:, 0])
    section(out, "Ball visibility")
    out.write("ball present: {:.1f}% of graded frames\n".format(100 * present[graded].mean()))
    for half in (1, 2):
        m = half_of == half
        out.write("  half {}: {:.1f}%\n".format(half, 100 * present[m].mean()))
    starts, lengths = runs(~present & graded)
    sec = lengths / FPS
    out.write("gaps (ball missing): n={}  median {:.2f}s  p75 {:.2f}s  p90 {:.2f}s  p99 {:.1f}s  max {:.1f}s\n"
              .format(len(sec), pct(sec, 50), pct(sec, 75), pct(sec, 90), pct(sec, 99), sec.max() if len(sec) else 0))
    for lo, hi in ((0, 0.2), (0.2, 1), (1, 3), (3, 10), (10, 1e9)):
        sel = (sec >= lo) & (sec < hi)
        out.write("  gaps {:>4}-{:<5}s: {:5d} gaps, {:5.1f}% of missing frames\n".format(
            lo, "inf" if hi > 1e8 else hi, int(sel.sum()), 100 * lengths[sel].sum() / max(lengths.sum(), 1)))

    # ---------------------------------------------------------------- ball dynamics
    section(out, "Ball position and speed (where present)")
    pg = present & graded
    out_of_bounds = pg & ((ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1))
    out.write("ball outside the 0-1 pitch box: {:.2f}% of present graded frames\n".format(
        100 * out_of_bounds.sum() / max(pg.sum(), 1)))
    near_line = pg & ((ball[:, 0] < 0.01) | (ball[:, 0] > 0.99) | (ball[:, 1] < 0.015) | (ball[:, 1] > 0.985))
    out.write("ball within ~1 m of a line: {:.2f}%\n".format(100 * near_line.sum() / max(pg.sum(), 1)))
    v = np.full(n, np.nan)
    both = present[1:] & present[:-1]
    v[1:][both] = np.hypot(np.diff(bx)[both], np.diff(by)[both]) * FPS
    vg = v[graded]
    out.write("ball speed (frame-to-frame) m/s: median {:.1f}  p75 {:.1f}  p90 {:.1f}  p99 {:.1f}\n"
              .format(pct(vg, 50), pct(vg, 75), pct(vg, 90), pct(vg, 99)))
    out.write("  speed < 0.5 m/s: {:.1f}%   > 15 m/s: {:.1f}%   > 40 m/s (likely jumps): {:.2f}%\n".format(
        100 * np.nanmean(vg < 0.5), 100 * np.nanmean(vg > 15), 100 * np.nanmean(vg > 40)))
    still_runs = runs(graded & (v < 0.5))[1] / FPS
    out.write("stationary-ball runs >= 2 s: {}   >= 5 s: {}\n".format(
        int((still_runs >= 2).sum()), int((still_runs >= 5).sum())))
    thirds = np.digitize(ball[pg, 0], [1 / 3, 2 / 3])
    out.write("ball x-thirds (present frames): {}\n".format(
        "  ".join("{:.1f}%".format(100 * np.mean(thirds == k)) for k in range(3))))

    # ---------------------------------------------------------------- players
    section(out, "Tracks")
    px = players[:, :, 0] * LENGTH
    py = players[:, :, 1] * WIDTH
    here = ~np.isnan(players[:, :, 0])
    n_home = here[:, team == 0].sum(1)
    n_away = here[:, team == 1].sum(1)
    out.write("slots: {} (home {}, away {})\n".format(len(team), int((team == 0).sum()), int((team == 1).sum())))
    out.write("tracked per frame: home mean {:.1f} (p5 {:.0f}, p95 {:.0f})  away mean {:.1f} (p5 {:.0f}, p95 {:.0f})\n"
              .format(n_home[graded].mean(), pct(n_home[graded], 5), pct(n_home[graded], 95),
                      n_away[graded].mean(), pct(n_away[graded], 5), pct(n_away[graded], 95)))
    out.write("frames with > 11 of one team: {:.2f}%   with < 9 of one team: {:.2f}%\n".format(
        100 * np.mean(((n_home > 11) | (n_away > 11))[graded]),
        100 * np.mean(((n_home < 9) | (n_away < 9))[graded])))
    seg_lengths = []
    for s in range(len(team)):
        seg_lengths.extend(runs(here[:, s] & graded)[1])
    seg = np.array(seg_lengths) / FPS
    out.write("tracklets: n={}  median {:.1f}s  p25 {:.1f}s  p75 {:.1f}s  (< 5 s: {:.1f}%)\n".format(
        len(seg), pct(seg, 50), pct(seg, 25), pct(seg, 75), 100 * np.mean(seg < 5)))

    # ---------------------------------------------------------------- nearest player
    section(out, "Nearest player to the ball (where ball present)")
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d_all = np.where(np.isnan(d), np.inf, d)
    nearest = d_all.argmin(1)
    dmin = d_all.min(1)
    dmin[~present] = np.nan
    dmin[np.isinf(dmin)] = np.nan
    dg = dmin[pg]
    out.write("distance m: median {:.1f}  p75 {:.1f}  p90 {:.1f}\n".format(pct(dg, 50), pct(dg, 75), pct(dg, 90)))
    out.write("  < 1 m: {:.1f}%   < 2 m: {:.1f}%   > 5 m: {:.1f}%   > 10 m: {:.1f}%\n".format(
        100 * np.nanmean(dg < 1), 100 * np.nanmean(dg < 2), 100 * np.nanmean(dg > 5), 100 * np.nanmean(dg > 10)))
    for t, name in ((0, "home"), (1, "away")):
        dt = np.where(np.isnan(d[:, team == t]), np.inf, d[:, team == t]).min(1)
        gap = dt[pg]
        out.write("  nearest {} player distance median {:.1f} m\n".format(name, pct(gap[np.isfinite(gap)], 50)))
    near_team = np.where(pg, team[nearest], -1)
    crowd2 = (d < 2).sum(1)
    crowd5 = (d < 5).sum(1)
    out.write("players within 2 m: mean {:.2f}; within 5 m: mean {:.2f} (frames with >= 3 within 5 m: {:.1f}%)\n".format(
        crowd2[pg].mean(), crowd5[pg].mean(), 100 * np.mean(crowd5[pg] >= 3)))
    seq = near_team[near_team >= 0]
    switches = int(np.sum(seq[1:] != seq[:-1]))
    out.write("nearest-team switches (raw, ball-present frames): {}  = {:.1f} per minute of ball-present play\n".format(
        switches, switches / (len(seq) / FPS / 60)))
    out.write("nearest team share: home {:.1f}%  away {:.1f}%\n".format(
        100 * np.mean(seq == 0), 100 * np.mean(seq == 1)))

    # where the ball gets lost
    section(out, "Context when the ball is lost (last visible frame before a gap >= 1 s)")
    long_gaps = starts[lengths >= FPS]
    before = long_gaps - 1
    before = before[(before >= 0) & present[np.clip(before, 0, n - 1)]]
    if len(before):
        th = np.digitize(ball[before, 0], [1 / 3, 2 / 3])
        out.write("gaps >= 1 s: {}  ball x-third at loss: {}\n".format(
            len(before), "  ".join("{:.0f}%".format(100 * np.mean(th == k)) for k in range(3))))
        out.write("players within 5 m at loss: mean {:.2f} (vs {:.2f} overall)\n".format(
            crowd5[before].mean(), crowd5[pg].mean()))
        out.write("ball speed at loss: median {:.1f} m/s (vs {:.1f} overall)\n".format(
            pct(v[before], 50), pct(vg, 50)))
        box = ((ball[before, 0] < 16.5 / LENGTH) | (ball[before, 0] > 1 - 16.5 / LENGTH)) & \
              (np.abs(ball[before, 1] - 0.5) < 20.16 / WIDTH)
        out.write("lost inside a penalty box: {:.0f}%\n".format(100 * box.mean()))
        same = 0
        ends = (long_gaps + lengths[lengths >= FPS])
        pairs = 0
        for s0, e0 in zip(long_gaps, ends):
            if s0 - 1 >= 0 and e0 < n and present[s0 - 1] and present[e0] and near_team[s0 - 1] >= 0 and near_team[e0] >= 0:
                pairs += 1
                same += near_team[s0 - 1] == near_team[e0]
        if pairs:
            out.write("nearest team same before/after gap: {:.0f}% of {} gaps\n".format(100 * same / pairs, pairs))

    # ---------------------------------------------------------------- team shape
    section(out, "Team shape (graded frames)")
    for t, name in ((0, "home"), (1, "away")):
        sx = np.where(team == t)[0]
        cx = np.nanmean(px[:, sx], 1)
        spread = np.nanstd(px[:, sx], 1) + np.nanstd(py[:, sx], 1)
        out.write("{}: centroid x median {:.1f} m  spread (sd x + sd y) median {:.1f} m  p10 {:.1f}  p90 {:.1f}\n".format(
            name, pct(cx[graded], 50), pct(spread[graded], 50), pct(spread[graded], 10), pct(spread[graded], 90)))

    # ---------------------------------------------------------------- smart tagging
    section(out, "Smart Tagging")
    events = load_tagging()
    out.write("instances: {}\n".format(len(events)))
    codes = Counter(e["code"] for e in events)
    for code, c in codes.most_common():
        durs = [(e["end_frame"] - e["start_frame"]) / FPS for e in events if e["code"] == code]
        out.write("  {:<30} {:4d}  median {:5.1f}s\n".format(code, c, float(np.median(durs))))
    out.write("Team tag: {}\n".format(dict(Counter(e["tags"].get("Team", "(none)") for e in events))))
    out.write("set piece Type: {}\n".format(dict(Counter(
        e["tags"].get("Type", "(none)") for e in events if (e["code"] or "").lower() == "set pieces"))))
    cover_any = np.zeros(n, dtype=bool)
    cover_team = np.zeros((n, 2), dtype=bool)
    setp = np.zeros(n, dtype=bool)
    names = {}
    for e in events:
        a, b = max(e["start_frame"], 0), min(e["end_frame"], n)
        cover_any[a:b] = True
        if (e["code"] or "").lower() == "set pieces":
            setp[a:b] = True
        tm = e["tags"].get("Team")
        if tm and tm != "N/A" and (e["code"] or "").lower() != "set pieces":
            idx = names.setdefault(tm, len(names))
            if idx < 2:
                cover_team[a:b, idx] = True
    out.write("graded frames covered by any tag: {:.1f}%   by a team-tagged phase: {:.1f}%   by both teams at once: {:.1f}%\n".format(
        100 * cover_any[graded].mean(), 100 * cover_team.any(1)[graded].mean(), 100 * cover_team.all(1)[graded].mean()))
    out.write("graded frames inside SET PIECES: {:.1f}%\n".format(100 * setp[graded].mean()))
    out.write("ball present inside SET PIECES: {:.1f}% (vs {:.1f}% overall)\n".format(
        100 * present[setp & graded].mean() if (setp & graded).any() else float("nan"), 100 * present[graded].mean()))
    out.write("ball speed inside SET PIECES: median {:.1f} m/s (vs {:.1f} overall)\n".format(
        pct(v[setp & graded], 50), pct(vg, 50)))
    out.write("ball-missing graded frames covered by a team-tagged phase: {:.1f}%\n".format(
        100 * cover_team.any(1)[graded & ~present].mean()))

    # ---------------------------------------------------------------- homography
    section(out, "Homography")
    with open(homog_path) as handle:
        rows = list(csv.DictReader(handle))
    valid = np.array([r.get("valid") == "1" for r in rows])
    err = np.array([float(r["reprojection_error"]) if r.get("reprojection_error") not in (None, "", "nan", "NaN") else np.nan
                    for r in rows])
    gm = graded[:len(valid)]
    out.write("rows: {}  valid on graded: {:.1f}%  reprojection error median {:.2f}  p95 {:.2f}\n".format(
        len(rows), 100 * valid[gm].mean(), pct(err[gm], 50), pct(err[gm], 95)))

    text = out.getvalue()
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
