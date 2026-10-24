"""Edge-case and reliability checks on the full tactical ATD (all 199,319 frames).

    python scripts/check_atd.py --out cache/check_atd.txt      # ~2-3 min

Answers, with aggregate numbers only (no rows, no coordinates):
  A. speed: unit of the provided speed channel, and how a speed derived from positions compares
  B. goalkeepers: which slots are the keepers, and each team's attacking direction per half
  C. long ball gaps: dead ball or lost ball? (where the ball vanishes/reappears, how fast the
     players move during the gap, overlap with Smart Tagging set pieces)
  D. collective player speed: live play vs ball gaps vs halftime
  E. tracking edge cases: teleports, duplicates, objects off the pitch, ball with nobody near
  F. Smart Tagging read with code polarity (attacking vs defending codes)
No ground truth is read.
"""
import argparse
import io
import math
import os
import sys
import warnings
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter")]

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)

FPS = GAMES[GAME]["fps"]
L, W = GAMES[GAME]["pitch"]
DT = 1.0 / FPS

# Smart Tagging codes: does the Team tag name the side WITH the ball or WITHOUT it?
ATTACKING = {"build up", "progression", "ball in final third", "ball in the box", "creating chances",
             "attacking transition", "long ball", "players in the box"}
DEFENDING = {"defending in defensive third", "defending in middle third", "defending in attacking third",
             "defensive transition"}


# --------------------------------------------------------------------------------------------- io
def num(s):
    if not s or s[0] == "N":
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def read_atd(raw_path, n, n_slots):
    """ball (n,3) and players (n,slots,3): x, y (0-1), provided speed. NaN where missing."""
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    pl = np.full((n, n_slots, 3), np.nan, dtype=np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < n:
                continue
            b = parts[2].split(",")
            ball[f, :len(b[:3])] = [num(v) for v in b[:3]]
            for s, g in enumerate(parts[1].split(";")[:n_slots]):
                if g and g[0] != "N":
                    v = g.split(",")
                    pl[f, s, :len(v[:3])] = [num(x) for x in v[:3]]
    return ball, pl


# ------------------------------------------------------------------------------------- helpers
def pct(a, q):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def summ(a, qs=(10, 50, 90, 99)):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return "n=0"
    return "n={}  ".format(a.size) + "  ".join("p{} {:.2f}".format(q, np.percentile(a, q)) for q in qs)


def runs(mask):
    m = np.concatenate(([False], np.asarray(mask, bool), [False])).astype(np.int8)
    d = np.diff(m)
    s = np.flatnonzero(d == 1)
    return s, np.flatnonzero(d == -1) - s


def speed_from_positions(x, y, k):
    """Centred-difference speed in m/s over +/-k frames; x, y in metres, NaN-aware."""
    v = np.full(x.shape, np.nan)
    if x.shape[0] <= 2 * k:
        return v
    dx = x[2 * k:] - x[:-2 * k]
    dy = y[2 * k:] - y[:-2 * k]
    v[k:-k] = np.hypot(dx, dy) / (2 * k * DT)
    return v


def h(out, t):
    out.write("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78 + "\n")


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    args = ap.parse_args()

    meta, raw = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    team = team_of_each_track(meta)
    S = len(team)
    graded = np.zeros(n, bool)
    half = np.zeros(n, np.int8)
    for hf, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
        half[a:b + 1] = hf
    halftime = np.zeros(n, bool)
    blocks = half_video_blocks(GAME)
    halftime[blocks[0][2] + 1:blocks[1][1]] = True

    print("reading ATD ...", file=sys.stderr)
    ball, pl = read_atd(raw, n, S)
    out = io.StringIO()
    out.write("ATD checks on all {} frames ({} graded, {} halftime)\n".format(n, graded.sum(), halftime.sum()))

    bx, by, bs = ball[:, 0].astype(float) * L, ball[:, 1].astype(float) * W, ball[:, 2].astype(float)
    bpres = np.isfinite(bx)
    px, py, ps = pl[:, :, 0].astype(float) * L, pl[:, :, 1].astype(float) * W, pl[:, :, 2].astype(float)
    here = np.isfinite(px)

    # ===================================================================== A. speed
    h(out, "A. SPEED - unit of the provided channel, and speed derived from positions")
    v1 = speed_from_positions(bx, by, 1)      # m/s, +/-1 frame (0.08 s)
    v2 = speed_from_positions(bx, by, 2)      # +/-2 frames (0.16 s)
    v5 = speed_from_positions(bx, by, 5)      # +/-5 frames (0.40 s)
    ok = bpres & np.isfinite(v2) & np.isfinite(bs)
    out.write("ball, frames with provided speed AND derivable speed: {}\n".format(int(ok.sum())))
    out.write("  provided speed          : {}\n".format(summ(bs[ok])))
    out.write("  derived speed m/s (k=2) : {}\n".format(summ(v2[ok])))
    moving = ok & (v2 > 2)
    if moving.any():
        ratio = bs[moving] / v2[moving]
        out.write("  ratio provided/derived (moving ball): median {:.3f}  p25 {:.3f}  p75 {:.3f}"
                  "   [3.6 => provided is km/h; 1.0 => m/s]\n".format(pct(ratio, 50), pct(ratio, 25), pct(ratio, 75)))
        out.write("  correlation provided vs derived: {:.3f}\n".format(np.corrcoef(bs[moving], v2[moving])[0, 1]))
    out.write("  provided speed exactly 0 on present frames: {:.1f}%\n".format(100 * np.mean(bs[bpres] == 0)))
    # players
    pv = np.full(px.shape, np.nan)
    for s in range(S):
        pv[:, s] = speed_from_positions(px[:, s], py[:, s], 2)
    okp = here & np.isfinite(pv) & np.isfinite(ps) & (pv > 1)
    if okp.any():
        r = ps[okp] / pv[okp]
        out.write("players: ratio provided/derived (moving): median {:.3f}  p25 {:.3f}  p75 {:.3f}  corr {:.3f}\n".format(
            pct(r, 50), pct(r, 25), pct(r, 75), np.corrcoef(ps[okp], pv[okp])[0, 1]))
        out.write("  derived player speed m/s on graded frames: {}\n".format(summ(pv[graded][here[graded]])))
    out.write("\nnoise of the derived ball speed (graded, present):\n")
    for name, v in (("k=1 (0.08 s)", v1), ("k=2 (0.16 s)", v2), ("k=5 (0.40 s)", v5)):
        g = v[graded & bpres]
        out.write("  {:<14} {}   >40 m/s: {:.2f}%\n".format(name, summ(g), 100 * np.nanmean(g > 40)))
    step = np.full(n, np.nan)
    both = bpres[1:] & bpres[:-1]
    step[1:][both] = np.hypot(np.diff(bx)[both], np.diff(by)[both])
    out.write("frame-to-frame ball step m: {}\n".format(summ(step[graded], (50, 90, 99, 99.9))))
    out.write("  teleports (step > 2 m in one frame = >50 m/s): {} frames\n".format(int(np.nansum(step[graded] > 2))))
    acc = np.abs(np.diff(v1)) / DT
    out.write("  |acceleration| from k=1 speed, m/s^2: {}\n".format(summ(acc[graded[1:]], (50, 90, 99))))
    # speed right after the ball reappears
    gs, gl = runs(~bpres & graded)
    reapp = gs + gl
    reapp = reapp[(reapp < n - 6)]
    out.write("derived speed in the first 0.4 s after the ball reappears (k=2): {}\n".format(
        summ(np.nanmedian(np.stack([v2[np.clip(reapp + j, 0, n - 1)] for j in range(2, 10)]), 0))))

    # ===================================================================== B. goalkeepers
    h(out, "B. GOALKEEPERS AND ATTACKING DIRECTION")
    out.write("{:>5} {:>4} {:>8} | {:>7} {:>6} {:>6} | {:>7} {:>6} {:>6}\n".format(
        "slot", "team", "present", "h1 x", "h1 sdx", "h1 pr", "h2 x", "h2 sdx", "h2 pr"))
    cand = []
    for s in range(S):
        pr = here[graded, s].mean()
        row = []
        for hf in (1, 2):
            m = (half == hf)
            row += [np.nanmean(pl[m, s, 0]), np.nanstd(pl[m, s, 0]), here[m, s].mean()]
        out.write("{:>5} {:>4} {:>7.1f}% | {:>7.3f} {:>6.3f} {:>5.0f}% | {:>7.3f} {:>6.3f} {:>5.0f}%\n".format(
            s + 1, "H" if team[s] == 0 else "A", 100 * pr, row[0], row[1], 100 * row[2], row[3], row[4], 100 * row[5]))
        if pr > 0.6 and row[1] < 0.12 and abs(row[0] - 0.5) > 0.3:
            cand.append((s, row[0], row[3]))
    out.write("\ngoalkeeper candidates (present >60% of graded, x sd <0.12, mean x near a goal line):\n")
    for s, x1, x2 in cand:
        out.write("  slot {} ({}): mean x half1 {:.3f} -> half2 {:.3f}   => defends {} goal in half 1, {} in half 2\n".format(
            s + 1, "H" if team[s] == 0 else "A", x1, x2, "x=0" if x1 < 0.5 else "x=1", "x=0" if x2 < 0.5 else "x=1"))
    # team centroids per half: a direction check that does not depend on the keepers
    for hf in (1, 2):
        m = (half == hf)
        ch = np.nanmean(np.nanmean(pl[m][:, team == 0, 0], 1))
        ca = np.nanmean(np.nanmean(pl[m][:, team == 1, 0], 1))
        out.write("half {}: mean team centroid x  home {:.3f}  away {:.3f}\n".format(hf, ch, ca))

    # ===================================================================== C. gaps
    h(out, "C. BALL GAPS (graded) - dead ball or lost ball?")
    # collective player speed per frame: mean of derived speeds of tracked players
    team_speed = np.nanmean(np.where(here, pv, np.nan), 1)
    out.write("gaps: {}   durations s: {}\n".format(len(gs), summ(gl / FPS, (25, 50, 75, 90))))

    def zone(x01, y01):
        """Coarse location of a ball position (normalised)."""
        if not (np.isfinite(x01) and np.isfinite(y01)):
            return "unknown"
        if x01 < 0 or x01 > 1 or y01 < 0 or y01 > 1:
            return "off pitch"
        if abs(x01 - 0.5) < 0.03 and abs(y01 - 0.5) < 0.05:
            return "centre spot"
        near_goal_line = x01 < 0.03 or x01 > 0.97
        near_side = y01 < 0.03 or y01 > 0.97
        if near_goal_line and near_side:
            return "corner"
        six = (x01 < 5.5 / L or x01 > 1 - 5.5 / L) and abs(y01 - 0.5) < 9.16 / W
        if six:
            return "six-yard box"
        if near_side:
            return "touchline"
        if near_goal_line:
            return "goal line"
        box = (x01 < 16.5 / L or x01 > 1 - 16.5 / L) and abs(y01 - 0.5) < 20.16 / W
        return "penalty box" if box else "open play"

    events = load_tagging()
    sp_start = np.zeros(n, bool)
    for e in events:
        if (e["code"] or "").lower() == "set pieces" and 0 <= e["start_frame"] < n:
            sp_start[e["start_frame"]] = True
    sp_cum = np.concatenate(([0], np.cumsum(sp_start)))

    live_speed = pct(team_speed[graded & bpres], 50)
    ht_speed = pct(team_speed[halftime], 50)
    rows = []
    for s0, ln in zip(gs, gl):
        e0 = s0 + ln
        lost = s0 - 1
        x_lost = ball[lost, 0] if lost >= 0 and bpres[lost] else np.nan
        y_lost = ball[lost, 1] if lost >= 0 and bpres[lost] else np.nan
        x_back = ball[e0, 0] if e0 < n and bpres[e0] else np.nan
        y_back = ball[e0, 1] if e0 < n and bpres[e0] else np.nan
        tsp = np.nanmedian(team_speed[s0:e0]) if ln else np.nan
        # a set piece tag starting inside the gap or within 3 s after it
        sp = (sp_cum[min(e0 + 3 * FPS, n)] - sp_cum[s0]) > 0
        rows.append((ln / FPS, zone(x_lost, y_lost), zone(x_back, y_back), tsp, sp,
                     v2[max(lost - 2, 0)] if lost >= 2 else np.nan))
    out.write("median collective player speed m/s: live play (ball visible) {:.2f}   halftime {:.2f}\n".format(
        live_speed, ht_speed))
    for lo, hi, name in ((0, 1, "< 1 s"), (1, 10, "1-10 s"), (10, 30, "10-30 s"), (30, 1e9, ">= 30 s")):
        sel = [r for r in rows if lo <= r[0] < hi]
        if not sel:
            continue
        tsp = np.array([r[3] for r in sel])
        out.write("\n-- gaps {} : {} gaps --\n".format(name, len(sel)))
        out.write("  collective player speed during gap: median {:.2f} m/s  (share of gaps slower than live median: {:.0f}%)\n".format(
            pct(tsp, 50), 100 * np.nanmean(tsp < live_speed)))
        out.write("  ball speed just before loss: median {:.1f} m/s\n".format(pct([r[5] for r in sel], 50)))
        out.write("  Smart Tagging set piece starting in gap or <=3 s after: {:.0f}%\n".format(
            100 * np.mean([r[4] for r in sel])))
        for title, col in (("zone where lost", 1), ("zone where it reappears", 2)):
            c = Counter(r[col] for r in sel)
            out.write("  {:<24}: {}\n".format(title, ", ".join("{} {:.0f}%".format(k, 100 * v / len(sel))
                                                           for k, v in c.most_common())))
    restart_zones = {"touchline", "corner", "six-yard box", "centre spot", "goal line", "off pitch"}
    long_rows = [r for r in rows if r[0] >= 3]
    if long_rows:
        tsp = np.array([r[3] for r in long_rows])
        slow = tsp < 0.75 * live_speed
        rz = np.array([r[2] in restart_zones for r in long_rows])
        sp = np.array([r[4] for r in long_rows])
        out.write("\ngaps >= 3 s ({}): evidence of a stoppage\n".format(len(long_rows)))
        out.write("  players slow (<75% of live median): {:.0f}%   reappears at a restart spot: {:.0f}%   set piece tag: {:.0f}%\n".format(
            100 * slow.mean(), 100 * rz.mean(), 100 * sp.mean()))
        k = slow.astype(int) + rz.astype(int) + sp.astype(int)
        out.write("  signals agreeing (0/1/2/3): {}\n".format(
            "  ".join("{}: {:.0f}%".format(i, 100 * np.mean(k == i)) for i in range(4))))
        dur = np.array([r[0] for r in long_rows])
        out.write("  total time in gaps >= 3 s: {:.1f} min; in gaps with >= 2 signals: {:.1f} min; with 0 signals: {:.1f} min\n".format(
            dur.sum() / 60, dur[k >= 2].sum() / 60, dur[k == 0].sum() / 60))

    # ===================================================================== D. collective speed
    h(out, "D. COLLECTIVE PLAYER SPEED (mean derived speed of tracked players, m/s)")
    long_gap = np.zeros(n, bool)
    for s0, ln in zip(gs, gl):
        if ln >= 3 * FPS:
            long_gap[s0:s0 + ln] = True
    for name, m in (("graded, ball visible", graded & bpres), ("graded, ball gap >= 3 s", long_gap),
                    ("halftime", halftime)):
        out.write("  {:<26} {}\n".format(name, summ(team_speed[m], (10, 25, 50, 75, 90))))
    # smoothed over 2 s, the version a model would use
    k = 2 * FPS
    ts = np.where(np.isfinite(team_speed), team_speed, 0.0)
    cnt = np.isfinite(team_speed).astype(float)
    cs, cc = np.concatenate(([0], np.cumsum(ts))), np.concatenate(([0], np.cumsum(cnt)))
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - k), np.minimum(n, idx + k + 1)
    sm = (cs[hi] - cs[lo]) / np.maximum(cc[hi] - cc[lo], 1)
    out.write("smoothed over +/-2 s:\n")
    for name, m in (("graded, ball visible", graded & bpres), ("graded, ball gap >= 3 s", long_gap),
                    ("halftime", halftime)):
        out.write("  {:<26} {}\n".format(name, summ(sm[m], (10, 25, 50, 75, 90))))

    # ===================================================================== E. edge cases
    h(out, "E. TRACKING EDGE CASES (graded frames)")
    g = graded
    out.write("ball present during halftime: {:.1f}% of halftime frames\n".format(100 * bpres[halftime].mean()))
    off_b = bpres & g & ((ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1))
    out.write("ball off the pitch box: {} frames in {} runs (median run {:.1f} s)\n".format(
        int(off_b.sum()), len(runs(off_b)[0]), pct(runs(off_b)[1] / FPS, 50)))
    out.write("  how far out (m beyond the line): {}\n".format(summ(np.maximum.reduce([
        -bx[off_b], bx[off_b] - L, -by[off_b], by[off_b] - W]), (50, 90, 99))))
    off_p = here & g[:, None] & ((pl[:, :, 0] < -0.02) | (pl[:, :, 0] > 1.02) | (pl[:, :, 1] < -0.03) | (pl[:, :, 1] > 1.03))
    out.write("player positions > ~2 m off the pitch: {:.2f}% of tracked player-frames\n".format(
        100 * off_p.sum() / max(here[g].sum(), 1)))
    # duplicates: two same-team tracks closer than 0.5 m
    dup = np.zeros(n, bool)
    for t in (0, 1):
        idx_t = np.where(team == t)[0]
        X, Y = px[:, idx_t], py[:, idx_t]
        for i in range(len(idx_t)):
            d = np.hypot(X[:, i:i + 1] - X[:, i + 1:], Y[:, i:i + 1] - Y[:, i + 1:])
            dup |= np.nansum(d < 0.5, 1) > 0
    out.write("frames with two same-team tracks < 0.5 m apart (likely duplicates): {:.2f}%\n".format(100 * dup[g].mean()))
    nh, na = here[:, team == 0].sum(1), here[:, team == 1].sum(1)
    out.write("tracked count per team on graded frames: home {}   away {}\n".format(
        {int(k): int(v) for k, v in sorted(Counter(nh[g].tolist()).items())},
        {int(k): int(v) for k, v in sorted(Counter(na[g].tolist()).items())}))
    d_all = np.hypot(px - bx[:, None], py - by[:, None])
    dmin = np.nanmin(np.where(np.isfinite(d_all), d_all, np.inf), 1)
    lonely = bpres & g & (dmin > 15)
    out.write("ball visible with nobody within 15 m: {:.2f}% of visible graded frames ({} runs)\n".format(
        100 * lonely.sum() / max((bpres & g).sum(), 1), len(runs(lonely)[0])))
    ts_slot = np.zeros(S)
    for s in range(S):
        st = np.full(n, np.nan)
        bo = here[1:, s] & here[:-1, s]
        st[1:][bo] = np.hypot(np.diff(px[:, s])[bo], np.diff(py[:, s])[bo])
        ts_slot[s] = np.nansum(st[g] > 1.0)
    out.write("player teleports (> 1 m in one frame = > 25 m/s): {} in total\n".format(int(ts_slot.sum())))

    # ===================================================================== F. smart tagging
    h(out, "F. SMART TAGGING WITH CODE POLARITY")
    names = {}
    poss = np.zeros((n, 2), bool)
    unknown_codes = Counter()
    for e in events:
        code = (e["code"] or "").strip().lower()
        tm = e["tags"].get("Team")
        if code == "set pieces" or not tm or tm == "N/A":
            continue
        ti = names.setdefault(tm, len(names))
        if ti > 1:
            continue
        a, b = max(e["start_frame"], 0), min(e["end_frame"], n)
        if code in ATTACKING:
            poss[a:b, ti] = True
        elif code in DEFENDING:
            poss[a:b, 1 - ti] = True
        else:
            unknown_codes[code] += 1
    out.write("team order in tags: {}   (compare with ATD home/away above)\n".format(names))
    if unknown_codes:
        out.write("codes not mapped: {}\n".format(dict(unknown_codes)))
    one = poss.any(1) & ~poss.all(1)
    out.write("graded frames with an implied possessing team: {:.1f}%   conflicting (both): {:.1f}%   none: {:.1f}%\n".format(
        100 * one[g].mean(), 100 * poss.all(1)[g].mean(), 100 * (~poss.any(1))[g].mean()))
    for name, m in (("ball visible", g & bpres), ("ball gap >= 3 s", long_gap)):
        out.write("  {:<16} implied team: {:.1f}%   conflicting: {:.1f}%\n".format(
            name, 100 * one[m].mean(), 100 * poss.all(1)[m].mean()))
    # agreement with the nearest player where the ball is close to someone (< 1 m)
    nearest = np.nanargmin(np.where(np.isfinite(d_all), d_all, np.inf), 1)
    close = g & bpres & (dmin < 1) & one
    if close.any():
        tag_team = np.where(poss[:, 0], 0, 1)
        names_inv = {v: k for k, v in names.items()}
        # map tag team index to ATD home/away by club name of the ATD metadata order
        import xml.etree.ElementTree as ET
        atd_names = [(t.findtext("Name") or "").strip() for t in ET.parse(meta).getroot().iter() if t.tag.endswith("Team")]
        to_atd = np.array([0 if names_inv.get(i, "").lower()[:4] == atd_names[0].lower()[:4] else 1 for i in (0, 1)])
        agree = to_atd[tag_team[close]] == team[nearest[close]]
        out.write("agreement tag-implied team vs nearest player (< 1 m from ball): {:.1f}% of {} frames\n".format(
            100 * agree.mean(), int(close.sum())))

    text = out.getvalue()
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
