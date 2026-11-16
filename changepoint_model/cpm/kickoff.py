"""Law 8: play starts with the kick-off. Every frame of a period before its kick-off is D.

Why this exists: in the provided match the second half is graded from video frame 110,808 but
kicks off at 113,742 (FAQ.md, "One trap worth knowing"): 2,934 graded frames that are all D, with
players warming up with balls on the pitch. A tracking model sees players near a ball and labels
possessions. Nothing here types that frame: the kick-off is found by code, the same way on every
match, from

1. the Smart Tagging "Kick Off (Start)" interval of the period (a ~5 s window around the restart,
   DATA.md), refined to the kick inside it: the first frame the ball, seen at rest (less than REST_M
   over REST_S) within KICK_SPOT_M of the centre spot, moves more than KICK_MOVE_M within
   KICK_WINDOW_S. No kick found in the
   window: the start of the window (the conservative end: fewer frames made D).
Without Smart Tagging nothing is changed. A tracking-only detector (both teams in their own half,
ball on the centre spot, kick) was tried and dropped: on the 18 external matches it put the kick-off
2-9 minutes late in 3 of 36 periods, because that formation also occurs in open play. On the
external matches the periods start at the kick-off, so the rule is a no-op there by construction.
Check on the provided match without any label: the kick-off frame printed by predict_atd.py against
the one published in FAQ.md (113,742 for the second half).

A kick-off found less than MIN_GAP_S after the start of the period changes nothing (the first half
of the provided match is graded from its kick-off; the external matches too). Constants are a
priori (pitch geometry and seconds), none measured on the provided match.
"""

import numpy as np

from .match import D, X

KICK_SPOT_M = 3.0
KICK_MOVE_M = 2.0
KICK_WINDOW_S = 1.0
REST_S = 0.2
REST_M = 0.5
TAG_PAD_S = 2.0
SEARCH_MIN = 10.0
MIN_GAP_S = 2.0


def _kick_frames(m, lo, hi):
    """Rows t in [lo, hi) where the ball is at rest on the centre spot (seen within KICK_SPOT_M of it,
    moving less than REST_M over the REST_S before t) and moves away by KICK_MOVE_M within
    KICK_WINDOW_S (Law 8: the ball is stationary on the centre mark, then kicked)."""
    fps = int(round(m.fps))
    bx, by = m.ball[:, 0], m.ball[:, 1]
    w = int(round(KICK_WINDOW_S * fps))
    r = max(1, int(round(REST_S * fps)))
    out = []
    for t in range(max(lo, r), min(hi, m.T - 1)):
        if np.isnan(bx[t]) or np.hypot(bx[t], by[t]) > KICK_SPOT_M:
            continue
        prev = slice(t - r, t + 1)
        if np.isnan(bx[prev]).any() or np.max(np.hypot(bx[prev] - bx[t], by[prev] - by[t])) > REST_M:
            continue
        seg = slice(t + 1, min(t + 1 + w, m.T))
        d = np.hypot(bx[seg] - bx[t], by[seg] - by[t])
        if np.nanmax(d, initial=0.0) > KICK_MOVE_M:
            moved = np.flatnonzero(np.nan_to_num(d, nan=0.0) > REST_M)
            out.append(t + int(moved[0]) if len(moved) else t)   # the last frame at rest
            break
    return out


def _rows_of_frames(m, f0, f1):
    """Rows whose video frame is in [f0, f1)."""
    fi = m.frame_index
    return int(np.searchsorted(fi, f0, "left")), int(np.searchsorted(fi, f1, "left"))


def kickoff_rows(m, tags=None):
    """{period: (row of the kick-off, source)}; source 'tag-kick', 'tag-start' or 'none' (no
    kick-off tag: the period is left as it is)."""
    fps = m.fps
    out = {}
    half_name = {1: "1st half", 2: "2nd half", 3: "1st extra", 4: "2nd extra"}
    for p, s, e in m.period_slices():
        found = None
        if tags:
            ko = [t for t in tags if (t.get("code") or "").strip().lower() == "set pieces"
                  and (t.get("tags", {}).get("Type") or "").strip().lower() == "kick off (start)"
                  and (t.get("tags", {}).get("Half") or "").strip().lower().startswith(half_name.get(p, "?")[:3])]
            for t in sorted(ko, key=lambda t: t["start_frame"]):
                a, b = _rows_of_frames(m, t["start_frame"] - int(TAG_PAD_S * fps),
                                       t["end_frame"] + int(TAG_PAD_S * fps))
                a, b = max(a, s), min(b, e)
                if a >= b:
                    continue
                k = _kick_frames(m, a, b)
                if k:
                    found = (k[0], "tag-kick")
                else:
                    found = (max(_rows_of_frames(m, t["start_frame"], t["start_frame"] + 1)[0], s),
                             "tag-start")
                break
        out[p] = found if found is not None else (s, "none")
    return out


def apply_kickoff(labels, m, tags=None):
    """labels with D from the start of each period to its kick-off. Returns (labels, info)."""
    out = labels.copy()
    info = {}
    for p, s, e in m.period_slices():
        k, src = kickoff_rows(m, tags).get(p, (s, "none"))
        gap = (k - s) / m.fps
        changed = 0
        if src != "none" and gap >= MIN_GAP_S and k - s <= SEARCH_MIN * 60 * m.fps:
            seg = out[s:k]
            changed = int(np.sum((seg != D) & (seg != X)))
            out[s:k] = np.where(seg == X, X, D)
        info[p] = {"kickoff_row": int(k), "kickoff_video_frame": int(m.frame_index[k]),
                   "source": src, "seconds_after_period_start": round(gap, 1),
                   "frames_set_D": changed}
    return out, info
