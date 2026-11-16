"""Labels H / A / D / X from event data, by one mechanical rule (Metrica event vocabulary).

CHALLENGE.md: a possession is a maximal run of one team with the ball alive, the ball is dead from
the moment it leaves play (or the whistle) to the restart, and possession stays with the last team
until the other team gains control. Read on the events:

- before the first SET PIECE of a period: X (not graded; the provided ground truth starts at the
  kick-off);
- a SET PIECE (kick-off, throw-in, free kick, corner, goal kick), or a PASS whose subtype says
  GOAL KICK, *restarts* play at its start frame for its team;
- BALL OUT: dead from the first frame the ball is unseen after the event's start (the crossing),
  else from its end frame;
- FAULT RECEIVED: dead from its start frame (the whistle);
- a SHOT whose subtype contains GOAL (not SAVED / OUT / BLOCKED): dead from the end frame;
- CARD events change nothing (the ball is already dead);
- while alive, the team is the team of the last PASS / CARRY / RECOVERY / SHOT / BALL LOST /
  SET PIECE: a BALL LOST keeps the ball with the loser until the opponent's RECOVERY or PASS,
  as CHALLENGE.md asks ("a simple touch is not a change"); CHALLENGE events never change it.

Event frames are source frame ids; `match.frame_index` maps them to rows (frame_id - 1).
"""

import numpy as np

from .match import H, A, D, X

RESTART_TYPES = {"SET PIECE"}
POSSESSION_TYPES = {"PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST", "SET PIECE"}


def labels_from_events(m, events):
    T = m.T
    labels = np.full(T, X, dtype=np.int8)
    row_of = {int(f): i for i, f in enumerate(m.frame_index + 1)}
    unseen = np.isnan(m.ball[:, 0])

    def row(frame):
        if frame in row_of:
            return row_of[frame]
        # nearest known frame
        keys = np.fromiter(row_of.keys(), dtype=np.int64)
        return row_of[int(keys[np.argmin(np.abs(keys - frame))])]

    for p, s, e in m.period_slices():
        evs = sorted([ev for ev in events if ev["period"] == p], key=lambda ev: (ev["start"], 0))
        if not evs:
            continue
        state = None            # None = before kick-off
        cursor = s
        timeline = []           # (row, new_state)
        for ev in evs:
            t, st, r0 = ev["type"], ev["subtype"], row(ev["start"])
            r1 = row(ev["end"]) if ev["end"] > 0 else r0
            if t == "SET PIECE" or (t == "PASS" and "GOAL KICK" in st):
                timeline.append((r0, ev["team"]))
            elif t == "BALL OUT":
                r = r1 if r1 > r0 else r0
                # first unseen ball between start and end (the crossing), if any
                seg = unseen[r0:max(r1, r0) + 1]
                if seg.any():
                    r = r0 + int(np.argmax(seg))
                timeline.append((r, D))
            elif t == "FAULT RECEIVED":
                timeline.append((r0, D))
            elif t == "SHOT" and "GOAL" in st and not any(k in st for k in ("SAVED", "OUT", "BLOCK")):
                timeline.append((max(r1, r0), D))
            elif t in POSSESSION_TYPES:
                timeline.append((r0, ev["team"]))
        timeline.sort(key=lambda x: x[0])
        cur = X
        prev_row = s
        for r, new in timeline:
            r = min(max(r, s), e - 1)
            labels[prev_row:r] = cur
            if cur == X and new in (H, A):
                cur = new            # kick-off: play begins
            elif new == D:
                cur = D
            elif new in (H, A):
                cur = new            # a team event ends a stoppage (restart) or changes the team
            prev_row = r
        labels[prev_row:e] = cur
    return labels
