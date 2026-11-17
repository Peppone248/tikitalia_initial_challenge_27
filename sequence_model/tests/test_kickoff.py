import dataclasses

import numpy as np

from seqmodel.kickoff import apply_kickoff, detect_kickoff, find_kickoffs
from seqmodel.match import Match

FORM = 75       # 3 s of formation before the kick


def build(kick=(400, 400), seed=0, wrong_halves=False, no_kicker=False, ball=True):
    """Two periods of 800 frames. Before kick[p-1]-FORM the players mill around both halves; then
    3 s of formation (own halves, a home kicker at the spot); from the kick on they mix again."""
    rng = np.random.default_rng(seed)
    T, P = 1600, 22
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    period = np.where(np.arange(T) < 800, 1, 2).astype(np.int8)
    px = np.tile(rng.uniform(-40, 40, P), (T, 1)) + rng.normal(0, .05, (T, P))   # walking: still
    py = np.tile(rng.uniform(-25, 25, P), (T, 1)) + rng.normal(0, .05, (T, P))
    bx, by = np.full(T, np.nan), np.zeros(T)
    for p, k in ((1, kick[0]), (2, kick[1])):
        s = 800 * (p - 1)
        f = slice(s + k - FORM, s + k)
        side = 1.0 if p == 1 else -1.0           # home in x<0 in period 1
        own = rng.uniform(12, 40, (FORM, P))
        px[f, :11] = -side * own[:, :11]
        px[f, 11:] = side * own[:, 11:]
        if wrong_halves:
            px[f, :11] = np.tile(np.linspace(-40, 40, 11), (FORM, 1))
        if not no_kicker:
            px[f, 0], py[f, 0] = 0.4, 0.0
        if ball:
            bx[s + k - FORM:s + k + 10] = 0.2
    return Match(name="syn", px=px.astype(np.float32), py=py.astype(np.float32), team=team,
                 ball_x=bx, ball_y=by, period=period, label=np.zeros(T, dtype=np.int8),
                 length=105., width=68., home_attacks_right={1: True, 2: False})


def test_detect_known_frame():
    m = build((400, 250))
    assert abs(detect_kickoff(m, 1) - 400) <= 3
    assert abs(detect_kickoff(m, 2) - (800 + 250)) <= 3


def test_ball_is_not_required():
    m = build((400, 250), ball=False)
    assert abs(detect_kickoff(m, 1) - 400) <= 3
    assert detect_kickoff(m, 1, info=True)["ball_seen"] is False
    assert detect_kickoff(build((400, 250)), 1, info=True)["ball_seen"] is True


def test_no_formation_or_no_kicker_gives_none():
    assert detect_kickoff(build(wrong_halves=True), 1) is None
    assert detect_kickoff(build(no_kicker=True), 1) is None


def test_late_formation_is_not_the_period_start():
    m = build((700, 700))               # formation 700 rows after the start: 28 s, inside search_s
    assert abs(detect_kickoff(m, 1) - 700) <= 3
    m = build((400, 400))
    short = dataclasses.replace(m, period=np.where(np.arange(1600) < 100, 1, 2).astype(np.int8))
    assert detect_kickoff(short, 2) is None or detect_kickoff(short, 2) >= 100


def test_apply_sets_exactly_before(capsys):
    m = build((400, 250))
    lab = np.zeros(1600, dtype=np.int8)
    out = apply_kickoff(lab, m, (2,))
    k = detect_kickoff(m, 2)
    assert (out[:800] == 0).all() and (out[800:k] == 2).all() and (out[k:] == 0).all()
    assert (lab == 0).all()
    bad = build(wrong_halves=True)
    assert (apply_kickoff(lab, bad, (1,)) == 0).all()
    assert "WARNING" in capsys.readouterr().err


def test_formation_beyond_window_leaves_period_unchanged(capsys):
    from seqmodel.kickoff import KC
    m = build((700, 700))               # 700 frames = 28 s after the period start
    lab = np.zeros(1600, dtype=np.int8)
    old = KC["max_after_start_s"]
    KC["max_after_start_s"] = 10         # window 250 frames: the formation at 700 is outside
    try:
        assert detect_kickoff(m, 1) is None
        assert (apply_kickoff(lab, m, (1,)) == lab).all()
        assert "WARNING" in capsys.readouterr().err
    finally:
        KC["max_after_start_s"] = old


def test_running_play_before_formation_is_a_restart_not_a_start():
    m = build((700, 700))
    px = m.px.copy()
    t = np.arange(600)[:, None]
    px[:600] = 30 * np.sin(t / 6.0 + np.arange(22)[None, :])     # all players running fast
    assert detect_kickoff(dataclasses.replace(m, px=px), 1) is None
    assert abs(detect_kickoff(m, 1) - 700) <= 3
