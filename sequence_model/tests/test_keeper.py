"""Goalkeeper fix: hold detection in the degradation, the keeper_ball feature, keeper identity cleaning."""
import numpy as np
import pytest

from seqmodel.clean import clean_keepers
from seqmodel.degrade import degrade, keeper_holds
from seqmodel.features import FEATURES, ODD, frame_features, swap_teams
from seqmodel.match import CFG, Match

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
DC = CFG["degrade"]
KB = FEATURES.index("keeper_ball")


def _match(T=300, hold=None, z=1.2, offset=0.3, keeper_x=-50.0):
    """Home attacks right. Columns: H keeper, H outfield x2, A keeper, A outfield x2. Ball at the
    centre (on the ground), except on rows `hold` (slice) where it sits `offset` m from the column
    given by hold_col at height z."""
    px = np.tile(np.array([keeper_x, -10, 5, 50, 10, -5], np.float32), (T, 1))
    py = np.tile(np.array([0, -10, 10, 0, -10, 10], np.float32), (T, 1))
    bx, by, bz = np.zeros(T), np.full(T, 20.0), np.full(T, 0.1)
    m = Match("k", px, py, np.array([0, 0, 0, 1, 1, 1], np.int8), bx, by, np.ones(T, np.int8),
              np.zeros(T, np.int8), 105.0, 68.0, {1: True}, ball_z=bz)
    if hold is not None:
        rows, col = hold
        m.ball_x[rows] = px[rows, col] + offset
        m.ball_y[rows] = py[rows, col]
        m.ball_z[rows] = z
    return m


# ---------- part 1: hold detection and hiding ----------

def test_hold_detected_for_each_team():
    assert keeper_holds(_match(hold=(slice(50, 100), 0)), DC) == [(50, 100, 0)]
    assert keeper_holds(_match(hold=(slice(120, 200), 3)), DC) == [(120, 200, 1)]


@pytest.mark.parametrize("kw", [dict(z=0.1), dict(offset=5.0), dict(keeper_x=-30.0)])
def test_hold_negatives(kw):
    """Ball on the ground, ball far from the keeper, keeper (deepest player) outside his box."""
    assert keeper_holds(_match(hold=(slice(50, 100), 0), **kw), DC) == []


def test_hold_too_short_and_no_height():
    assert keeper_holds(_match(hold=(slice(50, 60), 0)), DC) == []          # 0.4 s < 1 s
    m = _match(hold=(slice(50, 100), 0))
    m.ball_z = None
    assert keeper_holds(m, DC) == []


def test_hold_cut_at_period_boundary():
    m = _match(hold=(slice(50, 250), 0))
    m.period[150:] = 2
    m.home_attacks_right = {1: True, 2: True}
    assert keeper_holds(m, DC) == [(50, 150, 0), (150, 250, 0)]


def test_degrade_hides_holds():
    m = _match(T=3000, hold=(slice(500, 600), 0))
    m.label[:] = 0
    m.label[2000:2100] = 2
    always = degrade(m, 3, {**DC, "keeper_hide_p": 1.0})
    assert np.isnan(always.ball_x[500:600]).all() and np.isnan(always.ball_z[500:600]).all()
    assert np.array_equal(always.label, m.label)
    # keeper_hide_p 0 is the previous degradation exactly (separate random stream)
    off = degrade(m, 3, {**DC, "keeper_hide_p": 0.0})
    old = degrade(m, 3, {k: v for k, v in DC.items() if not k.startswith("keeper_")})
    assert np.array_equal(off.ball_x, old.ball_x, equal_nan=True)
    assert np.array_equal(off.px, old.px, equal_nan=True)


# ---------- part 2: keeper_ball feature ----------

def test_keeper_ball_values():
    assert ODD[KB]
    f = frame_features(_match(hold=(slice(50, 100), 0)))[:, KB]
    assert (f[50:100] == 1).all() and (f[:50] == 0).all() and (f[100:] == 0).all()
    f = frame_features(_match(hold=(slice(50, 100), 3)))[:, KB]
    assert (f[50:100] == -1).all()
    # within keeper_r on the ground also counts (no height in the feature); outside the box does not
    assert (frame_features(_match(hold=(slice(50, 100), 0), z=0.1, offset=2.5))[50:100, KB] == 1).all()
    assert (frame_features(_match(hold=(slice(50, 100), 0), offset=4.0))[50:100, KB] == 0).all()
    assert (frame_features(_match(hold=(slice(50, 100), 0), keeper_x=-30.0))[50:100, KB] == 0).all()


def test_keeper_ball_uses_last_seen_ball():
    m = _match(hold=(slice(50, 100), 0))
    m.ball_x[60:200], m.ball_y[60:200] = np.nan, np.nan
    f = frame_features(m)[:, KB]
    assert (f[50:200] == 1).all() and (f[200:] == 0).all()


def test_keeper_ball_antisymmetry(synthetic_match):
    m = synthetic_match(7, 2000)
    for k in (0, 11):                    # the synthetic keepers stand at y = -25 (outside the box): centre them
        near = np.hypot(m.ball_x - m.px[:, k], m.ball_y - m.py[:, k]) < 2.0
        m.ball_y[near] += 25.0
        m.py[:, k] += 25.0
    m.ball_x[400:700], m.ball_y[400:700] = np.nan, np.nan
    f, g = frame_features(m)[:, KB], frame_features(swap_teams(m))[:, KB]
    assert (f != 0).any()
    assert np.array_equal(g, -f)


# ---------- part 3: keeper identity cleaning ----------

def _teleport_match(T=600):
    m = _match(T=T)
    rng = np.random.default_rng(0)
    m.px += rng.normal(0, 0.2, m.px.shape).astype(np.float32)
    m.py += rng.normal(0, 0.2, m.py.shape).astype(np.float32)
    # rows 100-250: the home keeper slot follows the AWAY keeper (jump of ~100 m), frozen from 200
    m.px[100:250, 0] = 49.0 + rng.normal(0, 0.2, 150).astype(np.float32)
    m.py[100:250, 0] = 1.0 + rng.normal(0, 0.2, 150).astype(np.float32)
    m.px[200:250, 0], m.py[200:250, 0] = 48.5, 1.5
    return m


def test_clean_moves_teleported_keeper_stretch():
    m = _teleport_match()
    c, fixes = clean_keepers(m)
    assert len(fixes) == 1
    f = fixes[0]
    assert (f["team"], f["col"], f["start"], f["end"]) == (0, 0, 100, 250)
    assert f["moved"] == 100 and f["dropped"] == 50
    assert c.px.shape[1] == 7 and c.team[-1] == 1
    assert np.isnan(c.px[100:250, 0]).all() and np.isfinite(c.px[:100, 0]).all()
    assert np.allclose(c.px[100:200, 6], m.px[100:200, 0]) and np.isnan(c.px[200:250, 6]).all()
    assert np.isnan(c.px[:100, 6]).all() and np.isnan(c.px[250:, 6]).all()
    assert np.array_equal(m.px[100:250, 0], _teleport_match().px[100:250, 0])     # input untouched


def test_clean_leaves_keeper_running_up_for_a_corner():
    m = _match(T=900)
    run = np.linspace(-50, 45, 200)                      # 0.48 m per frame, from his own box
    m.px[100:300, 0] = run
    m.px[300:500, 0] = 45.0 + 0.1 * np.sin(np.arange(200))
    m.px[500:700, 0] = run[::-1]
    c, fixes = clean_keepers(m)
    assert fixes == [] and c is m
    # also when his track is lost for a moment inside the opponent's box (a new stretch starts there,
    # but it reaches his own half again before the next gap)
    m.px[350:360, 0] = np.nan
    assert clean_keepers(m)[1] == []


def test_clean_noop_on_clean_synthetic(synthetic_match):
    m = synthetic_match(0, 3000)
    c, fixes = clean_keepers(m)
    assert fixes == [] and c is m


def test_clean_needs_trusted_slot():
    m = _teleport_match()
    assert clean_keepers(m, {**CFG["clean"], "min_keeper_share": 1.01})[1] == []
