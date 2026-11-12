import numpy as np
import pytest

from seqmodel.decode import count_transitions, decode

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def runs(x):
    b = np.flatnonzero(np.diff(x)) + 1
    s = np.r_[0, b]
    e = np.r_[b, len(x)]
    return [(int(x[i]), int(j - i)) for i, j in zip(s, e)]


def onehot(lab, conf=0.9):
    p = np.full((len(lab), 3), (1 - conf) / 2)
    p[np.arange(len(lab)), lab] = conf
    return p


def test_min_durations():
    lab = np.r_[np.zeros(100), np.ones(8), np.zeros(100), np.ones(100), np.full(30, 2),
                np.ones(100), np.zeros(100), np.full(60, 2), np.zeros(100)].astype(int)
    out = decode(onehot(lab, 0.95), None, np.ones(len(lab)))
    r = runs(out)
    assert len(r) > 1
    for c, n in r[1:-1]:
        assert n >= (50 if c == 2 else 20)
    assert (out != lab).any()  # the short blips were really overruled


def test_constant_probs_no_change():
    p = np.tile([0.9, 0.05, 0.05], (500, 1))
    out = decode(p, None, np.ones(500))
    assert out.dtype == np.int8 and (out == 0).all()


def test_beta_moves_change():
    p = np.zeros((260, 3))
    p[:, :2] = 0.5
    p[:30] = [1, 0, 0]
    p[230:] = [0, 1, 0]
    pc = np.zeros(260)
    pc[120] = 1.0
    out = decode(p, pc, np.ones(260), beta=20.0)
    ch = int(np.flatnonzero(np.diff(out) != 0)[0]) + 1
    assert abs(ch - 120) <= 12
    assert out[0] == 0 and out[-1] == 1


def test_degenerate_probs():
    rng = np.random.default_rng(0)
    p = np.eye(3)[rng.integers(0, 3, 300)]
    for q in (p, np.full((300, 3), 1 / 3)):
        out = decode(q, rng.random(300), np.ones(300), beta=5.0)
        assert set(np.unique(out)) <= {0, 1, 2}


def test_halves_independent():
    p1 = onehot(np.ones(100, int), 0.99)
    p2 = np.tile([0.5, 0.45, 0.05], (100, 1))
    p2[:, 0] = 0.5
    p = np.vstack([p1, p2])
    per = np.r_[np.ones(100), 2 * np.ones(100)]
    out = decode(p, None, per)
    assert (out[:100] == 1).all() and out[100] == 0


def test_count_transitions():
    lab = np.r_[np.zeros(100), np.ones(40), np.full(10, 3), np.zeros(30)].astype(int)
    t = count_transitions([lab], [np.ones(len(lab))])
    assert set(t) == {"HA", "HD", "AH", "AD", "DH", "DA"}
    assert t["HA"] == pytest.approx(2 / (90 + 1))     # one H->A, H beyond min = 80 + 10
    assert t["AH"] == pytest.approx(1 / (20 + 1))     # A->X ignored, X->H ignored
    assert t["HD"] == pytest.approx(1 / (90 + 1))
    # period change splits runs and is never a transition
    per = np.r_[np.ones(100), 2 * np.ones(40)]
    t2 = count_transitions([np.r_[np.zeros(100), np.ones(40)]], [per])
    assert t2["HA"] == pytest.approx(1 / (80 + 1))


def test_degenerate_trans_from_counts():
    t = count_transitions([np.zeros(100, int)], [np.ones(100)])
    assert t["AH"] == 1.0 and t["DH"] == 1.0
    p = np.random.default_rng(1).dirichlet([1, 1, 1], 300)
    out = decode(p, np.random.default_rng(2).random(300), np.ones(300), beta=5.0, trans=t)
    assert out.dtype == np.int8 and set(np.unique(out)) <= {0, 1, 2}
