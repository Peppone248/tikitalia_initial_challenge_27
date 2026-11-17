import numpy as np

from seqmodel.match import attack_direction, controlled_labels, load_idsse

H, A, D, X = 0, 1, 2, 3


def test_controlled_labels_removes_short_flip():
    lab = np.array([H] * 50 + [A] * 10 + [H] * 50, dtype=np.int8)
    assert (controlled_labels(lab) == H).all()
    lab = np.array([H] * 50 + [A] * 30 + [H] * 50, dtype=np.int8)
    assert (controlled_labels(lab) == lab).all()


def test_controlled_labels_never_touches_d():
    lab = np.array([H] * 50 + [A] * 10 + [H] * 50 + [D] * 5 + [A] * 10 + [D] * 5 + [H] * 50,
                   dtype=np.int8)
    out = controlled_labels(lab)
    assert (out[lab == D] == D).all()
    assert (out[(lab != D)] != D).all()
    assert (out[50:60] == H).all()                  # flip between two H runs removed
    assert (out[115:125] == A).all()                # short A run bounded by D stays
    short_d = np.array([H] * 50 + [D] * 5 + [A] * 50, dtype=np.int8)
    assert (controlled_labels(short_d) == short_d).all()


def test_attack_direction_flips_between_halves(synthetic_match):
    d = attack_direction(synthetic_match())
    assert d[1] is True and d[2] is False


def test_load_idsse_roundtrip(tmp_path):
    T, P = 6, 4
    alive = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    poss = np.array([1, 2, 0, 1, 2, 2], dtype=np.int8)
    px = np.tile(np.array([-40.0, -10.0, 40.0, 10.0], dtype=np.float32), (T, 1))
    path = tmp_path / "m.npz"
    np.savez(path, period=np.array([1, 1, 1, 2, 2, 2], dtype=np.int8), ball_x=np.zeros(T),
             ball_y=np.zeros(T), ball_alive=alive, ball_possession=poss, px=px,
             py=np.zeros((T, P), dtype=np.float32),
             player_team=np.array([0, 0, 1, 1], dtype=np.int8),
             pitch_x=np.float64(0), pitch_y=np.float64(0))
    m = load_idsse(str(path))
    assert m.label.tolist() == [H, A, X, D, D, A]
    assert (m.length, m.width) == (105.0, 68.0)
    assert m.px.shape == (T, P)
    assert m.home_attacks_right == {1: True, 2: True}
    assert m.ball_z is None                 # source without ball height
    z = dict(np.load(path))
    z["ball_z"] = np.arange(T, dtype=np.float64)
    np.savez(path, **z)
    assert load_idsse(str(path)).ball_z.tolist() == list(range(T))


def test_attack_direction_nan_heavy_and_late_substitute():
    import warnings
    import pytest
    from seqmodel.match import Match
    T = 200
    px = np.tile(np.array([-45.0, -20.0, -10.0, 45.0, 20.0, 10.0, 30.0], dtype=np.float32), (T, 1))
    px[:, 2] = np.nan                       # always missing column
    px[:100, 1] = np.nan                    # late substitute
    px[:50, :] = np.nan                     # all-NaN frames
    py = np.zeros((T, 7), dtype=np.float32)
    m = Match("t", px, py, np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8), np.zeros(T),
              np.zeros(T), np.ones(T, dtype=np.int8), np.zeros(T, dtype=np.int8), 105.0, 68.0, {})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert attack_direction(m) == {1: True}
        m.px[:, :3] = np.nan
        with pytest.raises(ValueError):
            attack_direction(m)
