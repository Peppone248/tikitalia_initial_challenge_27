"""Unit tests: symmetry of the features, the decoder on synthetic data, the EPTS parser, the
labels rule, and that two predictions of the same input are identical.

    python -m pytest changepoint_model/tests -q
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cpm.match import Match, H, A, D, X, spells  # noqa: E402
from cpm.features import frame_features  # noqa: E402
from cpm import decode as dec  # noqa: E402
from cpm.models import boundary_targets, TeamModel  # noqa: E402
from cpm.labels import labels_from_events  # noqa: E402
from cpm.degrade import degrade  # noqa: E402
from cpm.loaders.epts import load_epts  # noqa: E402


def synthetic_match(T=3000, seed=0):
    rng = np.random.RandomState(seed)
    P = 22
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    period = np.ones(T, dtype=np.int16)
    period[: T // 2] = 1
    period[T // 2:] = 2
    px = np.cumsum(rng.randn(T, P) * 0.1, axis=0) + rng.uniform(-40, 40, P)
    py = np.cumsum(rng.randn(T, P) * 0.1, axis=0) + rng.uniform(-25, 25, P)
    labels = np.zeros(T, dtype=np.int8)
    t = 0
    s = H
    while t < T:
        L = rng.randint(40, 400)
        labels[t:t + L] = s
        t += L
        s = rng.choice([x for x in (H, A, D) if x != s])
    # ball near a player of the team in possession, unseen when dead
    ball = np.full((T, 2), np.nan)
    for t in range(T):
        if labels[t] < D:
            cands = np.flatnonzero(team == labels[t])
            j = cands[(t // 50) % len(cands)]
            ball[t] = (px[t, j] + 0.3, py[t, j] - 0.2)
    return Match("synthetic", period, ball, px, py, team, (105.0, 68.0), labels=labels)


def test_features_symmetry():
    m = synthetic_match()
    ev, od, en, on = frame_features(m)
    ev2, od2, _, _ = frame_features(m.swapped())
    assert np.abs(ev - ev2).max() < 1e-5
    assert np.abs(od + od2).max() < 1e-5
    assert not np.isnan(ev).any() and not np.isnan(od).any()


def test_team_model_antisymmetric():
    m = synthetic_match()
    _, od, _, on = frame_features(m)
    alive = m.labels < D
    tm = TeamModel(1.0).fit(od[alive], (m.labels[alive] == H).astype(float), on)
    p = tm.predict_proba(od)
    p2 = tm.predict_proba(-od)
    assert np.allclose(p + p2, 1.0, atol=1e-6)
    assert np.mean((p[alive] > 0.5) == (m.labels[alive] == H)) > 0.9


def test_viterbi_recovers_truth():
    rng = np.random.RandomState(1)
    T = 4000
    truth = np.zeros(T, dtype=int)
    t, s = 0, 0
    while t < T:
        L = rng.randint(60, 400)
        truth[t:t + L] = s
        t += L
        s = rng.choice([x for x in (0, 1, 2) if x != s])
    p = np.full((T, 3), 0.1)
    p[np.arange(T), truth] = 0.8
    p[rng.rand(T) < 0.2] = 1 / 3.0
    ph = p[:, 0] / (p[:, 0] + p[:, 1])
    logE = dec.emissions(ph, p[:, 2], [0.4, 0.3, 0.3], w=1.0)
    hc, hs, hr = dec.constant_hazards(T)
    lab = dec.viterbi(logE, hc, hs, hr)
    assert np.mean(lab == truth) > 0.98
    # boundary hazards: a sharp hazard at the true changes recovers them even with weak emissions
    bnd = np.zeros(T)
    for t in np.flatnonzero(truth[1:] != truth[:-1]) + 1:
        bnd[max(0, t - 12):t + 13] = 1.0
    hc2, hs2, hr2 = dec.hazards_from_boundaries(bnd, bnd, bnd, k=12, gamma=4.0)
    lab2 = dec.viterbi(dec.emissions(ph, p[:, 2], [0.4, 0.3, 0.3], w=0.3), hc2, hs2, hr2)
    assert np.mean(lab2 == truth) > 0.98


def test_viterbi_minimum_durations():
    T = 500
    truth = np.zeros(T, dtype=int)
    truth[100:105] = 1          # a 5-frame blip: below the minimum of 20 frames
    p = np.full((T, 3), 0.05)
    p[np.arange(T), truth] = 0.9
    ph = p[:, 0] / (p[:, 0] + p[:, 1])
    lab = dec.viterbi(dec.emissions(ph, p[:, 2], [0.4, 0.4, 0.2]), *dec.constant_hazards(T))
    runs = spells(lab, np.ones(T, dtype=int))
    assert all(e - s >= 20 for l, s, e in runs if l < D)


def test_boundary_targets():
    labels = np.array([H] * 30 + [A] * 30 + [D] * 60 + [H] * 30, dtype=np.int8)
    period = np.ones(len(labels), dtype=np.int16)
    ch, st, re = boundary_targets(labels, period, k=2)
    assert ch[28:33].all() and not ch[:28].any() and not ch[33:].any()
    assert st[58:63].all()
    assert re[118:123].all()


def test_labels_from_events():
    m = synthetic_match(T=1000)
    m.labels = None
    events = [
        {"team": 0, "type": "SET PIECE", "subtype": "KICK OFF", "period": 1, "start": 11, "end": 11},
        {"team": 0, "type": "PASS", "subtype": "", "period": 1, "start": 11, "end": 30},
        {"team": 1, "type": "RECOVERY", "subtype": "INTERCEPTION", "period": 1, "start": 101, "end": 101},
        {"team": 1, "type": "BALL OUT", "subtype": "", "period": 1, "start": 201, "end": 220},
        {"team": 0, "type": "SET PIECE", "subtype": "THROW IN", "period": 1, "start": 301, "end": 301},
        {"team": 0, "type": "PASS", "subtype": "", "period": 1, "start": 301, "end": 320},
    ]
    m.ball[:] = 1.0
    lab = labels_from_events(m, events)
    assert (lab[:10] == X).all()
    assert (lab[10:100] == H).all()
    assert (lab[100:219] == A).all()      # ball out: dead from its end frame (row 219)
    assert (lab[219:300] == D).all()
    assert (lab[300:500] == H).all()


def test_degrade_deterministic_and_hides_dead_ball():
    m = synthetic_match()
    d1 = degrade(m, "degraded", 2027)
    d2 = degrade(m, "degraded", 2027)
    assert np.array_equal(np.isnan(d1.ball[:, 0]), np.isnan(d2.ball[:, 0]))
    assert np.array_equal(np.isnan(d1.px), np.isnan(d2.px))
    alive = m.labels < D
    assert np.mean(~np.isnan(d1.ball[alive, 0])) < np.mean(~np.isnan(m.ball[alive, 0]))
    assert np.isnan(d1.px).mean() > np.isnan(m.px).mean()


def test_epts_parser(tmp_path):
    meta = """<?xml version="1.0"?><main><Metadata><GlobalConfig><FrameRate>25</FrameRate>
    <ProviderGlobalParameters>
    <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
    <ProviderParameter><Name>first_half_end</Name><Value>2</Value></ProviderParameter>
    <ProviderParameter><Name>second_half_start</Name><Value>3</Value></ProviderParameter>
    <ProviderParameter><Name>second_half_end</Name><Value>4</Value></ProviderParameter>
    </ProviderGlobalParameters></GlobalConfig>
    <Sessions><Session id="s"><FieldSize><Width>100</Width><Height>60</Height></FieldSize></Session></Sessions>
    <Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
    <Players><Player id="P1" teamId="T1"><Name>a</Name></Player><Player id="P2" teamId="T2"><Name>b</Name></Player></Players>
    <PlayerChannels>
    <PlayerChannel channelId="x" id="p1_x" playerId="P1"/><PlayerChannel channelId="y" id="p1_y" playerId="P1"/>
    <PlayerChannel channelId="s" id="p1_s" playerId="P1"/>
    <PlayerChannel channelId="x" id="p2_x" playerId="P2"/><PlayerChannel channelId="y" id="p2_y" playerId="P2"/>
    <PlayerChannel channelId="s" id="p2_s" playerId="P2"/>
    </PlayerChannels></Metadata>
    <DataFormatSpecifications><DataFormatSpecification startFrame="1" endFrame="4" separator=":">
    <StringRegister name="frameCount"/>
    <SplitRegister separator=";">
      <SplitRegister separator=","><PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/><PlayerChannelRef playerChannelId="p1_s"/></SplitRegister>
      <SplitRegister separator=","><PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/><PlayerChannelRef playerChannelId="p2_s"/></SplitRegister>
    </SplitRegister>
    <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/></SplitRegister>
    </DataFormatSpecification></DataFormatSpecifications></main>"""
    raw = "1:0.25,0.5,1.0;0.75,0.5,2.0:0.5,0.5\n2:NaN,NaN,NaN;0.75,0.25,2.0:NaN,NaN\n" \
          "3:0.0,0.0,0.0;1.0,1.0,0.0:0.1,0.9\n4:0.5,0.5,0.0;0.5,0.5,0.0:0.5,0.5\n"
    mp, rp = tmp_path / "m.xml", tmp_path / "r.txt"
    mp.write_text(meta)
    rp.write_text(raw)
    m = load_epts(str(mp), str(rp))
    assert m.T == 4 and m.n_players == 2
    assert list(m.team) == [0, 1]
    assert np.isclose(m.px[0, 0], -25.0) and np.isclose(m.px[0, 1], 25.0)
    assert np.isnan(m.px[1, 0]) and np.isclose(m.py[1, 1], 15.0)
    assert np.isnan(m.ball[1, 0]) and np.isclose(m.ball[0, 0], 0.0)
    assert list(m.period) == [1, 1, 2, 2]
    assert list(m.frame_index) == [0, 1, 2, 3]


def test_prediction_is_deterministic():
    from cpm import pipeline
    m = synthetic_match(T=6000, seed=3)
    P = pipeline.prepare(m, "degraded", 1)
    models = pipeline.fit([P], {"backend": "logistic"})
    pr1 = pipeline.predict(models, P)
    pr2 = pipeline.predict(models, P)
    lab1 = pipeline.decode(pr1, P.period, models)
    lab2 = pipeline.decode(pr2, P.period, models)
    assert np.array_equal(lab1, lab2)
    assert set(np.unique(lab1)) <= {H, A, D}


def _kickoff_match():
    """Two halves; the second has 100 s of warm-up (players scattered, a ball moving about) before
    its kick-off from the centre spot."""
    rng = np.random.RandomState(5)
    fps = 25
    T1, warm, T2 = 3000, 100 * fps, 3000
    T = T1 + warm + T2
    period = np.array([1] * T1 + [2] * (warm + T2), dtype=np.int16)
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    px = rng.uniform(-50, 50, (T, 22)).astype(np.float32)
    py = rng.uniform(-30, 30, (T, 22)).astype(np.float32)
    ball = np.stack([rng.uniform(-40, 40, T), rng.uniform(-25, 25, T)], axis=1)
    k1, k2 = 0, T1 + warm
    for k in (k1, k2):
        ball[k:k + 5] = 0.0                      # on the centre spot, then kicked
        ball[k + 5:k + 30, 0] = np.linspace(0.5, 12, 25)
        ball[k + 5:k + 30, 1] = 0.0
    labels = np.zeros(T, dtype=np.int8)
    return Match("ko", period, ball, px, py, team, (105.0, 68.0), labels=labels,
                 attack={1: 1, 2: -1}), k2


def test_kickoff_from_tags():
    from cpm.kickoff import apply_kickoff
    m, k2 = _kickoff_match()
    tags = [{"code": "SET PIECES", "tags": {"Type": "Kick Off (Start)", "Half": "1st Half"},
             "start_frame": -60, "end_frame": 65},
            {"code": "SET PIECES", "tags": {"Type": "Kick Off (Start)", "Half": "2nd Half"},
             "start_frame": k2 - 60, "end_frame": k2 + 65}]
    lab = np.zeros(m.T, dtype=np.int8)              # the model said H everywhere
    out, info = apply_kickoff(lab, m, tags)
    assert info[1]["frames_set_D"] == 0             # first half starts at its kick-off
    assert info[2]["source"] == "tag-kick"
    assert abs(info[2]["kickoff_row"] - (k2 + 4)) <= 3      # k2 + 4: the last frame at rest
    s2 = int(np.flatnonzero(m.period == 2)[0])
    assert (out[s2:info[2]["kickoff_row"]] == D).all()
    assert (out[info[2]["kickoff_row"]:] == H).all() and (out[:s2] == H).all()


def test_kickoff_without_tags_changes_nothing():
    from cpm.kickoff import apply_kickoff
    m, _ = _kickoff_match()
    lab = np.zeros(m.T, dtype=np.int8)
    out, info = apply_kickoff(lab, m, None)
    assert np.array_equal(out, lab)
    assert all(v["source"] == "none" for v in info.values())
