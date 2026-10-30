"""Tests of HMM_first_imp on synthetic data (invented, no frame of BAR-ATH, no ground truth).

    python -m unittest discover -s HMM_first_imp/tests -v
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hmm_v0 import paths  # noqa: E402,F401
from hmm_v0.config import Config  # noqa: E402
from hmm_v0.engine import emissions, omega_matrix  # noqa: E402
from hmm_v0.hmm import BakisHMM  # noqa: E402
from hmm_v0 import noise, events, votes  # noqa: E402
from hmm_v0.rules.base import SafeOutput  # noqa: E402

FPS = 25
CFG = Config()


def hmm_cfg(**changes):
    import copy
    c = copy.deepcopy(CFG.hmm["hmm"])
    auto = {"auto_visible": 20.0, "auto_missing": 15.0}       # as estimated on a real game
    c["mean_time_s"] = {a: {b: auto.get(t, t) for b, t in row.items()}
                        for a, row in c["mean_time_s"].items()}
    c.update(changes)
    return c


def evidence(n, spans):
    """(n, 4) rule evidence in nats: spans = [(start, end, class, nats)]."""
    E = np.zeros((n, 4))
    for s, e, c, v in spans:
        E[s:e, votes.CLASSES.index(c)] += v
    return E


def nn_probs(n, spans=()):
    """NN output: neutral (0.25 each) except spans = [(start, end, class, p)]."""
    P = np.full((n, 4), 0.25)
    for s, e, c, p in spans:
        k = votes.CLASSES.index(c)
        P[s:e] = (1 - p) / 3.0
        P[s:e, k] = p
    return P


def decode(E, boost=None, cfg=None, P=None):
    cfg = cfg or hmm_cfg()
    n = E.shape[0]
    P = nn_probs(n) if P is None else P
    prior = np.full(4, 0.25)
    log_b = emissions(P, prior, E, cfg)
    hmm = BakisHMM(cfg, FPS)
    path, _ = hmm.viterbi(log_b, boost)
    out = np.array([cfg["output_label"][hmm.macros[i]] for i in path])
    return "".join(out), hmm


class TestBakisHMM(unittest.TestCase):

    def test_topology_is_left_to_right(self):
        hmm = BakisHMM(hmm_cfg(), FPS)
        self.assertEqual(hmm.n_sub[hmm.index["H"]], 20)          # 0.8 s at 25 fps
        self.assertEqual(hmm.S, sum(hmm.n_sub))
        # exits only from the last sub-state: the first n-1 sub-states have no self loop / exit
        self.assertEqual(hmm.log_rho, -1e18)

    def test_short_flip_is_a_touch_not_a_change(self):
        # H for 4 s, A evidence for 0.3 s (< min duration 0.8 s), then H again
        E = evidence(150, [(0, 150, "H", 3.0), (60, 68, "H", -3.0), (60, 68, "A", 6.0)])
        out, _ = decode(E)
        self.assertEqual(set(out), {"H"})

    def test_real_change_is_not_delayed(self):
        # H then A from frame 75: the change must be at 75 (+-1 frame is free for spellF1)
        E = evidence(200, [(0, 75, "H", 3.0), (75, 200, "A", 3.0)])
        out, _ = decode(E)
        self.assertEqual(out[:74], "H" * 74)
        self.assertEqual(out[76:], "A" * 124)
        self.assertLessEqual(abs(out.index("A") - 75), 1)

    def test_occlusion_keeps_the_possession_in_one_piece(self):
        # H visible, 4 s of ball not seen (X evidence only), H again: one possession, all H
        E = evidence(300, [(0, 100, "H", 3.0), (100, 200, "X", 2.5), (200, 300, "H", 3.0)])
        out, _ = decode(E, P=nn_probs(300, [(100, 200, "X", 0.7)]))
        self.assertEqual(out, "H" * 300)

    def test_occlusion_inertia_and_team_evidence(self):
        # A visible, then the ball is lost; the NN sees live play without the ball (its X).
        # Weak (SOFT-like) H evidence: inertia keeps A. Strong H evidence during the occlusion:
        # the change happens inside the occlusion.
        P = nn_probs(400, [(100, 400, "X", 0.7)])
        weak = evidence(400, [(0, 100, "A", 3.0), (100, 400, "X", 2.5), (150, 400, "H", 0.4)])
        out, _ = decode(weak, P=P)
        self.assertEqual(out, "A" * 400)
        strong = evidence(400, [(0, 100, "A", 3.0), (100, 400, "X", 2.5), (150, 400, "H", 3.0)])
        out, _ = decode(strong, P=P)
        self.assertEqual(out[:100], "A" * 100)
        self.assertEqual(out[-200:], "H" * 200)

    def test_dead_needs_a_restart_to_end(self):
        # D evidence, then weak H evidence: without a restart the D holds, with one it ends there
        E = evidence(400, [(0, 100, "H", 3.0), (100, 200, "D", 3.0), (200, 400, "H", 0.2)])  # between the two exit costs
        out_no, _ = decode(E)
        boost = np.zeros(400, dtype=bool)
        boost[195:206] = True
        out_yes, _ = decode(E, boost)
        self.assertGreater(out_no.count("D"), out_yes.count("D"))
        self.assertEqual(out_no[100:], "D" * 300)
        self.assertEqual(out_yes[210:], "H" * 190)

    def test_unobserved_stoppage_can_be_dead(self):
        # ball lost (X) during a stoppage with D evidence: D wins over XH
        E = evidence(300, [(0, 100, "H", 3.0), (100, 250, "X", 2.5), (100, 250, "D", 1.0),
                           (250, 300, "H", 3.0)])
        boost = np.zeros(300, dtype=bool)
        boost[245:256] = True
        out, _ = decode(E, boost)
        self.assertGreater(out[100:250].count("D"), 140)

    def test_team_gate_blocks_unsupported_changes(self):
        # 4 s of weak A evidence (0.8 nats/frame) inside a possession of H. With the gates open
        # where the rules say a team took the ball, the change happens; with both gates closed
        # (no rule behind the change) the possession stays in one piece.
        cfg = hmm_cfg()
        E = evidence(400, [(0, 100, "H", 3.0), (100, 200, "A", 0.8), (300, 400, "H", 3.0)])
        P = nn_probs(400)
        log_b = emissions(P, np.full(4, 0.25), E, cfg)
        hmm = BakisHMM(cfg, FPS)
        self.assertTrue(hmm.team_gate)
        closed = np.zeros(400, dtype=int)                           # regime 0: both closed
        opened = closed.copy()
        opened[95:126] |= 1                                         # A may take the ball here
        opened[195:226] |= 2                                        # H may take it back here
        lab = lambda path: "".join(cfg["output_label"][hmm.macros[i]] for i in path)
        out_closed = lab(hmm.viterbi(log_b, regime=closed)[0])
        out_open = lab(hmm.viterbi(log_b, regime=opened)[0])
        self.assertEqual(set(out_closed), {"H"})
        self.assertGreater(out_open[100:200].count("A"), 90)

    def test_omega_rows_normalised(self):
        for src in ("omega_rules", "omega_nn"):
            om = omega_matrix(hmm_cfg(), src)
            np.testing.assert_allclose(om.sum(axis=1), 1.0)


class TestNoise(unittest.TestCase):

    def test_mad_equals_std_on_gaussian_and_resists_outliers(self):
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.normal(0, 0.02, 5000)) + rng.normal(0, 0.1, 5000)
        clean = noise.estimate(x, 5)
        self.assertAlmostEqual(clean["mad"] / clean["std"], 1.0, delta=0.1)
        y = x.copy()
        y[rng.choice(5000, 100, replace=False)] += 5.0                # 2% tracker jumps
        dirty = noise.estimate(y, 5)
        # a spike enters 5 residuals through the moving average (~10% contaminated): MAD moves
        # by ~10-15%, std by a factor > 3
        self.assertLess(abs(dirty["mad"] / clean["mad"] - 1), 0.2)
        self.assertGreater(dirty["std"] / clean["std"], 3.0)

    def test_modes(self):
        est = {"mad": 0.02, "std": 0.3}
        self.assertEqual(noise.pick_sigma(est, "std", 0.5, 0.01), 0.3)
        self.assertEqual(noise.pick_sigma(est, "mad", 0.5, 0.01), 0.02)
        self.assertEqual(noise.pick_sigma(est, "mad_tol", 0.5, 0.01), 0.5)
        self.assertEqual(noise.pick_sigma({"mad": 0.0, "std": 0.0}, "mad", 0.5, 0.01), 0.01)

    def test_strength_saturates_and_clips(self):
        s = noise.strength([np.array([-1.0, 0.0, 1.5, 3.0, 9.0])], 3.0)
        np.testing.assert_allclose(s, [0, 0, 0.5, 1, 1])


class TestVotesAndOutput(unittest.TestCase):

    def test_weight_to_evidence(self):
        e = votes.weight_to_evidence(np.array([0.0, 0.5, 1.0]), 0.99)
        self.assertAlmostEqual(e[0], 0.0)                 # w = 0: chance, no evidence
        self.assertGreater(e[1], 0.0)
        self.assertAlmostEqual(e[2], np.log(3 * 0.99 / 0.01))

    def test_safe_output_keeps_the_stronger_label(self):
        out = SafeOutput("T", 10)
        out.emit(0, 6, "H", 0.9)
        out.emit(4, 10, "A", 0.5)
        self.assertEqual("".join(out.label), "HHHHHHAAAA")
        self.assertEqual(out.overlap_frames, 2)
        out.emit(0, 2, "A", 1.0)
        self.assertEqual("".join(out.label[:2]), "AA")


class _Match(object):
    """Minimal stand-in for the fields pass_chains reads."""
    def __init__(self, n):
        self.fps = FPS
        self.half = np.ones(n, dtype=np.int8)


class TestEvents(unittest.TestCase):

    def ep(self, team, start, end, slot, x0, x1):
        return {"team": team, "start": start, "end": end, "slot": slot, "z": np.ones(end - start),
                "dur_s": (end - start) / FPS, "x0": x0, "y0": 30.0, "x1": x1, "y1": 30.0}

    def test_pass_chain_and_break(self):
        p = CFG.params
        eps = [self.ep(0, 0, 20, 1, 10, 12), self.ep(0, 40, 60, 2, 30, 32),
               self.ep(0, 80, 100, 3, 50, 52), self.ep(1, 120, 140, 14, 60, 62),
               self.ep(0, 160, 180, 4, 40, 42)]
        chains = events.pass_chains(eps, _Match(200), p)
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0]["team"], 0)
        self.assertEqual(chains[0]["receptions"], 2)
        self.assertEqual((chains[0]["start"], chains[0]["end"]), (0, 100))


class TestReproducibility(unittest.TestCase):

    def test_rf_net_is_not_chaotic(self):
        # a tiny change of the training data (another CPU / BLAS) must stay tiny in the output
        from hmm_v0.nn import RandomFeatureNet
        rng = np.random.default_rng(0)
        X = rng.normal(size=(3000, 20))
        y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 1)
        cw = np.ones(4)
        outs = []
        for eps in (0.0, 1e-9):
            net = RandomFeatureNet(20, 64, 27)
            Xp = X + eps * rng.normal(size=X.shape)
            net.mu, net.sd = Xp.mean(axis=0), Xp.std(axis=0)
            Phi = net.hidden(Xp, set_scale=True)
            net.fit(Phi, y, cw, 1e-4, 200, log=lambda m: None)
            outs.append(net.forward(X))
        self.assertLess(float(np.abs(outs[0] - outs[1]).max()), 1e-6)

    def test_exact_matmul_is_order_independent(self):
        from hmm_v0 import exact
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(50, 300)), rng.normal(size=(300, 7))
        perm = rng.permutation(300)                      # another summation order
        np.testing.assert_array_equal(exact.qmatmul(a, b), exact.qmatmul(a[:, perm], b[perm]))
        x = np.linspace(-30, 0, 1001)
        self.assertLess(float(np.max(np.abs(exact.det_exp(x) / np.exp(x) - 1))), 1e-14)


class TestSegmentC(unittest.TestCase):

    def test_change_only_at_candidates_and_penalty(self):
        from hmm_v0.segment_c import assign
        ev = np.zeros(300)
        ev[:100] = 1.0                    # home evidence
        ev[130:300] = -1.0                # away evidence from frame 130
        cands = np.array([50, 120, 200])  # possession may change only here
        out, m = assign(0, 300, cands, ev, FPS, lam=1.0)
        out = "".join(out)
        self.assertEqual(m, 4)
        self.assertEqual(out[:120], "H" * 120)       # change placed at the candidate 120
        self.assertEqual(out[120:], "A" * 180)

    def test_short_touch_does_not_buy_a_possession(self):
        from hmm_v0.segment_c import assign
        ev = np.ones(300)
        ev[100:110] = -1.0                # 0.4 s of away contact inside a home possession
        cands = np.array([100, 110])
        out, _ = assign(0, 300, cands, ev, FPS, lam=1.0)   # 0.8 s of evidence < 2 changes
        self.assertEqual(set(out), {"H"})
        out, _ = assign(0, 300, cands, ev, FPS, lam=0.1)   # cheap changes: it becomes A
        out = "".join(out)
        self.assertEqual(out[100:110], "A" * 10)

    def test_dead_cleaning(self):
        from hmm_v0.segment_c import _clean_dead
        d = np.zeros(400, dtype=bool)
        d[50:120] = True; d[130:200] = True   # two D spells, 0.4 s apart -> one stoppage
        d[300:330] = True                      # 1.2 s of D -> too short, dropped
        half = np.ones(400, dtype=np.int8)
        cfg = {"close_gap_s": 1.0, "min_dead_s": 2.0, "min_alive_s": 1.0}
        out = _clean_dead(d, half, FPS, cfg)
        self.assertTrue(out[50:200].all())
        self.assertFalse(out[300:330].any())


class _MatchT(object):
    def __init__(self, n):
        self.fps, self.n = FPS, n
        self.half = np.ones(n, dtype=np.int8)


class TestTurnover(unittest.TestCase):

    def ep(self, team, start, end):
        return {"team": team, "start": start, "end": end, "slot": team * 11, "z": np.full(end - start, 9.0),
                "dur_s": (end - start) / FPS, "x0": 0, "y0": 0, "x1": 0, "y1": 0}

    def fire(self, eps, n=600):
        from hmm_v0.rules import build_rules
        rule = build_rules(CFG.rules, CFG.params, only=["T-INTERCEPT"])[0]
        return rule.fires(_MatchT(n), None, {"control_episodes": eps})

    def test_intercept_confirmed_by_control_or_pass(self):
        # H plays, A intercepts with a short touch and passes to a team-mate 2 s later: change
        out = self.fire([self.ep(0, 0, 50), self.ep(1, 80, 90), self.ep(1, 140, 170)])
        changes = [e for e in out.episodes if "start" in e]
        self.assertEqual([(e["start"], e["label"]) for e in changes], [(80, "A")])

    def test_touch_lost_again_is_not_a_change(self):
        # A touches the ball (deflection) and H has it again: no turnover (CHALLENGE.md)
        out = self.fire([self.ep(0, 0, 50), self.ep(1, 80, 88), self.ep(0, 100, 160)])
        self.assertEqual([e for e in out.episodes if "start" in e], [])


class _MatchV3(object):
    """Minimal match for the v3 helpers: 2 players per team, pitch 105 x 68."""
    def __init__(self, n=1500):
        self.fps, self.n = FPS, n
        self.half = np.ones(n, dtype=np.int8)
        self.graded = np.ones(n, dtype=bool)
        self.team_of_slot = np.array([0, 0, 1, 1])
        self.px = np.tile(np.array([3.0, 40.0, 10.0, 60.0], np.float32), (n, 1))
        self.py = np.full((n, 4), 34.0, np.float32)
        self.ball_visible = np.zeros(n, dtype=bool)
        self.ball_speed = np.zeros(n)
        self.smart_tagging = []


class _PitchV3(object):
    length, width, box_depth, box_half_width = 105.0, 68.0, 16.5, 20.16


def _ep(team, start, end, slot, x0=50.0, y0=34.0):
    return {"team": team, "start": start, "end": end, "slot": slot, "dur_s": (end - start) / FPS,
            "z": np.full(end - start, 9.0), "x0": x0, "y0": y0, "x1": x0, "y1": y0}


class TestSegmentV3(unittest.TestCase):

    def test_confirmed_gains_follow_challenge_md(self):
        from hmm_v0.segment_c import confirmed_gains
        m = _MatchV3()
        eps = [_ep(0, 0, 50, 1),          # home keeps it 2 s: control
               _ep(1, 60, 65, 2),         # away touch 0.2 s, then home again: NOT a change
               _ep(0, 80, 90, 1),         # home 0.4 s, team-mate 1 s later: completed pass
               _ep(0, 115, 120, 0),       # home, then away controls
               _ep(1, 130, 180, 3)]       # away keeps it 2 s: change
        gains = confirmed_gains(eps, m, CFG.params)
        self.assertEqual(gains, [(0, 0), (80, 0), (130, 1)])

    def test_goalkeeper_window_until_next_touch(self):
        from hmm_v0.segment_c import goalkeeper_windows
        m = _MatchV3()
        eps = [_ep(0, 100, 150, 0, x0=5.0),     # home keeper (deepest, in his box) holds 2 s
               _ep(1, 110, 140, 2, x0=6.0),     # away forward pressing him: ignored
               _ep(1, 260, 300, 3)]             # next touch after the clearance
        ctx = {"control_episodes": eps, "attack_dir": {(0, 1): 1, (1, 1): -1}}
        wins = goalkeeper_windows(m, _PitchV3(), ctx, {"merge_s": 2.0, "min_hold_s": 1.0})
        self.assertEqual(wins, [(100, 260, 0)])
        # a 0.4 s touch at his feet is not a hold
        eps[0] = _ep(0, 100, 110, 0, x0=5.0)
        wins = goalkeeper_windows(m, _PitchV3(), ctx, {"merge_s": 0.0, "min_hold_s": 1.0})
        self.assertEqual(wins, [])

    def test_smart_tagging_dead_ball_with_ball_unseen(self):
        from hmm_v0.segment_c import st_dead
        m = _MatchV3()
        m.smart_tagging = [{"code": "SET PIECES", "type": "throw-in", "start_frame": 500,
                            "end_frame": 1375}]
        ctx = {"control_episodes": [_ep(0, 300, 350, 1),    # last action before the tag
                                    _ep(1, 700, 760, 2)]}   # first contact after the restart
        cfg = {"types": ["throw-in"], "pre_s": 3.0, "lead_max_s": 15.0, "default_lead_s": 5.0,
               "kick_speed": 8.0, "back_to_last_action_s": 20}
        mask, spans = st_dead(m, ctx, cfg)
        self.assertEqual(spans, [(350, 700)])            # from the last action to the restart
        ctx["control_episodes"] = []                      # nothing seen at all
        mask, spans = st_dead(m, ctx, cfg)
        self.assertEqual(spans, [(500 - 75, 500 + 125)])  # tag - pre_s .. tag + default_lead_s


if __name__ == "__main__":
    unittest.main()
