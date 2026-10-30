"""Hidden Markov Model of the MOTORE: Bakis-expanded states, time-varying transitions, Viterbi.

Hidden macro-states (hmm.yaml): H, A, D, XH, XA. XH / XA = "the team still has the ball, the
tracker cannot see it": the X of the NN is a wildcard that REMEMBERS who had the ball, so an
occlusion never cuts a possession in two (spellF1 counts H H X X H H as one possession only if
the X frames are filled with H -- here they are, by construction).

Bakis (left-to-right) topology. A strict left-to-right model over a whole half is impossible
(possession goes back and forth), so the topology is applied INSIDE each macro-state, as in a
word model: macro-state m is a chain of n_m = round(min_duration_s[m] * fps) sub-states,

    sub_i -> sub_i     (inner_self_loop)        sub_i -> sub_{i+1}   (the rest)
    sub_last -> sub_last  (self-loop)           sub_last -> sub_0 of another macro (exits)

with no backward jump. The only way out of a macro-state is from its last sub-state, so every
visit lasts at least n_m frames: the decoder cannot produce a 1-frame possession or a 1-frame
dead ball. All sub-states of m share the emission of m.

Transitions between macro-states are given as mean times (hmm.yaml), p = 1 / (t * fps) per
frame. They are TIME-VARYING, through three gates computed from the rules (engine.py):

    restart   exits from D multiplied by `restart.boost` at a detected restart:
              D -> H/A "requires a throw-in or a kick-off"
    open_H    a change of team TO home (A/XA -> H/XH) is possible at its base rate only where
    open_A    a rule says the new team took the ball (start of a control, a pass chain, a
              TURNOVER, a restart); elsewhere it is divided by `team_change.closed_factor`.
              The NN alone can no longer flip the team (MOTORE.md: only anchors change state).

The 8 combinations of the gates are precomputed; `regime[t]` = 4*restart + 2*open_H + open_A.
"""

import numpy as np

NEG = -1e18


class BakisHMM(object):

    def __init__(self, cfg_hmm, fps):
        self.macros = list(cfg_hmm["states"])
        M = len(self.macros)
        self.index = {m: i for i, m in enumerate(self.macros)}
        self.n_sub = [max(1, int(round(cfg_hmm["min_duration_s"][m] * fps))) for m in self.macros]
        self.offset = np.concatenate(([0], np.cumsum(self.n_sub)[:-1])).astype(int)
        self.S = int(sum(self.n_sub))
        self.final = self.offset + np.array(self.n_sub) - 1
        self.macro_of = np.repeat(np.arange(M), self.n_sub)
        self.sub_of = np.concatenate([np.arange(k) for k in self.n_sub])

        rho = float(cfg_hmm.get("inner_self_loop", 0.0))
        self.log_rho = np.log(rho) if rho > 0 else NEG
        self.log_fwd = np.log(1.0 - rho) if rho < 1 else NEG

        # macro exit probabilities per frame
        P = np.zeros((M, M))
        for a, row in cfg_hmm["mean_time_s"].items():
            for b, t in row.items():
                P[self.index[a], self.index[b]] = 1.0 / (float(t) * fps)
        self.P = P
        self.d = self.index.get("D")
        self.boost = float(cfg_hmm["restart"]["boost"])
        tc = cfg_hmm.get("team_change", {}) or {}
        self.team_gate = bool(tc.get("gate", False))
        self.closed = float(tc.get("closed_factor", 1.0))
        team_of = cfg_hmm["output_label"]
        self.team_states = {t: [self.index[m] for m in self.macros if team_of[m] == t]
                            for t in ("H", "A")}
        self.grid = cfg_hmm.get("log_grid_bits")
        self.regimes = [self._log_exits(self._regime_P(r)) for r in range(8)]
        self.logA = self.regimes[3]                      # no restart, both gates open
        self.logA_boost = self.regimes[7]

        init = np.array([cfg_hmm["initial"].get(m, 0.0) for m in self.macros], dtype=float)
        self.log_init = np.log(np.maximum(init / init.sum(), 1e-300))

    def _regime_P(self, r):
        restart, open_h, open_a = bool(r & 4), bool(r & 2), bool(r & 1)
        P = self.P.copy()
        if restart and self.d is not None:
            P[self.d] *= self.boost
        if self.team_gate:
            for team, is_open in (("H", open_h), ("A", open_a)):
                if is_open:
                    continue
                other = "A" if team == "H" else "H"
                for a in self.team_states[other]:
                    for b in self.team_states[team]:
                        P[a, b] /= self.closed
        return P

    def _log_exits(self, P):
        stay = 1.0 - P.sum(axis=1)
        if np.any(stay <= 0):
            raise ValueError("hmm.mean_time_s: exits sum to >= 1 per frame; use longer times")
        A = P.copy()
        np.fill_diagonal(A, stay)
        with np.errstate(divide="ignore"):
            logA = np.where(A > 0, np.log(A), NEG)
        if self.grid:                                    # exact sums in Viterbi (exact.py)
            s = np.ldexp(1.0, int(self.grid))
            logA = np.where(A > 0, np.rint(logA * s) / s, NEG)
        return logA

    def viterbi(self, log_b, boost_mask=None, regime=None):
        """log_b: (T, M) macro emission log-likelihoods. regime: (T,) in 0..7 (see module doc);
        default: both team gates open, restart where boost_mask. Returns macro path, score."""
        T, M = log_b.shape
        S = self.S
        if T == 0:
            return np.zeros(0, dtype=int), 0.0
        if regime is None:
            boost_mask = np.zeros(T, dtype=bool) if boost_mask is None else boost_mask
            regime = np.where(boost_mask, 7, 3)
        mo = self.macro_of
        final, entry = self.final, self.offset
        has_fwd = self.sub_of > 0
        prev_idx = np.arange(S) - 1

        def stay_vec(logA):
            v = np.full(S, self.log_rho)
            v[final] = np.diag(logA)
            return v

        offdiag = ~np.eye(M, dtype=bool)
        stays = [stay_vec(A) for A in self.regimes]
        offs = [np.where(offdiag, A, NEG) for A in self.regimes]

        delta = np.full(S, NEG)
        delta[final] = self.log_init + log_b[0]          # a half starts in a "mature" state
        back = np.zeros((T, S), dtype=np.int32)
        back[0] = np.arange(S)
        ar = np.arange(S)
        for t in range(1, T):
            r = regime[t]
            stay = delta + stays[r]
            best = stay
            arg = ar.copy()
            fwd = np.full(S, NEG)
            fwd[has_fwd] = delta[prev_idx[has_fwd]] + self.log_fwd
            take = fwd > best
            best = np.where(take, fwd, best)
            arg = np.where(take, prev_idx, arg)
            cand = delta[final][:, None] + offs[r]                  # (from macro, to macro)
            src = cand.argmax(axis=0)
            val = cand[src, np.arange(M)]
            take = val > best[entry]
            best[entry] = np.where(take, val, best[entry])
            arg[entry] = np.where(take, final[src], arg[entry])
            delta = best + log_b[t, mo]
            back[t] = arg
        s = int(np.argmax(delta))
        score = float(delta[s])
        path = np.empty(T, dtype=int)
        for t in range(T - 1, -1, -1):
            path[t] = s
            s = back[t, s]
        return mo[path], score
