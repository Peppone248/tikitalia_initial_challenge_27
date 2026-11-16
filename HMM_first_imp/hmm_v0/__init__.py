"""HMM_first_imp -- first end-to-end pipeline of the Initial Challenge (v0, HMM engine).

    raw data -> Match -> features -> events -> rules -> MOTORE (NN + Bakis HMM) -> CSV

Reuses the team package `tikitalia` (../src) for Match, features, kick-offs, D-OUT and D-GOAL,
and adds on top of it, without modifying it:
    noise.py          MAD noise per feature, strength z / sigma (NOMENCLATURA 2)
    events.py         control_episode, restart_kick, pass chains
    rules/            P-DRIB, P-FREE, P-PASS, P-SHAPE, S-PHASE, D-KO, D-HOLD, D-WHISTLE, D-GAP,
                      D-SP, X-NOBALL, R-TOUCH, R-KO
    votes.py          w = q * s per rule -> log-odds evidence per frame
    frame_features.py per-frame vector + context window (t-k .. t+k) for the NN
    pseudo_labels.py  NN targets from the rules only (never from any ground truth)
    nn.py             light MLP in numpy -> P(H, A, D, X | frame)
    hmm.py            Bakis (left-to-right) expanded HMM, time-varying transitions, Viterbi
    engine.py         the MOTORE: emissions = NN + rule evidence, Viterbi per half
    submission.py     CSV writer
All parameters live in config/*.yaml.
"""

__version__ = "0.1.0"
