"""changepoint_model (package `cpm`): possession labels H / A / D from tracking, built around the
detection of the *changes* of possession rather than the state only.

Layout
    match.py      one in-memory representation of a match, whatever the source
    loaders/      IDSSE .npz, Metrica sample data (CSV and EPTS), FIFA/EPTS raw text (the ATD)
    labels.py     labels from event data (Metrica), the same rule for every match
    degrade.py    a priori degradation of clean tracking (ball hidden in stoppages, random gaps,
                  broken player tracks), seeded and deterministic
    features.py   per-frame features, split into even and odd under the team swap
    models.py     symmetric logistic team model, boosted state and boundary classifiers
    seq.py        bidirectional GRU for P(home) (torch to train, numpy to predict)
    pipeline.py   prepare -> predict -> decode
    decode.py     3-state HMM with minimum durations whose transition hazards follow the boundary
                  classifiers (time-varying Viterbi)
    kickoff.py    Law 8: no play before each kick-off (Smart Tagging kick-off tag)
    whistle.py    Law 5: whistle blasts from the audio, play stops at the whistle
    evaluate.py   the official scorer applied to a prediction
"""

__version__ = "4.2"
