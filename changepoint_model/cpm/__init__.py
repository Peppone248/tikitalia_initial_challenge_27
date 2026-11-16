"""changepoint_model (package `cpm`): possession labels H / A / D from tracking, built around the
detection of the *changes* of possession rather than the state only.

Layout
    match.py      one in-memory representation of a match, whatever the source
    loaders/      IDSSE .npz, Metrica sample data (CSV and EPTS), FIFA/EPTS raw text (the ATD)
    labels.py     labels from event data (Metrica), the same rule for every match
    degrade.py    a priori degradation of clean tracking (ball hidden in stoppages, random gaps,
                  broken player tracks), seeded and deterministic
    features.py   per-frame features, split into even and odd under the team swap
    models.py     symmetric logistic frame classifier, boundary classifiers (numpy only)
    decode.py     3-state HMM with minimum durations whose transition hazards follow the boundary
                  classifiers (time-varying Viterbi)
    evaluate.py   the official scorer applied to a prediction, leave-one-match-out helpers
"""

__version__ = "0.1.0"
