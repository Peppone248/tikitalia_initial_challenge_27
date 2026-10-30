"""Rule output with a defined behaviour on overlaps (STATO_LAVORI 4.1, `emit()` item).

The team RuleOutput overwrites the label and keeps the max strength, so an overlap of two
different labels inside ONE rule silently mixes them. Here, on frames already labelled
differently, the stronger activation wins (ties: the earlier one) and the overlap is counted.
Labels: 'H' | 'A' | 'D' | 'X' | '' (abstain). 'X' = "unobservable", used only by X-NOBALL.
"""

import numpy as np

from tikitalia.rules.base import Rule, RuleOutput, ABSTAIN  # noqa: F401

LABEL_OF_TEAM = ("H", "A")


class SafeOutput(RuleOutput):

    def __init__(self, rule_id, n):
        super(SafeOutput, self).__init__(rule_id, n)
        self.overlap_frames = 0

    def emit(self, start, end, label, strength, **info):
        start, end = int(start), int(end)
        if end <= start:
            return None
        strength = np.broadcast_to(np.asarray(strength, dtype=float), (end - start,)).copy()
        cur_l = self.label[start:end]
        cur_s = self.strength[start:end]
        other = (cur_l != ABSTAIN) & (cur_l != label)
        self.overlap_frames += int(other.sum())
        take = ~other | (strength > cur_s)
        cur_l[take] = label
        cur_s[take] = np.where(other[take], strength[take], np.maximum(cur_s[take], strength[take]))
        episode = {"rule": self.rule_id, "start": start, "end": end, "label": label,
                   "strength": float(strength.max()) if strength.size else 0.0}
        episode.update(info)
        self.episodes.append(episode)
        return episode


class HmmRule(Rule):
    """Base of the rules of HMM_first_imp. `options` = free per-rule settings from rules_hmm.yaml
    (lists of Smart Tagging codes, scales): not thresholds, those stay in params."""

    def __init__(self, rule_id, spec, params):
        super(HmmRule, self).__init__(rule_id, spec, params)
        self.options = spec.get("options", {}) or {}

    def new_output(self, match):
        return SafeOutput(self.id, match.n)
