"""Rule interface. One class per rule, grouped in one module per category (dead, possession, ...).

A rule reads a prepared Match (+ Pitch + context) and returns a RuleOutput:
  label     per frame: 'H' | 'A' | 'D' | '' (abstain)
  strength  per frame in [0, 1] -- how clear THIS activation is (z-score of the margin, 4.1)
  episodes  one dict per activation episode, for the report and the technical document

The quality q of the rule (prior_q, later estimated) is NOT the rule's business: the engine
multiplies q * strength.
"""

import numpy as np

KINDS = ("ANCHOR", "SOFT", "RESTART", "TURNOVER")
SCOPES = ("OPEN_PLAY", "DEAD", "ANY")
ABSTAIN = ""


class RuleOutput(object):

    def __init__(self, rule_id, n):
        self.rule_id = rule_id
        self.label = np.full(n, ABSTAIN, dtype="<U1")
        self.strength = np.zeros(n)
        self.episodes = []

    def emit(self, start, end, label, strength, **info):
        """Label frames [start, end) and record the episode. Per-frame strength may be an array."""
        self.label[start:end] = label
        self.strength[start:end] = np.maximum(self.strength[start:end], strength)
        episode = {"rule": self.rule_id, "start": int(start), "end": int(end), "label": label,
                   "strength": float(np.max(strength)) if np.size(strength) else 0.0}
        episode.update(info)
        self.episodes.append(episode)
        return episode

    def record(self, **info):
        """An episode that labels no frame (e.g. a rejected candidate) -- kept for the report."""
        episode = {"rule": self.rule_id, "label": ABSTAIN, "strength": 0.0}
        episode.update(info)
        self.episodes.append(episode)
        return episode

    @property
    def fired(self):
        return self.label != ABSTAIN


class Rule(object):
    """Base class. Subclasses set `emits` and implement fires()."""

    emits = ()

    def __init__(self, rule_id, spec, params):
        self.id = rule_id
        self.kind = spec["kind"]
        self.scope = spec["scope"]
        self.inputs = tuple(spec.get("inputs", ()))
        self.prior_q = float(spec["prior_q"])
        self.param_names = tuple(spec.get("params", ()))
        self._params = params
        if self.kind not in KINDS or self.scope not in SCOPES:
            raise ValueError("{}: bad kind/scope {}/{}".format(rule_id, self.kind, self.scope))

    @property
    def p(self):
        """Parameters, restricted to the ones declared in rules.yaml."""
        return _Restricted(self._params, self.param_names, self.id)

    def fires(self, match, pitch, context):
        raise NotImplementedError

    def describe(self):
        return (self.__doc__ or "").strip()


class DeadRule(Rule):
    emits = ("D",)


class _Restricted(object):
    """A rule may only read the parameters it declares: undeclared reads are a bug."""

    def __init__(self, params, allowed, rule_id):
        self._params, self._allowed, self._rule_id = params, set(allowed), rule_id

    def __getattr__(self, name):
        if name not in self._allowed:
            raise AttributeError("{} reads {} but does not declare it in rules.yaml".format(
                self._rule_id, name))
        return getattr(self._params, name)
