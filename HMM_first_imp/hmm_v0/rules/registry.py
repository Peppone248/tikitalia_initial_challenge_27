"""Registry of HMM_first_imp: its own rule classes + the team ones (tikitalia.rules.dead).

rules_hmm.yaml names a class as `module.ClassName`:
    team.dead.BallOut     -> tikitalia.rules.dead.BallOut          (D-OUT, unchanged)
    possession.Dribble    -> hmm_v0.rules.possession.Dribble
"""

import importlib

_CLASSES = {}
MODULES = ("possession", "dead_extra", "restart", "noball", "turnover")


def register(cls):
    module = cls.__module__.rsplit(".", 1)[-1]
    _CLASSES["{}.{}".format(module, cls.__name__)] = cls
    return cls


def resolve(name):
    if name.startswith("team."):
        module, cls = name[len("team."):].rsplit(".", 1)
        return getattr(importlib.import_module("tikitalia.rules." + module), cls)
    for m in MODULES:
        importlib.import_module("hmm_v0.rules." + m)
    if name not in _CLASSES:
        raise KeyError("rule class {} is not registered".format(name))
    return _CLASSES[name]


def build_rules(specs, params, only=None):
    """Enabled rules in file order, grouped by `stage` (1 first). Undeclared params = error."""
    rules = []
    for rule_id, spec in specs.items():
        if only and rule_id not in only:
            continue
        if not spec.get("enabled", True) and not only:
            continue
        missing = [n for n in spec.get("params", ()) if n not in params.names()]
        if missing:
            raise KeyError("{}: params {} are not defined".format(rule_id, missing))
        rule = resolve(spec["class"])(rule_id, spec, params)
        rule.stage = int(spec.get("stage", 1))
        rule.options = getattr(rule, "options", spec.get("options", {}) or {})
        rules.append(rule)
    return sorted(rules, key=lambda r: r.stage)
