"""Rule registry: classes register themselves, rules.yaml decides which run and with what."""

import importlib

from ..config import load_rules

_CLASSES = {}


def register(cls):
    """Class decorator: makes `module.ClassName` resolvable from rules.yaml."""
    module = cls.__module__.rsplit(".", 1)[-1]
    _CLASSES["{}.{}".format(module, cls.__name__)] = cls
    return cls


def build_rules(params, only=None, config_dir=None):
    """Instantiate the enabled rules, in file order. `only`: restrict to these ids."""
    specs = load_rules(config_dir)
    rules = []
    for rule_id, spec in specs.items():
        if only and rule_id not in only:
            continue
        if not spec.get("enabled", True) and not only:
            continue
        module, _ = spec["class"].split(".")
        importlib.import_module("tikitalia.rules." + module)
        if spec["class"] not in _CLASSES:
            raise KeyError("{}: class {} is not registered".format(rule_id, spec["class"]))
        missing = [name for name in spec.get("params", ()) if name not in params.names()]
        if missing:
            raise KeyError("{}: params {} are not in params.yaml".format(rule_id, missing))
        rules.append(_CLASSES[spec["class"]](rule_id, spec, params))
    return rules
