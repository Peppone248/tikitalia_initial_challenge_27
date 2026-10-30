"""Glue: configuration -> Match -> shared context -> rule outputs."""

from .config import Params, load_params
from .events import attack_directions, kickoffs
from .features import compute_features
from .geometry import Pitch
from .preprocessing import load_match
from .rules import build_rules


def setup(config_dir=None):
    return Pitch.from_config(config_dir), Params(load_params(config_dir))


def build_context(match, pitch, params):
    """Events and features every rule may read. Computed once.

    Order matters: the team-relative features need the attack direction, which needs the
    kick-offs.
    """
    ko = kickoffs(match, pitch, params)
    attack_dir, method = attack_directions(match, pitch, params, ko)
    features = compute_features(match, pitch, params, attack_dir)
    return {"kickoffs": ko, "attack_dir": attack_dir, "attack_method": method,
            "features": features}


def run(only=None, game="BAR-ATH", data_dir=None, cache_path=None, config_dir=None):
    pitch, params = setup(config_dir)
    match = load_match(pitch, params, game, data_dir, cache_path)
    context = build_context(match, pitch, params)
    rules = build_rules(params, only, config_dir)
    outputs = {rule.id: rule.fires(match, pitch, context) for rule in rules}
    return match, pitch, params, context, rules, outputs
