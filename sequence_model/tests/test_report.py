import pandas as pd

from seqmodel.report import criterion, render


def test_criterion():
    good = [0.03, 0.02, 0.01, 0.02, 0.0, -0.01, 0.02]            # mean 0.0128, 5 positive
    assert criterion({1: good, 2: good}) is True
    assert criterion({}) is False
    assert criterion({1: good}) is False                          # one seed is not enough
    assert criterion({1: good, 2: good[:6]}) is False             # fewer than 7 paired matches
    # 4 positive + 1 zero, mean above 0.01: > 0 is required, not >= 0
    assert criterion({1: [0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0], 2: good}) is False
    # mean exactly 0.01 in each seed with 5 positives passes
    assert criterion({1: [0.01] * 7, 2: [0.01] * 7}) is True
    # pooled mean 0.015 would pass, but seed 2 has mean 0.0
    assert criterion({1: [0.03] * 7, 2: [0.01, 0.01, 0.01, 0.01, 0.01, -0.025, -0.025]}) is False
    assert criterion({1: [0.01] * 7, 2: [0.005] * 7}) is False  # seed 2 mean below threshold
    assert criterion({1: [0.2] * 5 + [-2, -2], 2: good}) is False  # seed 1 mean negative


def _df(model, seed, vals):
    return pd.DataFrame([dict(match=f"m{i}", spellF1=v, macroF1=v, n_pred_poss=10, n_true_poss=12,
                              transF1_25=v, model=model, seed=seed) for i, v in enumerate(vals)])


def test_render_verdict_and_deltas():
    fr = [0.4] * 7
    ok = pd.concat([_df("frame", s, fr) for s in (1, 2)] +
                   [_df("tcn_change", s, [0.5] * 7).assign(beta=5.0, w=1.0) for s in (1, 2)])
    assert render(ok).rstrip().splitlines()[-1] == "criterion: SODDISFATTO"
    bad = pd.concat([_df("frame", s, fr) for s in (1, 2)] +
                    [_df("tcn_change", s, [0.3] * 7).assign(beta=5.0, w=1.0) for s in (1, 2)])
    text = render(bad)
    assert text.rstrip().splitlines()[-1] == "criterion: NON SODDISFATTO"
    assert "in-sample" in text
    assert "0/7" in text and "7/7" in text            # better/worse counts


def test_render_single_seed_has_no_verdict():
    one = pd.concat([_df("frame", 1, [0.4] * 7), _df("tcn_change", 1, [0.9] * 7).assign(beta=5.0, w=1.0)])
    assert render(one).rstrip().splitlines()[-1] == "criterion: needs at least 2 seeds"
