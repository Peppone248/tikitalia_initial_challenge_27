import pandas as pd

from seqmodel.report import criterion, render


def test_criterion():
    good = [0.03, 0.02, 0.01, 0.02, 0.0, -0.01, 0.02]
    assert criterion({1: good, 2: good}) is True
    assert criterion({1: good, 2: [0.1, 0.1, 0.1, 0.1, -0.5, -0.5, -0.5]}) is False   # 4 positive
    assert criterion({1: [0.0] * 7, 2: [0.0] * 7}) is False                          # mean 0
    assert criterion({1: [0.01] * 7, 2: [0.005] * 7}) is False                       # mean .0075 < .01
    assert criterion({1: [-0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                      2: [0.05] * 7}) is True            # mean exactly 0.0 + ... above 0.01? yes
    assert criterion({1: [0.2, 0.2, 0.2, 0.2, 0.2, -2, -2], 2: good}) is False        # mean < .01
    assert criterion({}) is False


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
