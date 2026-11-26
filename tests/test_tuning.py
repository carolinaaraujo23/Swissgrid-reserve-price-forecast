import numpy as np
import pandas as pd
import pytest

from src.models.lgbm import PARAMS
from src.models.tuning import bidding_summary, make_factory, simulate_bids, summarise


def test_factory_overrides_one_parameter_and_keeps_the_rest():
    before = dict(PARAMS)
    model = make_factory(num_leaves=7)(0.1, 100)
    params = model.get_params()
    assert params["num_leaves"] == 7
    assert params["alpha"] == 0.1 and params["n_estimators"] == 100
    assert params["min_child_samples"] == PARAMS["min_child_samples"]
    assert PARAMS == before                        # PARAMS itself is not modified


def test_factory_rejects_unknown_parameters():
    with pytest.raises(ValueError):
        make_factory(num_leafs=7)                  # typo


def make_case():
    auctions = pd.DataFrame({
        "auction_id": ["A", "B", "C", "D"],
        "freq": "daily", "direction": "upwards",
        "gc": pd.to_datetime(["2023-01-01", "2023-01-02", "2024-01-01", "2024-01-02"], utc=True),
        "marginal": [100.0, 200.0, np.nan, 50.0],   # C is a no-buy auction
    })
    contract = pd.DataFrame({
        "auction_id": ["A", "B", "C", "D"],
        "marginal_price_pred": [90.0, 250.0, 80.0, 40.0],
        "marginal_price_p10": [60.0, 150.0, 50.0, 30.0],
    })
    return auctions, contract


def test_pay_as_bid_rules():
    auctions, contract = make_case()
    sim = simulate_bids(contract, auctions, contract["marginal_price_pred"]).set_index("auction_id")
    assert sim.loc["A", "accepted"] and sim.loc["A", "revenue"] == 90       # 90 <= 100: paid its bid
    assert not sim.loc["B", "accepted"] and sim.loc["B", "revenue"] == 0    # 250 > 200: rejected
    assert not sim.loc["C", "accepted"] and sim.loc["C", "perfect"] == 0    # no-buy: nothing to win
    assert sim.loc["D", "revenue"] == 40


def test_bidding_summary_share_and_mean_per_year():
    auctions, contract = make_case()
    s = bidding_summary({"m": contract}, auctions).set_index("strategy")
    # bid p50: revenue 90 + 40 = 130 over 2 years; perfect 100 + 200 + 50 = 350
    assert np.isclose(s.loc["bid p50", "revenue"], 130 / 2)
    assert np.isclose(s.loc["bid p50", "share"], 130 / 350)
    assert np.isclose(s.loc["bid p50", "accepted"], 2 / 4)
    # bid p10: 60 + 150 + 30 = 240, all three priced auctions accepted
    assert np.isclose(s.loc["bid p10", "share"], 240 / 350)


def test_summarise_counts_groups_better_than_default():
    rows = []
    for config, loss in [("default", 1.0), ("better", 0.9), ("mixed", None)]:
        for i, (freq, direction) in enumerate([("daily", "up"), ("daily", "down"), ("weekly", "up"), ("weekly", "down")]):
            value = loss if loss is not None else (0.8 if i < 2 else 1.2)
            rows.append(dict(config=config, freq=freq, direction=direction,
                             pinball_q10=value, pinball_q50=value, pinball_q90=value))
    t = summarise(pd.DataFrame(rows))
    assert t.index[0] == "better"
    assert t.loc["better", "groups_better"] == 4
    assert t.loc["mixed", "groups_better"] == 2