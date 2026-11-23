"""Tests of the previous-known-auction baseline (src/models/baseline.py).

Hand-made tables for the definition and the contract, real mFRR data for the
full run and the leakage check.
Run with: uv run pytest tests/test_baseline.py
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import PredictionValidator
from src.models.baseline import PreviousKnownAuctionBaseline, select_test_auctions

UP = "daily|upwards|08_12"


def make_auctions(rows):
    """rows: list of (auction_id, series, gate_closure_utc, marginal_price).

    marginal_price None = no-buy auction. Columns follow src.data.data.build_auctions.
    """
    df = pd.DataFrame(rows, columns=["auction_id", "series", "gc", "marginal"])
    df["gc"] = pd.to_datetime(df["gc"], utc=True)
    df["marginal"] = df["marginal"].astype(float)
    df["y"] = np.log10(df["marginal"])
    df["avail"] = df["gc"] + pd.Timedelta(hours=2)
    df["freq"] = df["series"].str.split("|").str[0]
    df["direction"] = df["series"].str.split("|").str[1]
    return df


def predict_ids(auctions, ids):
    """Fit on the whole table, predict the auctions `ids`; returns price by auction_id."""
    to_predict = auctions[auctions["auction_id"].isin(ids)]
    preds = PreviousKnownAuctionBaseline().fit(auctions).predict(to_predict)
    return preds.set_index("auction_id")["marginal_price_pred"]


# ---------------------------------------------------------------------------
# Definition and contract, on hand-made tables
# ---------------------------------------------------------------------------

def test_forecast_is_latest_known_price_of_the_same_series():
    a = make_auctions([
        ("seed",       UP,                    "2024-12-20 13:30", 7.0),     # first of the series
        ("seed_other", "daily|upwards|12_16", "2024-12-20 13:30", 9.0),
        ("prev_week",  UP,                    "2025-01-03 13:30", 50.0),
        ("no_buy",     UP,                    "2025-01-06 13:30", None),    # skipped
        ("other_blk",  "daily|upwards|12_16", "2025-01-09 13:30", 999.0),   # other series
        ("monday",     UP,                    "2025-01-10 13:30", 10.0),    # known Fri 15:30
        ("tuesday",    UP,                    "2025-01-10 14:30", 1.0),     # closes Fri 14:30
    ])
    pred = predict_ids(a, ["monday", "tuesday"])
    assert pred["tuesday"] == pytest.approx(50.0)   # not monday (1 h < 2 h), not other_blk
    assert pred["monday"] == pytest.approx(50.0)    # the no-buy auction is skipped


def test_output_follows_the_prediction_contract():
    a = make_auctions([
        ("h1", UP, "2024-12-20 13:30", 10.0),
        ("t1", UP, "2025-01-10 13:30", 20.0),
        ("t2", UP, "2025-01-11 13:30", None),     # no-buy: still predicted
    ])
    model = PreviousKnownAuctionBaseline().fit(a)
    test = select_test_auctions(a)
    preds = model.predict(test)

    assert list(preds.columns) == ["auction_id", "marginal_price_pred", "model"]   # no y leaked
    assert sorted(preds["auction_id"]) == ["t1", "t2"]
    assert (preds["model"] == "baseline").all()
    PredictionValidator.validate(preds, expected_ids=test["auction_id"])


def test_predict_refuses_unusable_inputs():
    a = make_auctions([("h1", UP, "2024-12-20 13:30", 10.0), ("t1", UP, "2025-01-10 13:30", 20.0)])

    with pytest.raises(RuntimeError, match="fit"):
        PreviousKnownAuctionBaseline().predict(a)

    unknown = make_auctions([("never_seen", UP, "2025-02-01 13:30", 5.0)])
    with pytest.raises(ValueError, match="not in the history"):
        PreviousKnownAuctionBaseline().fit(a).predict(unknown)


def test_series_without_any_known_price_raises():
    a = make_auctions([
        ("first", UP, "2025-01-10 13:30", 10.0),                    # nothing known before
        ("other", "daily|downwards|08_12", "2025-01-01 13:30", 3.0),
    ])
    model = PreviousKnownAuctionBaseline().fit(a)
    with pytest.raises(ValueError, match="fallback"):
        model.predict(a[a["auction_id"] == "first"])


# ---------------------------------------------------------------------------
# Real mFRR data
# ---------------------------------------------------------------------------

def test_full_run_on_mfrr(mfrr_auctions):
    model = PreviousKnownAuctionBaseline().fit(mfrr_auctions)
    test = select_test_auctions(mfrr_auctions)
    preds = model.predict(test)
    metrics = model.evaluate(mfrr_auctions, preds)       # validates the file, then scores it

    assert len(preds) == len(test)
    assert set(zip(metrics["freq"], metrics["direction"])) == {
        ("daily", "upwards"), ("daily", "downwards"),
        ("weekly", "upwards"), ("weekly", "downwards"),
    }
    assert metrics["n"].sum() == test["y"].notna().sum()  # every auction with a price is scored
    assert (metrics["mae"] > 0).all() and (metrics["mae"] < 0.5).all()


@pytest.mark.parametrize("cutoff", ["2025-03-01", "2025-06-12 15:00"])
def test_changing_future_results_does_not_change_past_predictions(mfrr_auctions, cutoff):
    t0 = pd.Timestamp(cutoff, tz="UTC")
    test = select_test_auctions(mfrr_auctions)
    before = (PreviousKnownAuctionBaseline().fit(mfrr_auctions).predict(test)
              .set_index("auction_id")["marginal_price_pred"])

    perturbed = mfrr_auctions.copy()
    future = perturbed["avail"] > t0                      # results not published at t0
    perturbed.loc[future, "y"] = perturbed.loc[future, "y"] + 3
    after = (PreviousKnownAuctionBaseline().fit(perturbed).predict(test)
             .set_index("auction_id")["marginal_price_pred"])

    past = test.loc[test["gc"] <= t0, "auction_id"]
    later = test.loc[test["gc"] > t0 + pd.Timedelta(days=7), "auction_id"]
    pd.testing.assert_series_equal(before.loc[past], after.loc[past])
    assert not np.allclose(before.loc[later], after.loc[later])   # the perturbation is visible
