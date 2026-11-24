import numpy as np
import pandas as pd

from src.features.build_features import (
    LAG_FEATURES, add_lag_features, add_model_features, add_other_direction_features,
    add_weekly_features, build_features,
)
from src.models.ridge import model_feature_columns, walk_forward_ridge

def make_auctions(rows):
    df = pd.DataFrame(rows, columns=["auction_id", "direction", "gc", "marginal"])
    df["gc"] = pd.to_datetime(df["gc"], utc=True)
    df["avail"] = df["gc"] + pd.Timedelta(hours=2)
    df["y"] = np.log10(df["marginal"])
    df["freq"] = "weekly"
    df["series"] = df["freq"] + "|" + df["direction"] + "|week"
    return df

def perturbe_future(auctions, t):
    perturbed = auctions.copy()
    future = perturbed["avail"] > t
    perturbed.loc[future, "y"] = perturbed.loc[future, "y"] + 3
    return perturbed

def make_weekly_auctions(n_weeks=20):
    rows = []
    for w in range(n_weeks):
        gc = pd.Timestamp("2024-01-02 12:00") + pd.Timedelta(weeks=w)
        rows.append((f"W{w}_up", "upwards", gc, 100.0 + w))
        rows.append((f"W{w}_down", "downwards", gc, 300.0 + w))
    return make_auctions(rows)

def test_available_exactly_at_gate_closure_time():
    auctions = make_auctions([
        ("H", "upwards", "2024-03-01 10:00", 100.0),   
        ("A", "upwards", "2024-03-01 12:00", 1.0),    
    ])
    features = build_features(auctions).set_index("auction_id")
    assert features.loc["A", "h_auction_id"] == "H"


def test_available_one_second_after_gate_closure_time():
    auctions = make_auctions([
        ("H", "upwards", "2024-03-01 10:00:01", 100.0),  
        ("A", "upwards", "2024-03-01 12:00:00", 1.0),
    ])
    features = build_features(auctions).set_index("auction_id")
    assert pd.isna(features.loc["A", "h_auction_id"])

def test_direction_selection():
    auctions = make_auctions([
            ("H_up", "upwards", "2024-03-01 10:00:00", 100.0), 
            ("H_down", "downwards", "2024-03-08 10:00:00", 10.0), 
            ("A", "upwards", "2024-03-15 10:00:00", 1.0),
    ])
    features = build_features(auctions).set_index("auction_id")
    assert features.loc["A", "h_auction_id"] == "H_up"

def test_features_do_not_change_when_future_is_perturbed():
    auctions = make_auctions([
        ("H", "upwards",   "2024-03-01 10:00", 100.0),
        ("A", "upwards",   "2024-03-08 10:00", 200.0),
        ("B",  "upwards",   "2024-03-15 10:00", 1.0),    
        ("C",  "downwards", "2024-03-15 10:00", 5.0),    
        ("D",  "upwards",   "2024-03-22 10:00", 7.0), 
    ])   
    t = pd.Timestamp("2024-03-15 10:00", tz="UTC")
    lags = ["lag1", "lag2", "lag3"]
    features = build_features(auctions).set_index("auction_id")
    perturbed_features = build_features(perturbe_future(auctions, t)).set_index("auction_id")

    pd.testing.assert_series_equal(features.loc["B", lags], perturbed_features.loc["B", lags])

    assert not features.loc["D", lags].equals(perturbed_features.loc["D", lags])

def test_predictions_do_not_change_when_future_is_perturbed():
    auctions = make_weekly_auctions()                      
    t = pd.Timestamp("2024-03-26 12:00", tz="UTC")         

    kw = dict(start_test="2024-03-01", end_test="2024-05-31")   
    preds = walk_forward_ridge(build_features(auctions), **kw).set_index("auction_id")
    perturbed_preds = walk_forward_ridge(build_features(perturbe_future(auctions, t)), **kw).set_index("auction_id")

    past = preds["gc"] <= t
    pd.testing.assert_series_equal(preds.loc[past, "pred_ridge"], perturbed_preds.loc[past, "pred_ridge"])
    # the p10 / p90 come from training errors only: they must not change either
    pd.testing.assert_series_equal(preds.loc[past, "p10_ridge"], perturbed_preds.loc[past, "p10_ridge"])
    pd.testing.assert_series_equal(preds.loc[past, "p90_ridge"], perturbed_preds.loc[past, "p90_ridge"])

    assert not preds.loc[~past, "pred_ridge"].equals(perturbed_preds.loc[~past, "pred_ridge"])

def test_predictions_with_extra_features_do_not_change_when_future_is_perturbed():
    """Same check with the block / day-type one-hots as inputs and the delta target."""
    auctions = make_weekly_auctions()
    t = pd.Timestamp("2024-03-26 12:00", tz="UTC")
    kw = dict(start_test="2024-03-01", end_test="2024-05-31", target="delta")

    feats = add_model_features(build_features(auctions))
    cols = model_feature_columns(feats)
    assert any(c.startswith("block_") for c in cols)
    preds = walk_forward_ridge(feats, feature_cols=cols, **kw).set_index("auction_id")
    perturbed = add_model_features(build_features(perturbe_future(auctions, t)))
    perturbed_preds = walk_forward_ridge(perturbed, feature_cols=cols, **kw).set_index("auction_id")

    past = preds["gc"] <= t
    pd.testing.assert_series_equal(preds.loc[past, "pred_ridge"], perturbed_preds.loc[past, "pred_ridge"])
    assert not preds.loc[~past, "pred_ridge"].equals(perturbed_preds.loc[~past, "pred_ridge"])


if __name__ == "__main__":
    test_available_exactly_at_gate_closure_time()
    test_available_one_second_after_gate_closure_time()
    test_direction_selection()
    test_features_do_not_change_when_future_is_perturbed()
    test_predictions_do_not_change_when_future_is_perturbed()


def test_lag_features_values():
    auctions = make_auctions([
        ("H1", "upwards", "2024-03-01 10:00", 100.0),    # y = 2
        ("H2", "upwards", "2024-03-08 10:00", 1000.0),   # y = 3
        ("H3", "upwards", "2024-03-15 10:00", 10000.0),  # y = 4
        ("A",  "upwards", "2024-03-22 10:00", 1.0),
    ])
    f = add_lag_features(build_features(auctions)).set_index("auction_id")
    # A sees lag1 = 4 (H3), lag2 = 3 (H2), lag3 = 2 (H1)
    assert np.isclose(f.loc["A", "mom1"], 1.0)
    assert np.isclose(f.loc["A", "mom2"], 1.0)
    assert np.isclose(f.loc["A", "lag_mean"], 3.0)
    assert np.isclose(f.loc["A", "lag_std"], 1.0)
    # H2 only sees H1 (lag1 = 2, no lag2/lag3): differences are NaN, the mean uses what exists
    assert np.isnan(f.loc["H2", "mom1"])
    assert np.isclose(f.loc["H2", "lag_mean"], 2.0)


def test_lag_features_do_not_change_when_future_is_perturbed():
    auctions = make_weekly_auctions(n_weeks=12)
    t = pd.Timestamp("2024-02-13 12:00", tz="UTC")
    before = add_lag_features(build_features(auctions)).set_index("auction_id")
    after = add_lag_features(build_features(perturbe_future(auctions, t))).set_index("auction_id")
    past = before["gc"] <= t
    pd.testing.assert_frame_equal(before.loc[past, LAG_FEATURES], after.loc[past, LAG_FEATURES])
    assert not before.loc[~past, "lag1"].equals(after.loc[~past, "lag1"])


def with_ratio(auctions, ratios):
    out = auctions.copy()
    out["supply_ratio"] = ratios
    return out


def test_ratio_lag_uses_previous_auction_not_own():
    auctions = with_ratio(make_auctions([
        ("H", "upwards", "2024-03-01 10:00", 100.0),
        ("A", "upwards", "2024-03-08 10:00", 200.0),
    ]), [10.0, 1000.0])
    f = build_features(auctions, ratio_lags=1).set_index("auction_id")
    assert np.isclose(f.loc["A", "ratio_lag1"], 1.0)   # log10(10), from H
    assert np.isnan(f.loc["H", "ratio_lag1"])          # H's own ratio (1000) is not visible to H


def test_ratio_lag_same_sitting_not_visible():
    auctions = with_ratio(make_auctions([
        ("H_down", "downwards", "2024-03-01 10:00", 300.0),
        ("A_up",   "upwards",   "2024-03-08 10:00", 100.0),
        ("A_down", "downwards", "2024-03-08 10:00", 300.0),
    ]), [5.0, 50.0, 500.0])
    f = build_features(auctions, ratio_lags=1).set_index("auction_id")
    assert np.isclose(f.loc["A_down", "ratio_lag1"], np.log10(5.0))
    assert np.isnan(f.loc["A_up", "ratio_lag1"])


def test_ratio_lag_does_not_change_when_future_is_perturbed():
    auctions = make_weekly_auctions(n_weeks=12)
    auctions["supply_ratio"] = np.linspace(2, 8, len(auctions))
    t = pd.Timestamp("2024-02-13 12:00", tz="UTC")
    perturbed = auctions.copy()
    future = perturbed["avail"] > t
    perturbed.loc[future, "supply_ratio"] *= 100

    before = build_features(auctions, ratio_lags=1).set_index("auction_id")
    after = build_features(perturbed, ratio_lags=1).set_index("auction_id")
    past = before["gc"] <= t
    pd.testing.assert_series_equal(before.loc[past, "ratio_lag1"], after.loc[past, "ratio_lag1"])
    assert not before.loc[~past, "ratio_lag1"].equals(after.loc[~past, "ratio_lag1"])


def as_daily(auctions, ids, block="08_12"):
    """Turn some synthetic auctions into daily ones (block 08_12)."""
    out = auctions.copy()
    daily = out["auction_id"].isin(ids)
    out.loc[daily, "freq"] = "daily"
    out["series"] = out["freq"] + "|" + out["direction"] + "|" + out["freq"].map({"weekly": "week", "daily": block})
    return out


def test_other_direction_same_sitting_not_visible():
    auctions = make_auctions([
        ("H_down", "downwards", "2024-03-01 10:00", 100.0),   # previous week: visible
        ("A_up",   "upwards",   "2024-03-08 10:00", 10.0),
        ("A_down", "downwards", "2024-03-08 10:00", 1000.0),  # same sitting as A_up: not visible
    ])
    f = add_other_direction_features(build_features(auctions), auctions).set_index("auction_id")
    assert np.isclose(f.loc["A_up", "other_lag1"], 2.0)       # log10(100), from H_down
    assert np.isnan(f.loc["A_down", "other_lag1"])            # no earlier upwards auction


def test_weekly_feature_for_daily_auctions():
    auctions = as_daily(make_auctions([
        ("W1", "upwards",   "2024-03-01 10:00", 1000.0),    # weekly, visible
        ("W2", "upwards",   "2024-03-05 10:00", 10000.0),   # weekly, closes after D: not visible
        ("Wd", "downwards", "2024-03-01 10:00", 5.0),       # other direction: ignored
        ("D0", "upwards",   "2024-03-02 10:00", 10.0),      # daily
        ("D",  "upwards",   "2024-03-04 10:00", 20.0),      # daily
    ]), ids=["D0", "D"])
    f = add_weekly_features(build_features(auctions), auctions).set_index("auction_id")
    assert np.isclose(f.loc["D", "weekly_lag1"], 3.0)         # log10(1000), from W1
    assert np.isclose(f.loc["D", "weekly_gap"], 3.0 - 1.0)    # D's lag1 is D0 (log10(10) = 1)


def test_cross_series_features_do_not_change_when_future_is_perturbed():
    auctions = make_weekly_auctions(n_weeks=12)
    t = pd.Timestamp("2024-02-13 12:00", tz="UTC")
    cols = ["other_lag1", "other_gap", "weekly_lag1", "weekly_gap"]

    def run(a):
        f = add_weekly_features(add_other_direction_features(build_features(a), a), a)
        return f.set_index("auction_id")

    before, after = run(auctions), run(perturbe_future(auctions, t))
    past = before["gc"] <= t
    pd.testing.assert_frame_equal(before.loc[past, cols], after.loc[past, cols].loc[before.loc[past].index])
    assert not before.loc[~past, "other_lag1"].equals(after.loc[~past, "other_lag1"])