import numpy as np
import pandas as pd

from src.features.build_features import add_model_features, build_features
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