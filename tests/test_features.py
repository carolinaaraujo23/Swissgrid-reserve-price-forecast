import numpy as np
import pandas as pd

from src.features.build_features import build_features
from src.models.ridge import walk_forward_ridge

def make_auctions(rows):
    """rows is a list of (auction_id, directiosn, gate_closure_utc, marginal_price)"""
    df = pd.DataFrame(rows, columns=["auction_id", "direction", "gate_closure_time_utc", "marginal_price"])
    df["gate_closure_time_utc"] = pd.to_datetime(df["gate_closure_time_utc"], utc=True)
    df["available_time_utc"] = df["gate_closure_time_utc"] + pd.Timedelta(hours=2)
    df["log10_price"] = np.log10(df["marginal_price"])
    return df

def perturbe_future(auctions, t):
    perturbed = auctions.copy()
    future = perturbed["available_time_utc"] > t
    perturbed.loc[future, "log10_price"] = perturbed.loc[future, "log10_price"] * 1000
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

    past = preds["gate_closure_time_utc"] <= t
    pd.testing.assert_series_equal(preds.loc[past, "pred_ridge"], perturbed_preds.loc[past, "pred_ridge"])

    assert not preds.loc[~past, "pred_ridge"].equals(perturbed_preds.loc[~past, "pred_ridge"])

if __name__ == "__main__":
    test_available_exactly_at_gate_closure_time()
    test_available_one_second_after_gate_closure_time()
    test_direction_selection()
    test_features_do_not_change_when_future_is_perturbed()
    test_predictions_do_not_change_when_future_is_perturbed()

