import numpy as np
import pandas as pd

from src.features.build_features import build_features

def make_auctions(rows):
    """rows is a list of (auction_id, directiosn, gate_closure_utc, marginal_price)"""
    df = pd.DataFrame(rows, columns=["auction_id", "direction", "gate_closure_time_utc", "marginal_price"])
    df["gate_closure_time_utc"] = pd.to_datetime(df["gate_closure_time_utc"], utc=True)
    df["available_time_utc"] = df["gate_closure_time_utc"] + pd.Timedelta(hours=2)
    df["log10_price"] = np.log10(df["marginal_price"])
    return df

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

if __name__ == "__main__":
    test_available_exactly_at_gate_closure_time()
    test_available_one_second_after_gate_closure_time()
    test_direction_selection()

