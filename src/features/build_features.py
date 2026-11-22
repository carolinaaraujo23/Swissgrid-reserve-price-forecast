import pandas as pd

def build_history(auctions, n_lags=3):
    history = auctions.loc[auctions["marginal_price"].notna(),
                       ["auction_id", "direction", "available_time_utc", "log10_price"]].copy()
    history = history.sort_values(["direction", "available_time_utc"])       

    for i in range(n_lags):
        history[f"lag{i + 1}"] = history.groupby("direction")["log10_price"].shift(i)

    return history.drop(columns="log10_price")     

def build_features(auctions, n_lags=3):
    history = build_history(auctions, n_lags)
    history = history.rename(columns={"auction_id": "h_auction_id", "available_time_utc": "h_available_time_utc"})        
    targets = auctions.sort_values("gate_closure_time_utc")            
    history = history.sort_values("h_available_time_utc")             

    features = pd.merge_asof(
        targets,
        history,
        left_on="gate_closure_time_utc",
        right_on="h_available_time_utc",
        by="direction",
        direction="backward",
        allow_exact_matches=True,
    )

    return features


if __name__ == "__main__":
    from src.data.load import load_offers, build_auctions

    offers = load_offers("data/raw/offers", "mFRR", "weekly")
    auctions = build_auctions(offers)
    features = build_features(auctions)

    print(features[["auction_id", "gate_closure_time_utc", "lag1", "lag2", "lag3",
                    "h_auction_id", "h_available_time_utc"]].head(8))
    print("no future data:", (features["h_available_time_utc"] <= features["gate_closure_time_utc"]).all())
    print("no self-reference:", (features["h_auction_id"] != features["auction_id"]).all())