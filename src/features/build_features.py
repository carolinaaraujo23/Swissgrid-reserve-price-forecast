import pandas as pd
from src.data.data import build_auctions
 
def build_history(auctions, n_lags=3):
    history = auctions.loc[auctions["y"].notna(), ["auction_id", "series", "avail", "y"]].copy()
    history = history.sort_values(["series", "avail"])
 
    for i in range(n_lags):
        history[f"lag{i + 1}"] = history.groupby("series")["y"].shift(i)
 
    return history.drop(columns="y")
 
 
def build_features(auctions, n_lags=3):
    history = build_history(auctions, n_lags)
    history = history.rename(columns={"auction_id": "h_auction_id", "avail": "h_avail"})
    targets = auctions.sort_values("gc")
    history = history.sort_values("h_avail")
 
    features = pd.merge_asof(
        targets,
        history,
        left_on="gc",
        right_on="h_avail",
        by="series",
        direction="backward",
        allow_exact_matches=True,
    )
 
    return features
 
 
if __name__ == "__main__":
    auctions = build_auctions(product="mFRR")
    auctions = auctions[~auctions["auction_id"].str.contains("KORR")]
    features = build_features(auctions)
    print(features[["auction_id", "series", "gc", "lag1", "lag2", "lag3", "h_auction_id", "h_avail"]].head(8))
    ok = features["h_avail"].isna() | (features["h_avail"] <= features["gc"])
    print("no future data:", ok.all())
    print("no self-reference:", (features["h_auction_id"] != features["auction_id"]).all())
 