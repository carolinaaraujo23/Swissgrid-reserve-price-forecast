import numpy as np
import pandas as pd

LAGS = ["lag1", "lag2", "lag3"]
LAG_FEATURES = ["mom1", "mom2", "lag_mean", "lag_std"]
RATIO_FEATURES = ["ratio_lag1"]
OTHER_DIRECTION_FEATURES = ["other_lag1", "other_gap"]
WEEKLY_FEATURES = ["weekly_lag1", "weekly_gap"]

OTHER_DIRECTION = {"upwards": "downwards", "downwards": "upwards"}

 
def build_history(auctions, n_lags=3, ratio_lags=0):
    cols = ["auction_id", "series", "avail", "y"] + (["supply_ratio"] if ratio_lags else [])
    history = auctions.loc[auctions["y"].notna(), cols].copy()
    history = history.sort_values(["series", "avail"])

    for i in range(n_lags):
        history[f"lag{i + 1}"] = history.groupby("series")["y"].shift(i)

    if ratio_lags:
        history["log_ratio"] = np.log10(history["supply_ratio"])
        for i in range(ratio_lags):
            history[f"ratio_lag{i + 1}"] = history.groupby("series")["log_ratio"].shift(i)
        history = history.drop(columns=["supply_ratio", "log_ratio"])

    return history.drop(columns="y")


def build_features(auctions, n_lags=3, ratio_lags=0):
    """Attach to each auction the lags of the last auction of its series available at its gate closure.

    Warning: the output still has this auction's own result columns (n_bids, offered,
    awarded, supply_ratio, ...). Those are only known after the auction: never use
    them as model inputs. Use the lag columns instead.
    """
    history = build_history(auctions, n_lags, ratio_lags)
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
    from src.data.data import build_auctions

    auctions = build_auctions(product="mFRR")
    auctions = auctions[~auctions["auction_id"].str.contains("KORR")]
    features = build_features(auctions)
    print(features[["auction_id", "series", "gc", "lag1", "lag2", "lag3", "h_auction_id", "h_avail"]].head(8))
    ok = features["h_avail"].isna() | (features["h_avail"] <= features["gc"])
    print("no future data:", ok.all())
    print("no self-reference:", (features["h_auction_id"] != features["auction_id"]).all())
 

def add_model_features(features):
    out = features.copy()
    block = out["series"].str.split("|").str[-1]
    out = out.join(pd.get_dummies(block, prefix="block", dtype=int))
    if "day_type" in out:
        out = out.join(pd.get_dummies(out["day_type"], prefix="daytype", dtype=int))
    return out


def add_lag_features(features):
    out = features.copy()
    out["mom1"] = out["lag1"] - out["lag2"]
    out["mom2"] = out["lag2"] - out["lag3"]
    out["lag_mean"] = out[LAGS].mean(axis=1)
    out["lag_std"] = out[LAGS].std(axis=1)
    return out


def _series_parts(df):
    parts = df["series"].str.split("|")
    return parts.str[0], parts.str[1], parts.str[2]


def _attach_last_price(features, auctions, target_key, history_key, name):
    history = auctions.loc[auctions["y"].notna()].copy()
    history["key"] = history_key(history)
    history = history[["key", "avail", "y"]].rename(columns={"avail": "_h_avail", "y": name})
    history = history.dropna(subset=["key"]).sort_values("_h_avail")

    out = features.copy()
    out["key"] = target_key(out)
    out = pd.merge_asof(
        out.sort_values("gc"),
        history,
        left_on="gc",
        right_on="_h_avail",
        by="key",
        direction="backward",
        allow_exact_matches=True,
    )
    return out.drop(columns=["key", "_h_avail"])


def add_other_direction_features(features, auctions):
    def target_key(df):
        freq, direction, block = _series_parts(df)
        return freq + "|" + direction.map(OTHER_DIRECTION) + "|" + block

    out = _attach_last_price(features, auctions, target_key, lambda df: df["series"], "other_lag1")
    out["other_gap"] = out["other_lag1"] - out["lag1"]
    return out


def add_weekly_features(features, auctions):
    def history_key(df):
        freq, direction, _ = _series_parts(df)
        return direction.where(freq == "weekly")   

    def target_key(df):
        return _series_parts(df)[1]

    out = _attach_last_price(features, auctions, target_key, history_key, "weekly_lag1")
    out["weekly_gap"] = out["weekly_lag1"] - out["lag1"]
    return out