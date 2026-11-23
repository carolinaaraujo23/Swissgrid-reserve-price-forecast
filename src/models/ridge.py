import pandas as pd 
from sklearn.linear_model import Ridge

from src.evaluation import WalkForwardSplitter

FEATURES = ["lag1", "lag2", "lag3"]
DIRECTIONS = ["upwards", "downwards"]

def walk_forward_ridge(features, start_test="2024-01-01", end_test="2024-12-31", alpha=1.0):

    splitter = WalkForwardSplitter(
        start_test=start_test, end_test=end_test, freq="M",
        avail_col="available_time_utc", gc_col="gate_closure_time_utc",
    )

    results = []

    for train, test in splitter.splits(features):
        for direction in DIRECTIONS:

            tr = train[train["direction"] == direction].dropna(subset=FEATURES + ["log10_price"])
            te = test[test["direction"] == direction]

            model = Ridge(alpha=alpha)
            model.fit(tr[FEATURES], tr["log10_price"])

            out = te[["auction_id", "direction", "gate_closure_time_utc", "log10_price"]].copy()
            out["pred_ridge"] = model.predict(te[FEATURES])
            out["pred_baseline"] = te["lag1"]
            results.append(out)

    return pd.concat(results, ignore_index=True)