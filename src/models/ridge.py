from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge

from src.evaluation import Evaluator, PredictionValidator, WalkForwardSplitter

FEATURES = ["lag1", "lag2", "lag3"]
GROUP_COLS = ["freq", "direction"]      
MODEL_NAME = "ridge_lags3"
TEST_START = "2025-01-01"


def prepare_auctions(auctions, frequencies=("daily", "weekly")):
    if isinstance(frequencies, str):
        frequencies = [frequencies]
    keep = auctions["freq"].isin(frequencies) & ~auctions["auction_id"].str.contains("KORR")
    return auctions.loc[keep].reset_index(drop=True)


def walk_forward_ridge(features, start_test="2024-01-01", end_test="2024-12-31", alpha=1.0):
    splitter = WalkForwardSplitter(start_test=start_test, end_test=end_test, freq="M")
    results = []

    for train, test in splitter.splits(features):
        for (freq, direction), te in test.groupby(GROUP_COLS):

            in_group = (train["freq"] == freq) & (train["direction"] == direction)
            tr = train[in_group].dropna(subset=FEATURES + ["y"])

            model = Ridge(alpha=alpha)
            model.fit(tr[FEATURES], tr["y"])

            X = te[FEATURES].ffill(axis=1)

            out = te[["auction_id", "freq", "direction", "series", "gc", "y"]].copy()
            out["pred_ridge"] = model.predict(X)
            out["pred_baseline"] = te["lag1"]
            results.append(out)

    return pd.concat(results, ignore_index=True)


def to_contract(preds, model_name=MODEL_NAME):
    return pd.DataFrame({
        "auction_id": preds["auction_id"],
        "marginal_price_pred": 10 ** preds["pred_ridge"],
        "model": model_name,
    })


def evaluate(contract_preds, auctions, start_test=TEST_START):
    test_ids = auctions.loc[auctions["gc"] >= pd.Timestamp(start_test, tz="UTC"), "auction_id"]
    PredictionValidator.validate(contract_preds, expected_ids=test_ids)
    return Evaluator(actuals=auctions, start_test=start_test).evaluate(contract_preds)


def save(contract_preds, metrics, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model = contract_preds["model"].iloc[0]
    contract_preds.to_parquet(results_dir / f"predictions_{model}.parquet", index=False)
    metrics.to_parquet(results_dir / f"metrics_{model}.parquet", index=False)


if __name__ == "__main__":
    from src.data.data import ROOT, build_auctions
    from src.features.build_features import build_features

    FREQUENCIES = ("daily", "weekly")        
    auctions = prepare_auctions(build_auctions(product="mFRR"), FREQUENCIES)
    features = build_features(auctions)

    preds = walk_forward_ridge(features, start_test=TEST_START, end_test=None)
    contract = to_contract(preds)
    metrics = evaluate(contract, auctions)
    save(contract, metrics, ROOT / "results")

    print(metrics.round(3).to_string(index=False))