from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import Evaluator, PredictionValidator, WalkForwardSplitter

FEATURES = ["lag1", "lag2", "lag3"]
LAGS = ["lag1", "lag2", "lag3"]
GROUP_COLS = ["freq", "direction"]      
MODEL_NAME = "ridge_lags3"
TEST_START = "2025-01-01"
QUANTILES = (0.1, 0.9)                 


def prepare_auctions(auctions, frequencies=("daily", "weekly")):
    if isinstance(frequencies, str):
        frequencies = [frequencies]
    keep = auctions["freq"].isin(frequencies) & ~auctions["auction_id"].str.contains("KORR")
    return auctions.loc[keep].reset_index(drop=True)


def model_feature_columns(features):
    """Lags + the block and day-type one-hots made by add_model_features."""
    extra = [c for c in features.columns if c.startswith(("block_", "daytype_"))]
    return LAGS + extra


def walk_forward_ridge(features, start_test="2024-01-01", end_test="2024-12-31", alpha=1.0,
                       feature_cols=FEATURES, target="level", group_cols=GROUP_COLS,
                       quantiles=QUANTILES):
    if target not in ("level", "delta"):
        raise ValueError('target must be "level" or "delta"')
    feature_cols = list(feature_cols)
    splitter = WalkForwardSplitter(start_test=start_test, end_test=end_test, freq="M")
    results = []

    for train, test in splitter.splits(features):
        for key, te in test.groupby(list(group_cols)):

            key = key if isinstance(key, tuple) else (key,)
            in_group = (train[list(group_cols)] == pd.Series(key, index=list(group_cols))).all(axis=1)
            tr = train[in_group].dropna(subset=feature_cols + ["y"])
            y_train = tr["y"] - tr["lag1"] if target == "delta" else tr["y"]

            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(tr[feature_cols], y_train)

            X = te[feature_cols].copy()
            lags = [c for c in LAGS if c in feature_cols]
            X[lags] = X[lags].ffill(axis=1)              
            X = X.fillna(tr[feature_cols].median())       

            pred = model.predict(X)
            pred_train = model.predict(tr[feature_cols])
            if target == "delta":
                pred = pred + te["lag1"].to_numpy()
                pred_train = pred_train + tr["lag1"].to_numpy()
            errors = tr["y"].to_numpy() - pred_train         

            out = te[["auction_id", "freq", "direction", "series", "gc", "y"]].copy()
            out["pred_ridge"] = pred
            for q in quantiles:
                out[f"p{round(100 * q)}_ridge"] = pred + np.quantile(errors, q)
            out["pred_baseline"] = te["lag1"]
            results.append(out)

    return pd.concat(results, ignore_index=True)


def to_contract(preds, model_name=MODEL_NAME):
    out = pd.DataFrame({
        "auction_id": preds["auction_id"],
        "marginal_price_pred": 10 ** preds["pred_ridge"],
        "model": model_name,
    })
    if {"p10_ridge", "p90_ridge"}.issubset(preds.columns):
        # the contract requires p10 <= pred <= p90
        out["marginal_price_p10"] = 10 ** np.minimum(preds["p10_ridge"], preds["pred_ridge"])
        out["marginal_price_p90"] = 10 ** np.maximum(preds["p90_ridge"], preds["pred_ridge"])
    return out


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
    from src.features.build_features import add_model_features, build_features

    FREQUENCIES = ("daily", "weekly")
    MODEL = "ridge_allfeats_delta"      

    auctions = prepare_auctions(build_auctions(product="mFRR"), FREQUENCIES)
    features = add_model_features(build_features(auctions))
    variants = {
        "ridge_lags3": dict(feature_cols=FEATURES, target="level"),
        "ridge_lags3_delta": dict(feature_cols=FEATURES, target="delta"),
        "ridge_allfeats_delta": dict(feature_cols=model_feature_columns(features), target="delta"),
        "ridge_lags3_delta_per_series": dict(feature_cols=FEATURES, target="delta", group_cols=["series"]),
    }

    preds = walk_forward_ridge(features, start_test=TEST_START, end_test=None, **variants[MODEL])
    contract = to_contract(preds, MODEL)
    metrics = evaluate(contract, auctions)
    save(contract, metrics, ROOT / "results/parquets")

    print(metrics.round(3).to_string(index=False))