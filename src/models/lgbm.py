import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from src.evaluation import WalkForwardSplitter
from src.models.ridge import GROUP_COLS, TEST_START, evaluate, model_feature_columns, prepare_auctions, save

QUANTILES = (0.1, 0.5, 0.9)                 
MODEL_NAME = "lgbm_quantile"
N_ESTIMATORS_GRID = (100, 300, 600)         
VALID_MONTHS = 3                            
MIN_FIT_ROWS, MIN_VALID_ROWS = 200, 50     
PARAMS = dict(
    learning_rate=0.05,  
    num_leaves=15,
    min_child_samples=50,
    subsample=0.8,          # uses 0.8 of the auctions chosen randomly per tree
    subsample_freq=1,       
    colsample_bytree=0.8,   # uses 0.8 of the features that exist so differente trees learn different patterns
    verbose=-1,
)

def lgbm_quantile_model(q, n_estimators):
    return LGBMRegressor(objective="quantile", alpha=q, n_estimators=n_estimators, **PARAMS)


def pinball(y, pred, q):
    diff = y - pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def choose_n_estimators(tr, feature_cols, y_train, q, make_model=lgbm_quantile_model): 
    """Chooses the number of trees."""
    start_valid = tr["gc"].max() - pd.DateOffset(months=VALID_MONTHS)
    fit = (tr["avail"] < start_valid).to_numpy()
    valid = (tr["gc"] >= start_valid).to_numpy()
    if fit.sum() < MIN_FIT_ROWS or valid.sum() < MIN_VALID_ROWS:
        return N_ESTIMATORS_GRID[0]

    X, y = tr[feature_cols], y_train.to_numpy()
    losses = {}
    for n in N_ESTIMATORS_GRID:
        model = make_model(q, n).fit(X[fit], y[fit])
        losses[n] = pinball(y[valid], model.predict(X[valid]), q)
    return min(losses, key=losses.get)


def walk_forward_lgbm(features, start_test="2024-01-01", end_test="2024-12-31", feature_cols=None,
                      target="delta", group_cols=GROUP_COLS, quantiles=QUANTILES,
                      make_model=lgbm_quantile_model):
    if target not in ("level", "delta"):
        raise ValueError('target must be "level" or "delta"')
    if 0.5 not in quantiles:
        raise ValueError("quantiles must include 0.5")
    feature_cols = list(feature_cols) if feature_cols is not None else model_feature_columns(features)
    quantiles = sorted(quantiles)
    splitter = WalkForwardSplitter(start_test=start_test, end_test=end_test, freq="M")
    results, tuning = [], []

    for train, test in splitter.splits(features):
        for key, te in test.groupby(list(group_cols)):
            key = key if isinstance(key, tuple) else (key,)
            in_group = (train[list(group_cols)] == pd.Series(key, index=list(group_cols))).all(axis=1)
            tr = train[in_group].dropna(subset=["y", "lag1"]).sort_values("gc")
            y_train = tr["y"] - tr["lag1"] if target == "delta" else tr["y"]

            preds = []
            for q in quantiles:
                n = choose_n_estimators(tr, feature_cols, y_train, q, make_model)
                model = make_model(q, n).fit(tr[feature_cols], y_train)
                pred = model.predict(te[feature_cols])
                if target == "delta":
                    pred = pred + te["lag1"].to_numpy()
                preds.append(pred)
                tuning.append({"month": te["gc"].min().strftime("%Y-%m"), **dict(zip(group_cols, key)),
                               "quantile": q, "n_estimators": n, "n_train": len(tr)})

           
            preds = np.sort(np.column_stack(preds), axis=1)     # to be sure q10 <= q50 <= q90

            out = te[["auction_id", "freq", "direction", "series", "gc", "y"]].copy()
            for j, q in enumerate(quantiles):
                out[f"p{round(100 * q)}_lgbm"] = preds[:, j]
            out["pred_lgbm"] = out["p50_lgbm"]
            out["pred_baseline"] = te["lag1"]
            results.append(out)

    result = pd.concat(results, ignore_index=True)
    result.attrs["tuning"] = pd.DataFrame(tuning)
    return result


def to_contract(preds, model_name=MODEL_NAME):
    out = pd.DataFrame({
        "auction_id": preds["auction_id"],
        "marginal_price_pred": 10 ** preds["pred_lgbm"],
        "model": model_name,
    })
    if {"p10_lgbm", "p90_lgbm"}.issubset(preds.columns):
        out["marginal_price_p10"] = 10 ** preds["p10_lgbm"]
        out["marginal_price_p90"] = 10 ** preds["p90_lgbm"]
    return out


if __name__ == "__main__":
    from src.data.data import ROOT, build_auctions
    from src.features.build_features import add_model_features, build_features

    FREQUENCIES = ("daily", "weekly")

    auctions = prepare_auctions(build_auctions(product="mFRR"), FREQUENCIES)
    features = add_model_features(build_features(auctions))

    preds = walk_forward_lgbm(features, start_test=TEST_START, end_test=None)
    contract = to_contract(preds)
    metrics = evaluate(contract, auctions)
    save(contract, metrics, ROOT / "results/parquets")

    print(metrics.round(3).to_string(index=False))
    print("\nnumber of trees chosen (per quantile):")
    print(preds.attrs["tuning"].groupby("quantile")["n_estimators"].value_counts().unstack(fill_value=0))