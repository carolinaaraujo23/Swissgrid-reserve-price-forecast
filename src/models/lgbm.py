import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.evaluation import WalkForwardSplitter
from src.models.ridge import GROUP_COLS, TEST_START, evaluate, model_feature_columns, prepare_auctions, save

QUANTILES = (0.1, 0.5, 0.9)                 # p10 / p50 (the prediction) / p90
MODEL_NAME = "lgbm_quantile"
N_ESTIMATORS_GRID = (100, 300, 600)         # candidates for the number of trees
VALID_MONTHS = 3                            # inner time-only validation window
MIN_FIT_ROWS, MIN_VALID_ROWS = 200, 50      # below this, no tuning: first value of the grid
PARAMS = dict(
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)


def lgbm_quantile_model(q, n_estimators):
    return LGBMRegressor(objective="quantile", alpha=q, n_estimators=n_estimators, **PARAMS)


def pinball(y, pred, q):
    """Mean pinball loss: being above the actual value costs (1 - q), below costs q."""
    diff = y - pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def recency_weights(gc, reference, half_life_months):
    """Weight of each training auction: 1 at `reference`, halved every `half_life_months` before it."""
    age_months = (reference - gc).dt.total_seconds() / (30.44 * 24 * 3600)
    return 0.5 ** (age_months / half_life_months)


def choose_n_estimators(tr, feature_cols, y_train, q, make_model=lgbm_quantile_model, weights=None):
    start_valid = tr["gc"].max() - pd.DateOffset(months=VALID_MONTHS)
    fit = (tr["avail"] < start_valid).to_numpy()
    valid = (tr["gc"] >= start_valid).to_numpy()
    if fit.sum() < MIN_FIT_ROWS or valid.sum() < MIN_VALID_ROWS:
        return N_ESTIMATORS_GRID[0]

    X, y = tr[feature_cols], y_train.to_numpy()
    w = None if weights is None else np.asarray(weights)[fit]
    losses = {}
    for n in N_ESTIMATORS_GRID:
        model = make_model(q, n).fit(X[fit], y[fit], sample_weight=w)
        losses[n] = pinball(y[valid], model.predict(X[valid]), q)
    return min(losses, key=losses.get)


def walk_forward_lgbm(features, start_test="2024-01-01", end_test="2024-12-31", feature_cols=None,
                      target="delta", group_cols=GROUP_COLS, quantiles=QUANTILES,
                      make_model=lgbm_quantile_model, half_life_months=None, train_years=None):
    if target not in ("level", "delta"):
        raise ValueError('target must be "level" or "delta"')
    if 0.5 not in quantiles:
        raise ValueError("quantiles must include 0.5 (the prediction)")
    feature_cols = list(feature_cols) if feature_cols is not None else model_feature_columns(features)
    quantiles = sorted(quantiles)
    splitter = WalkForwardSplitter(start_test=start_test, end_test=end_test, freq="M")
    results, tuning = [], []

    for train, test in splitter.splits(features):
        for key, te in test.groupby(list(group_cols)):
            key = key if isinstance(key, tuple) else (key,)
            in_group = (train[list(group_cols)] == pd.Series(key, index=list(group_cols))).all(axis=1)
            tr = train[in_group].dropna(subset=["y", "lag1"]).sort_values("gc")
            if train_years is not None:
                tr = tr[tr["gc"] >= te["gc"].min() - pd.DateOffset(years=train_years)]
            y_train = tr["y"] - tr["lag1"] if target == "delta" else tr["y"]
            weights = None
            if half_life_months is not None:
                weights = recency_weights(tr["gc"], tr["gc"].max(), half_life_months).to_numpy()

            preds = []
            for q in quantiles:
                n = choose_n_estimators(tr, feature_cols, y_train, q, make_model, weights)
                model = make_model(q, n).fit(tr[feature_cols], y_train, sample_weight=weights)
                pred = model.predict(te[feature_cols])
                if target == "delta":
                    pred = pred + te["lag1"].to_numpy()
                preds.append(pred)
                tuning.append({"month": te["gc"].min().strftime("%Y-%m"), **dict(zip(group_cols, key)),
                               "quantile": q, "n_estimators": n, "n_train": len(tr)})

            # no quantile crossing: sort the predictions of each auction (p10 <= p50 <= p90)
            preds = np.sort(np.column_stack(preds), axis=1)

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
    save(contract, metrics, ROOT / "results")

    print(metrics.round(3).to_string(index=False))
    print("\nnumber of trees chosen (per quantile):")
    print(preds.attrs["tuning"].groupby("quantile")["n_estimators"].value_counts().unstack(fill_value=0))