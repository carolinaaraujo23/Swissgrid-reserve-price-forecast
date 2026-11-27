"""LightGBM quantile model: quantile order and leakage (future perturbation)."""

import numpy as np
import pandas as pd
from test_features import make_weekly_auctions, perturbe_future

from src.features.build_features import add_model_features, build_features
from src.models.lgbm import N_ESTIMATORS_GRID, choose_n_estimators, to_contract, walk_forward_lgbm
from src.models.lgbm import recency_weights

KW = dict(start_test="2024-03-01", end_test="2024-05-31")


def test_quantiles_do_not_cross():
    feats = add_model_features(build_features(make_weekly_auctions()))
    preds = walk_forward_lgbm(feats, **KW)
    assert len(preds) > 0
    assert (preds["p10_lgbm"] <= preds["p50_lgbm"]).all()
    assert (preds["p50_lgbm"] <= preds["p90_lgbm"]).all()
    contract = to_contract(preds)
    assert (contract["marginal_price_p10"] <= contract["marginal_price_pred"]).all()
    assert (contract["marginal_price_pred"] <= contract["marginal_price_p90"]).all()
    assert (contract["marginal_price_pred"] > 0).all()


def test_lgbm_predictions_do_not_change_when_future_is_perturbed():
    auctions = make_weekly_auctions()
    t = pd.Timestamp("2024-03-26 12:00", tz="UTC")

    preds = walk_forward_lgbm(add_model_features(build_features(auctions)), **KW).set_index("auction_id")
    perturbed = add_model_features(build_features(perturbe_future(auctions, t)))
    perturbed_preds = walk_forward_lgbm(perturbed, **KW).set_index("auction_id")

    past = preds["gc"] <= t
    for col in ["p10_lgbm", "p50_lgbm", "p90_lgbm"]:
        pd.testing.assert_series_equal(preds.loc[past, col], perturbed_preds.loc[past, col])
    assert not preds.loc[~past, "p50_lgbm"].equals(perturbed_preds.loc[~past, "p50_lgbm"])


def test_number_of_trees_comes_from_the_grid():
    """With enough history the inner time-only validation picks a value of the grid;
    with too little history it falls back to the first value."""
    tr = add_model_features(build_features(make_weekly_auctions(n_weeks=80))).dropna(subset=["y", "lag1"])
    y_delta = tr["y"] - tr["lag1"]
    assert choose_n_estimators(tr, ["lag1", "lag2", "lag3"], y_delta, 0.5) in N_ESTIMATORS_GRID
    short = tr.head(10)
    assert choose_n_estimators(short, ["lag1", "lag2", "lag3"], y_delta.head(10), 0.5) == N_ESTIMATORS_GRID[0]

def test_recency_weights_halve_every_half_life():
    ref = pd.Timestamp("2024-07-01", tz="UTC")
    gc = pd.Series(pd.to_datetime(["2024-07-01", "2024-01-01", "2023-07-01"], utc=True))
    w = recency_weights(gc, ref, half_life_months=6)
    assert np.isclose(w.iloc[0], 1.0)
    assert np.isclose(w.iloc[1], 0.5, atol=0.01)      # ~6 months before: half
    assert np.isclose(w.iloc[2], 0.25, atol=0.01)     # ~12 months before: a quarter


def test_training_window_uses_only_recent_years():
    feats = build_features(make_weekly_auctions(n_weeks=150))
    full = walk_forward_lgbm(feats, start_test="2026-06-01", end_test="2026-08-31").attrs["tuning"]
    short = walk_forward_lgbm(feats, start_test="2026-06-01", end_test="2026-08-31", train_years=1).attrs["tuning"]
    assert (short["n_train"] < full["n_train"]).all()
    assert (short["n_train"] <= 54).all()             # about one year of weekly auctions


def test_weighted_predictions_do_not_change_when_future_is_perturbed():
    auctions = make_weekly_auctions(n_weeks=80)
    t = pd.Timestamp("2024-03-26 12:00", tz="UTC")
    kw = dict(start_test="2024-03-01", end_test="2024-05-31", half_life_months=6)
    before = walk_forward_lgbm(build_features(auctions), **kw).set_index("auction_id")
    after = walk_forward_lgbm(build_features(perturbe_future(auctions, t)), **kw).set_index("auction_id")
    past = before["gc"] <= t
    for col in ["p10_lgbm", "p50_lgbm", "p90_lgbm"]:
        pd.testing.assert_series_equal(before.loc[past, col], after.loc[past, col])