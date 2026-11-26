"""Hyper-parameter tuning of the LightGBM quantile model, and the bidding simulation.

Everything here is meant for the VALIDATION period only: the configuration and the bidding strategy are
chosen there, then 2025 is predicted once with the choices fixed.

- make_factory(**overrides): a `make_model` for walk_forward_lgbm with some PARAMS changed
- run_configs(...): one walk-forward per configuration, evaluated with the team's Evaluator
- summarise(report): mean pinball loss per configuration and group, compared with the default
- simulate_bids / bidding_summary: revenue of offering 1 MW in every auction (pay-as-bid)
"""

import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.models.lgbm import PARAMS, to_contract, walk_forward_lgbm

# Round 1: change one parameter at a time, starting from the current PARAMS.
# learning_rate stays at 0.05: the number of trees is chosen on the grid (100, 300, 600), which is
# calibrated for it.
CONFIGS = {
    "default": {},
    "leaves_7": {"num_leaves": 7},
    "leaves_31": {"num_leaves": 31},
    "child_20": {"min_child_samples": 20},
    "child_100": {"min_child_samples": 100},
    "child_200": {"min_child_samples": 200},
    "colsample_05": {"colsample_bytree": 0.5},
    "colsample_10": {"colsample_bytree": 1.0},
    "subsample_06": {"subsample": 0.6},
    "lambda_1": {"reg_lambda": 1.0},
    "lambda_5": {"reg_lambda": 5.0},
}

PINBALL_COLS = ["pinball_q10", "pinball_q50", "pinball_q90"]

# LightGBM parameters that are valid but not in LGBMRegressor().get_params() (passed as **kwargs)
EXTRA_LGBM_PARAMS = {"max_bin", "min_data_in_bin", "extra_trees", "path_smooth", "verbose", "seed"}


def make_factory(**overrides):
    """A make_model(q, n_estimators) for walk_forward_lgbm, with some PARAMS changed.

    make_factory(num_leaves=7) builds the same model as now, only with 7 leaves.
    """
    known = set(LGBMRegressor().get_params()) | EXTRA_LGBM_PARAMS
    unknown = set(overrides) - known
    if unknown:   # catches typos (num_leafs): with verbose=-1, LightGBM would ignore them silently
        raise ValueError(f"unknown LightGBM parameters: {sorted(unknown)}")
    params = {**PARAMS, **overrides}

    def make(q, n_estimators):
        return LGBMRegressor(objective="quantile", alpha=q, n_estimators=n_estimators, **params)

    return make


def run_configs(configs, features, feature_cols, evaluator, start_test, end_test):
    """Run one walk-forward per configuration and evaluate it with the team's Evaluator.

    Returns (report, contracts):
      - report: one row per configuration and group (columns of Evaluator.evaluate_many, plus
        `config` and `params`)
      - contracts: {configuration name: its predictions in the contract format}, for the bidding simulation
    """
    reports, contracts = [], {}
    for name, overrides in configs.items():
        print(f"{name}: running {overrides or '(current PARAMS)'}")
        preds = walk_forward_lgbm(features, start_test=start_test, end_test=end_test,
                                  feature_cols=feature_cols, make_model=make_factory(**overrides))
        contracts[name] = to_contract(preds, name)
        r = evaluator.evaluate_many({name: contracts[name]})
        r["config"] = name
        r["params"] = json.dumps(overrides, sort_keys=True)
        reports.append(r)
    return pd.concat(reports, ignore_index=True), contracts


def summarise(report, reference="default"):
    """Mean pinball loss (q10, q50, q90) per configuration and group.

    Columns: the four groups ("daily upwards", ...), `mean` (over the groups) and `groups_better` (in how many groups the
    configuration beats the reference). Sorted from best to worst mean.
    """
    r = report.copy()
    r["pinball_mean"] = r[PINBALL_COLS].mean(axis=1)
    table = r.pivot_table(index="config", columns=["freq", "direction"], values="pinball_mean")
    table.columns = [f"{freq} {direction}" for freq, direction in table.columns]   # e.g. "daily upwards"
    groups = list(table.columns)
    table["mean"] = table[groups].mean(axis=1)
    if reference in table.index:
        table["groups_better"] = (table[groups] < table.loc[reference, groups]).sum(axis=1)
    return table.sort_values("mean")


# ------------------------------------------------------------------------------- bidding simulation
STRATEGIES = {
    "bid p50": lambda c: c["marginal_price_pred"],
    "bid p10": lambda c: c.get("marginal_price_p10"),        # None if the model has no p10
    "bid 0.9 x p50": lambda c: 0.9 * c["marginal_price_pred"],
}


def simulate_bids(contract, auctions, bid):
    """One row per auction: the bid, whether it is accepted (pay-as-bid) and the revenue for 1 MW.

    A bid is accepted if bid <= marginal price. No-buy auctions (no marginal price) are never accepted.
    `perfect` is the revenue of bidding exactly the marginal price (perfect foresight).
    """
    df = contract[["auction_id"]].assign(bid=np.asarray(bid)).merge(
        auctions[["auction_id", "freq", "direction", "gc", "marginal"]], on="auction_id")
    df["accepted"] = df["marginal"].notna() & (df["bid"] <= df["marginal"])
    df["revenue"] = np.where(df["accepted"], df["bid"], 0.0)
    df["perfect"] = df["marginal"].fillna(0.0)
    df["year"] = df["gc"].dt.year
    return df


def bidding_summary(contracts, auctions, strategies=STRATEGIES):
    """Per group, model and strategy: mean revenue per year (CHF per MW offered), share of the
    perfect-foresight revenue, and share of accepted bids."""
    rows = []
    for name, contract in contracts.items():
        for strategy, make_bid in strategies.items():
            bid = make_bid(contract)
            if bid is None:
                continue
            sim = simulate_bids(contract, auctions, bid)
            g = sim.groupby(["freq", "direction"])
            n_years = sim["year"].nunique()
            rows.append(pd.DataFrame({
                "model": name,
                "strategy": strategy,
                "revenue": g["revenue"].sum() / n_years,
                "perfect": g["perfect"].sum() / n_years,
                "accepted": g["accepted"].mean(),
            }).reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["share"] = out["revenue"] / out["perfect"]
    return out