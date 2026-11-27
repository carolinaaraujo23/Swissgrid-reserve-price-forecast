import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.evaluation import Evaluator
from src.models.lgbm import PARAMS, to_contract, walk_forward_lgbm

# change one parameter at a time, starting from the current PARAMS.
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

EXTRA_LGBM_PARAMS = {"max_bin", "min_data_in_bin", "extra_trees", "path_smooth", "verbose", "seed"}

WEIGHTING = {
    "default": {},
    "half_life_6m": {"half_life_months": 6},
    "half_life_12m": {"half_life_months": 12},
    "half_life_24m": {"half_life_months": 24},
}
WINDOWS = {
    "default": {},
    "last_2_years": {"train_years": 2},
    "last_3_years": {"train_years": 3},
}

STRATEGIES = {
    "bid p50": lambda c: c["marginal_price_pred"],
    "bid p10": lambda c: c.get("marginal_price_p10"),        
    "bid 0.9 x p50": lambda c: 0.9 * c["marginal_price_pred"],
}


def make_factory(**overrides):
  
    known = set(LGBMRegressor().get_params()) | EXTRA_LGBM_PARAMS
    unknown = set(overrides) - known
    if unknown:   # catches typos (num_leafs): with verbose=-1, LightGBM would ignore them silently
        raise ValueError(f"unknown LightGBM parameters: {sorted(unknown)}")
    params = {**PARAMS, **overrides}

    def make(q, n_estimators):
        return LGBMRegressor(objective="quantile", alpha=q, n_estimators=n_estimators, **params)

    return make


def run_configs(configs, features, feature_cols, evaluator, start_test, end_test):
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


def run_variants(variants, features, feature_cols, evaluator, start_test, end_test):
    reports, contracts = [], {}
    for name, kwargs in variants.items():
        print(f"{name}: running {kwargs or '(default training)'}")
        preds = walk_forward_lgbm(features, start_test=start_test, end_test=end_test,
                                  feature_cols=feature_cols, **kwargs)
        contracts[name] = to_contract(preds, name)
        r = evaluator.evaluate_many({name: contracts[name]})
        r["config"] = name
        r["params"] = json.dumps(kwargs, sort_keys=True)
        reports.append(r)
    return pd.concat(reports, ignore_index=True), contracts


def per_year_report(contracts, auctions, first_year, last_year):
    reports = []
    for year in range(first_year, last_year + 1):
        r = Evaluator(actuals=auctions, start_test=f"{year}-01-01", end_test=f"{year + 1}-01-01").evaluate_many(contracts)
        r["year"] = year
        reports.append(r)
    return pd.concat(reports, ignore_index=True)


def summarise(report, reference="default"):
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

def simulate_bids(contract, auctions, bid):
    df = contract[["auction_id"]].assign(bid=np.asarray(bid)).merge(
        auctions[["auction_id", "freq", "direction", "gc", "marginal"]], on="auction_id")
    df["accepted"] = df["marginal"].notna() & (df["bid"] <= df["marginal"])
    df["revenue"] = np.where(df["accepted"], df["bid"], 0.0)
    df["perfect"] = df["marginal"].fillna(0.0)
    df["year"] = df["gc"].dt.year
    return df


def bidding_summary(contracts, auctions, strategies=STRATEGIES):
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