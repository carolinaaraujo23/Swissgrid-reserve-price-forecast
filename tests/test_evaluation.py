"""Unit tests of the evaluation harness (src/evaluation/evaluation.py).

Small hand-made tables with known answers: no real data needed.
Run with: uv run pytest tests/test_evaluation.py
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import Evaluator, Metrics, PredictionValidator, WalkForwardSplitter


def make_auctions(rows):
    """rows: list of (auction_id, freq, direction, gate_closure_utc, marginal_price).

    marginal_price None = no-buy auction. Columns follow src.data.data.build_auctions.
    """
    df = pd.DataFrame(rows, columns=["auction_id", "freq", "direction", "gc", "marginal"])
    df["gc"] = pd.to_datetime(df["gc"], utc=True)
    df["marginal"] = df["marginal"].astype(float)
    df["y"] = np.log10(df["marginal"])
    df["avail"] = df["gc"] + pd.Timedelta(hours=2)
    df["series"] = df["freq"] + "|" + df["direction"]
    return df


def perfect_preds(auctions, model="m"):
    """Exact predictions for every auction (no-buy ones get an arbitrary price)."""
    return pd.DataFrame({
        "auction_id": auctions["auction_id"],
        "marginal_price_pred": auctions["marginal"].fillna(1.0),
        "model": model,
    })


# 4 groups (freq x direction), one no-buy auction, one auction before the test period
AUCTIONS = make_auctions([
    ("old",  "daily",  "upwards",   "2024-12-20 13:30", 50.0),   # before 2025: never scored
    ("du1",  "daily",  "upwards",   "2025-01-10 13:30", 10.0),
    ("du2",  "daily",  "upwards",   "2025-01-11 13:30", 100.0),
    ("dd1",  "daily",  "downwards", "2025-01-10 13:30", 5.0),
    ("dd2",  "daily",  "downwards", "2025-01-11 13:30", None),   # no-buy: never scored
    ("wu1",  "weekly", "upwards",   "2025-01-07 12:00", 200.0),
    ("wd1",  "weekly", "downwards", "2025-01-07 12:00", 20.0),
])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_mae_and_rmse_hand_computed():
    y, p = [1.0, 2.0], [1.1, 1.7]                      # errors 0.1 and 0.3
    assert Metrics.mae(y, p) == pytest.approx(0.2)
    assert Metrics.rmse(y, p) == pytest.approx(np.sqrt((0.01 + 0.09) / 2))


def test_pinball_is_asymmetric():
    # under-prediction by 1: costs q
    assert Metrics.pinball([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert Metrics.pinball([1.0], [0.0], 0.1) == pytest.approx(0.1)
    # over-prediction by 1: costs 1 - q
    assert Metrics.pinball([0.0], [1.0], 0.9) == pytest.approx(0.1)


def test_pinball_matches_roadmap_formula():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=50), rng.normal(size=50)
    d = y - p
    for q in (0.1, 0.5, 0.9):
        assert Metrics.pinball(y, p, q) == pytest.approx(np.mean(np.maximum(q * d, (q - 1) * d)))


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_pinball_rejects_invalid_quantile(q):
    with pytest.raises(ValueError):
        Metrics.pinball([1.0], [1.0], q)


def test_coverage_counts_bounds_as_inside():
    y = [1.0, 2.0, 3.0, 4.0]
    lo = [1.0, 1.5, 3.5, 3.0]      # y=1 sits exactly on lo, y=3 is below lo
    hi = [2.0, 2.0, 4.0, 4.0]      # y=2 and y=4 sit exactly on hi
    assert Metrics.coverage(y, lo, hi) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# WalkForwardSplitter
# ---------------------------------------------------------------------------

SPLIT_DF = make_auctions([
    ("nov",        "daily", "upwards", "2024-11-15 13:30", 10.0),
    ("dec_late",   "daily", "upwards", "2024-12-31 23:30", 10.0),  # known 2025-01-01 01:30
    ("jan_first",  "daily", "upwards", "2025-01-01 00:00", 10.0),
    ("jan_last",   "daily", "upwards", "2025-01-31 23:00", 10.0),  # known 2025-02-01 01:00
    ("mar",        "daily", "upwards", "2025-03-10 13:30", 10.0),  # February is empty
])


def splits_by_month(df, **kwargs):
    out = {}
    for train, test in WalkForwardSplitter(**kwargs).splits(df):
        month = test["gc"].min().strftime("%Y-%m")
        out[month] = (set(train["auction_id"]), set(test["auction_id"]))
    return out


def test_one_split_per_non_empty_month():
    assert list(splits_by_month(SPLIT_DF)) == ["2025-01", "2025-03"]   # February skipped


def test_test_months_contain_their_auctions():
    s = splits_by_month(SPLIT_DF)
    assert s["2025-01"][1] == {"jan_first", "jan_last"}   # 31 Jan 23:00 is still January
    assert s["2025-03"][1] == {"mar"}


def test_train_only_has_results_known_before_the_month():
    s = splits_by_month(SPLIT_DF)
    # dec_late closes in December but its result is known on 1 Jan 01:30: not usable in January
    assert s["2025-01"][0] == {"nov"}
    assert s["2025-03"][0] == {"nov", "dec_late", "jan_first", "jan_last"}


def test_train_never_overlaps_test():
    for train, test in WalkForwardSplitter().splits(SPLIT_DF):
        assert (train["avail"] < test["gc"].min()).all()
        assert set(train["auction_id"]).isdisjoint(test["auction_id"])


def test_end_test_limits_the_test_period():
    assert list(splits_by_month(SPLIT_DF, end_test="2025-01-31")) == ["2025-01"]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def test_one_row_per_freq_and_direction():
    report = Evaluator(actuals=AUCTIONS).evaluate(perfect_preds(AUCTIONS))
    groups = set(zip(report["freq"], report["direction"]))
    assert groups == {("daily", "upwards"), ("daily", "downwards"),
                      ("weekly", "upwards"), ("weekly", "downwards")}
    n = report.set_index(["freq", "direction"])["n"]
    assert n[("daily", "upwards")] == 2       # "old" is before the test period
    assert n[("daily", "downwards")] == 1     # the no-buy auction is not scored


def test_perfect_predictions_score_zero():
    report = Evaluator(actuals=AUCTIONS).evaluate(perfect_preds(AUCTIONS))
    assert np.allclose(report["mae"], 0)
    assert np.allclose(report["rmse"], 0)


def test_errors_are_measured_in_log10():
    preds = perfect_preds(AUCTIONS)
    preds["marginal_price_pred"] *= 10 ** 0.1          # every forecast 0.1 too high in log10
    report = Evaluator(actuals=AUCTIONS).evaluate(preds)
    assert np.allclose(report["mae"], 0.1)
    assert np.allclose(report["rmse"], 0.1)


def test_missing_prediction_raises():
    preds = perfect_preds(AUCTIONS)
    preds = preds[preds["auction_id"] != "du2"]
    with pytest.raises(ValueError, match="no prediction"):
        Evaluator(actuals=AUCTIONS).evaluate(preds)


def test_allow_missing_scores_a_partial_file():
    preds = perfect_preds(AUCTIONS)
    preds = preds[preds["auction_id"] != "du2"]
    report = Evaluator(actuals=AUCTIONS, allow_missing=True).evaluate(preds)
    assert report.set_index(["freq", "direction"]).loc[("daily", "upwards"), "n"] == 1


def test_no_buy_and_pre_test_auctions_need_no_prediction():
    preds = perfect_preds(AUCTIONS)
    preds = preds[~preds["auction_id"].isin(["dd2", "old"])]
    Evaluator(actuals=AUCTIONS).evaluate(preds)        # does not raise


def test_weekly_only_tables_work():
    weekly = AUCTIONS[AUCTIONS["freq"] == "weekly"]
    report = Evaluator(actuals=weekly).evaluate(perfect_preds(weekly))
    assert len(report) == 2
    assert set(report["freq"]) == {"weekly"}


def test_weekly_preds_against_full_table_raise():
    weekly = AUCTIONS[AUCTIONS["freq"] == "weekly"]
    with pytest.raises(ValueError, match="no prediction"):
        Evaluator(actuals=AUCTIONS).evaluate(perfect_preds(weekly))


def test_quantile_metrics_only_with_p10_and_p90():
    preds = perfect_preds(AUCTIONS)
    report = Evaluator(actuals=AUCTIONS).evaluate(preds)
    assert report["coverage_80"].isna().all()

    preds["marginal_price_p10"] = preds["marginal_price_pred"] / 2
    preds["marginal_price_p90"] = preds["marginal_price_pred"] * 2
    report = Evaluator(actuals=AUCTIONS).evaluate(preds)
    assert np.allclose(report["coverage_80"], 1.0)
    assert np.allclose(report["pinball_q50"], 0.0)
    assert report[["pinball_q10", "pinball_q90"]].notna().all().all()


def test_evaluate_many_uses_the_dict_key_as_model_name():
    preds = perfect_preds(AUCTIONS, model="wrong_name")
    report = Evaluator(actuals=AUCTIONS).evaluate_many({"a": preds, "b": preds})
    assert set(report["model"]) == {"a", "b"}
    assert len(report) == 8                             # 2 models x 4 groups


def test_missing_columns_raise():
    preds = perfect_preds(AUCTIONS).drop(columns="model")
    with pytest.raises(ValueError, match="missing columns"):
        Evaluator(actuals=AUCTIONS).evaluate(preds)


# ---------------------------------------------------------------------------
# PredictionValidator
# ---------------------------------------------------------------------------

def valid_preds():
    return pd.DataFrame({
        "auction_id": ["a", "b", "c"],
        "marginal_price_pred": [10.0, 20.0, 30.0],
        "marginal_price_p10": [5.0, 10.0, 15.0],
        "marginal_price_p90": [20.0, 40.0, 60.0],
        "model": "m",
    })


def test_valid_file_passes(tmp_path):
    PredictionValidator.validate(valid_preds(), expected_ids=["a", "b", "c"])

    path = tmp_path / "predictions_m.parquet"
    valid_preds().to_parquet(path, index=False)
    PredictionValidator.validate(str(path))

    path = tmp_path / "predictions_m.csv"
    valid_preds().to_csv(path, index=False)
    PredictionValidator.validate(str(path))


def _drop_model(df):
    return df.drop(columns="model")


def _nan_pred(df):
    df.loc[0, "marginal_price_pred"] = np.nan
    return df


def _zero_pred(df):
    df.loc[0, "marginal_price_pred"] = 0.0
    return df


def _duplicate(df):
    df.loc[1, "auction_id"] = "a"
    return df


def _p10_above_pred(df):
    df.loc[0, "marginal_price_p10"] = 11.0
    return df


def _pred_above_p90(df):
    df.loc[0, "marginal_price_p90"] = 9.0
    return df


@pytest.mark.parametrize("breaker, message", [
    (_drop_model, "missing required column"),
    (_nan_pred, "NaN"),
    (_zero_pred, "<= 0"),
    (_duplicate, "duplicated"),
    (_p10_above_pred, "p10 > pred"),
    (_pred_above_p90, "pred > p90"),
])
def test_contract_violations_raise(breaker, message):
    with pytest.raises(ValueError, match=message):
        PredictionValidator.validate(breaker(valid_preds()))


def test_missing_expected_auction_raises():
    with pytest.raises(ValueError, match="no prediction"):
        PredictionValidator.validate(valid_preds(), expected_ids=["a", "b", "c", "d"])
