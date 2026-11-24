"""
Previous-known-auction baseline (Sprint 1, card 3).

Forecast for auction A = marginal price of the most recent auction of the same
series (freq | direction | block) whose result was known at A's gate closure
(avail <= gc of A). No-buy auctions are skipped. This is y_lag1 from
src.features.lags.LeakageSafeLag.

Statement: "Baseline first. [...] Report it next to every model."

sklearn-like interface:
    model = PreviousKnownAuctionBaseline()
    model.fit(auctions)                      # store the known history
    preds = model.predict(test_auctions)     # contract format, CHF/MW
    metrics = model.evaluate(auctions, preds)
    model.save(preds, metrics)

Run everything (build table, predict, validate, evaluate, save):
    uv run python -m src.models.baseline
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.data import ROOT, build_auctions, save_auctions
from src.evaluation import Evaluator, PredictionValidator
from src.features.lags import LeakageSafeLag

START_TEST = "2025-01-01"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_PATH = RESULTS_DIR / "predictions_baseline.parquet"
METRICS_PATH = RESULTS_DIR / "metrics_baseline.parquet"


def select_test_auctions(auctions: pd.DataFrame, start_test: str = START_TEST) -> pd.DataFrame:
    """Auctions of the test period (gc >= start_test), no-buy auctions included since very one of them must get a prediction."""
    return auctions[auctions["gc"] >= pd.Timestamp(start_test, tz="UTC")]


class PreviousKnownAuctionBaseline:
    """Latest known auction of the same series, as a model with fit / predict.

    fit() only stores the history: the baseline has no parameters to learn.
    predict() never uses a result unknown at the auction's gate closure, because
    the lag only takes history rows with avail <= gc. So:
      - fit(full table) then predict(test auctions) is leakage-free, and the
        forecast uses the latest result published before each gate closure
        (also within a test month);
      - fit(train) inside WalkForwardSplitter gives a stricter variant whose
        lag is frozen at the start of each test month.

    Attributes set by fit (sklearn convention: trailing underscore):
        history_ : auctions table seen by fit (columns of build_auctions)
    """

    name = "baseline"

    def __init__(self, start_test: str = START_TEST):
        self.start_test = start_test
        self.lag = LeakageSafeLag()

    def fit(self, auctions: pd.DataFrame) -> PreviousKnownAuctionBaseline:
        """Store the auctions whose results the lag may use. Returns self."""
        if not set(["auction_id", "series", "gc", "avail", "y"]).issubset(auctions.columns):
            raise RuntimeError(
                "Baseline fit one of the following column is missing :auction_id, series, gc, avail, y")
        # baseline doesn't learn any param, we can give the full dataset
        self.history_ = auctions

        return self

    def predict(self, auctions: pd.DataFrame) -> pd.DataFrame:
        """Prediction file for `auctions`, following results/PREDICTION_CONTRACT.md.
        Returns columns: auction_id, marginal_price_pred (CHF/MW), model.
        """
        # here auction in input is only the test set (anywhere from 2025 onwards)
        if not hasattr(self, "history_"):
            raise RuntimeError("fit() was not called")
        lagged = self.lag.compute(self.history_)
        # only keep auctions asked in input
        lagged = lagged[lagged["auction_id"].isin(auctions["auction_id"])]
        if len(lagged) != len(auctions):
            raise ValueError(
                "some auctions to predict are not in the history given to fit()")
        lagged["y_lag1"] = lagged["y_lag1"].fillna(self._fallback(lagged))
        lagged["marginal_price_pred"] = 10**lagged["y_lag1"]
        lagged["model"] = self.name
        # no need to retrn y column, let's avoid catastrophies later on
        return lagged[["auction_id", "marginal_price_pred", "model"]].reset_index(drop=True)

    def evaluate(self, auctions: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
        """Validate the prediction file, then score it by freq x direction (4 rows)."""
        # validates that it has the rights columns, no NaNs or double, and a prediction for each auction
        test_ids = select_test_auctions(
            auctions, self.start_test)["auction_id"]
        PredictionValidator.validate(preds, expected_ids=test_ids)
        results = Evaluator(
            actuals=auctions, start_test=self.start_test).evaluate(preds)
        return results

    def save(self, preds: pd.DataFrame, metrics: pd.DataFrame,
             preds_path: Path = PREDICTIONS_PATH,
             metrics_path: Path = METRICS_PATH) -> None:
        """Write the prediction file and the metric table to results/."""
        preds_path, metrics_path = Path(preds_path), Path(metrics_path)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(preds_path, index=False)
        metrics.to_parquet(metrics_path, index=False)
        return

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _fallback(self, lagged: pd.DataFrame) -> pd.Series:
        """log10 forecast for auctions whose y_lag1 is NaN (no known price in the series).
        Must only use results known at each auction's gate closure.
        Sends back a series of prices in log10 not the whole df !!!
        """
        missing = lagged.loc[lagged["y_lag1"].isna(), "auction_id"]
        if len(missing):
            raise ValueError(
                f"{len(missing)} auctions have no known price in their series "
                f"(e.g. {missing.head(5).tolist()}): a fallback rule is needed"
            )
        return pd.Series(index=lagged.index, dtype=float)


if __name__ == "__main__":
    auctions = build_auctions()
    save_auctions(auctions)

    model = PreviousKnownAuctionBaseline().fit(auctions)
    preds = model.predict(select_test_auctions(auctions))
    metrics = model.evaluate(auctions, preds)
    print(metrics)
    model.save(preds, metrics)
