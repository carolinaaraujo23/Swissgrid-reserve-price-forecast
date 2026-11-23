"""
Evaluation harness for Swissgrid control-power marginal price forecasting.

Every model is scored with this module, so that results are comparable.
All metrics are computed on log10 of the marginal price (CHF/MW).

Usage:
    from src.evaluation import Metrics, WalkForwardSplitter, Evaluator, PredictionValidator

    # Individual metrics (inputs in log10)
    Metrics.mae(y_true, y_pred)
    Metrics.rmse(y_true, y_pred)
    Metrics.pinball(y_true, y_pred, q=0.5)
    Metrics.coverage(y_true, lo, hi)

    # Walk-forward split: retrain before each test month
    splitter = WalkForwardSplitter(start_test="2025-01-01", freq="M")
    for train, test in splitter.splits(auctions):
        ...

    # Score a prediction file, by model x freq x direction
    evaluator = Evaluator(actuals=auctions)
    report = evaluator.evaluate(preds)

    # Check a prediction file against results/PREDICTION_CONTRACT.md
    PredictionValidator.validate("results/predictions_baseline.parquet",
                                 expected_ids=test_auction_ids)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, root_mean_squared_error


def _to_utc(s) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


# ---------------------------------------------------------------------------
# 1. Metrics
# ---------------------------------------------------------------------------

class Metrics:
    """Metrics of the evaluation contract. Inputs are log10 prices."""

    @staticmethod
    def mae(y_true, y_pred) -> float:
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def rmse(y_true, y_pred) -> float:
        return float(root_mean_squared_error(y_true, y_pred))

    @staticmethod
    def pinball(y_true, y_pred, q: float) -> float:
        """Pinball loss at quantile q, 0 < q < 1."""
        if not 0 < q < 1:
            raise ValueError(f"q must be in (0, 1), got {q}")
        return float(mean_pinball_loss(y_true, y_pred, alpha=q))

    @staticmethod
    def coverage(y_true, lo, hi) -> float:
        """Share of y_true inside [lo, hi].

        Target 0.80 for p10-p90: below, the model is overconfident;
        above, its intervals are too wide.
        """
        y_true, lo, hi = np.asarray(y_true), np.asarray(lo), np.asarray(hi)
        return float(np.mean((y_true >= lo) & (y_true <= hi)))


# ---------------------------------------------------------------------------
# 2. Walk-forward splitter
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardSplitter:
    """Walk-forward split: one test period of length `freq` at a time,
    starting at `start_test`.

    train = auctions whose result is known before the period starts (avail < start)
    test  = auctions whose gate closure falls inside the period
    Train may contain no-buy auctions (y NaN); drop them before fitting.
    """

    start_test: str = "2025-01-01"
    end_test: str | None = None  # None = until the last gate closure in df
    freq: str = "M"              # "M" monthly, "W" weekly, "Q" quarterly
    avail_col: str = "avail"     # time from which the auction result is known
    gc_col: str = "gc"           # gate closure time (UTC)

    def splits(self, df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Yield (train, test) for each test period."""
        t0 = _to_utc(self.start_test)
        t_end = _to_utc(self.end_test) if self.end_test else df[self.gc_col].max()

        for period in pd.period_range(t0, t_end, freq=self.freq):
            p_start = period.start_time.tz_localize("UTC")
            # next period's start, with a strict "<" below: avoids 23:59:59.999999999
            p_end = (period + 1).start_time.tz_localize("UTC")

            train = df.loc[df[self.avail_col] < p_start]
            test = df.loc[(df[self.gc_col] >= p_start) & (df[self.gc_col] < p_end)]

            if test.empty:
                continue
            yield train, test


# ---------------------------------------------------------------------------
# 3. Evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvaluationRow:
    """One row of the evaluation report, per (model, freq, direction)."""
    model: str
    freq: str
    direction: str
    n: int
    mae: float
    rmse: float
    pinball_q10: float | None = None
    pinball_q50: float | None = None
    pinball_q90: float | None = None
    coverage_80: float | None = None


@dataclass
class Evaluator:
    """Score prediction files against the auction table.

    `actuals` is the table from src.data.data.build_auctions, with at least
    auction_id, y (log10 marginal price, NaN for no-buy auctions), freq,
    direction and gc.

    No-buy auctions (y NaN) are not scored. Every other auction of the test
    period [start_test, end_test] must have a prediction: missing ones raise
    an error, unless allow_missing=True.
    """

    actuals: pd.DataFrame
    start_test: str = "2025-01-01"
    end_test: str | None = None  # None = until the end of the data
    group_cols: tuple[str, ...] = ("freq", "direction")
    allow_missing: bool = False

    def _scorable(self) -> pd.DataFrame:
        a = self.actuals
        mask = a["y"].notna() & (a["gc"] >= _to_utc(self.start_test))
        if self.end_test:
            mask &= a["gc"] < _to_utc(self.end_test)
        return a.loc[mask]

    def _merge(self, preds: pd.DataFrame) -> pd.DataFrame:
        missing_cols = {"auction_id", "marginal_price_pred", "model"} - set(preds.columns)
        if missing_cols:
            raise ValueError(f"missing columns in preds: {missing_cols}")

        p = preds.copy()
        p["log10_pred"] = np.log10(p["marginal_price_pred"].astype(float))
        if {"marginal_price_p10", "marginal_price_p90"}.issubset(p.columns):
            p["log10_p10"] = np.log10(p["marginal_price_p10"].astype(float))
            p["log10_p90"] = np.log10(p["marginal_price_p90"].astype(float))

        scorable = self._scorable()
        if not self.allow_missing:
            for model, g in p.groupby("model"):
                missing = ~scorable["auction_id"].isin(g["auction_id"])
                if missing.any():
                    examples = scorable.loc[missing, "auction_id"].head(5).tolist()
                    raise ValueError(
                        f"model {model!r}: {missing.sum()} of {len(scorable)} scorable "
                        f"test auctions have no prediction (e.g. {examples}). "
                        "Pass allow_missing=True to score a partial file."
                    )

        merged = scorable.merge(p, on="auction_id", how="inner", suffixes=("", "_pred"))
        if merged.empty:
            raise ValueError(
                "no auction left after the merge: check the test period and auction_id"
            )
        return merged

    def evaluate(self, preds: pd.DataFrame) -> pd.DataFrame:
        """Return one row per (model, freq, direction)."""
        m = self._merge(preds)
        rows: list[EvaluationRow] = []

        for (model, freq, direction), g in m.groupby(["model", *self.group_cols]):
            y = g["y"].to_numpy()
            y_hat = g["log10_pred"].to_numpy()
            row = EvaluationRow(
                model=model, freq=freq, direction=direction, n=len(g),
                mae=Metrics.mae(y, y_hat),
                rmse=Metrics.rmse(y, y_hat),
            )
            if "log10_p10" in g.columns and "log10_p90" in g.columns:
                row.pinball_q10 = Metrics.pinball(y, g["log10_p10"].to_numpy(), 0.1)
                row.pinball_q50 = Metrics.pinball(y, y_hat, 0.5)
                row.pinball_q90 = Metrics.pinball(y, g["log10_p90"].to_numpy(), 0.9)
                row.coverage_80 = Metrics.coverage(
                    y, g["log10_p10"].to_numpy(), g["log10_p90"].to_numpy()
                )
            rows.append(row)

        return pd.DataFrame([r.__dict__ for r in rows])

    def evaluate_many(self, preds_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Score several models at once (key = model name)."""
        reports = []
        for name, preds in preds_by_model.items():
            p = preds.copy()
            p["model"] = name  # the key wins over any model column in the file
            reports.append(self.evaluate(p))
        return pd.concat(reports, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Prediction validator (results/PREDICTION_CONTRACT.md)
# ---------------------------------------------------------------------------

@dataclass
class PredictionValidator:
    """Check that a prediction file follows the contract."""

    required: tuple[str, ...] = ("auction_id", "marginal_price_pred", "model")
    recommended: tuple[str, ...] = ("marginal_price_p10", "marginal_price_p90")

    @classmethod
    def validate(
        cls,
        preds: str | pd.DataFrame,
        expected_ids: Iterable[str] | None = None,
    ) -> None:
        """Raise ValueError if the file breaks the contract.

        preds: path to a .parquet / .csv file, or a DataFrame.
        expected_ids: auction_id of every test auction (no-buy included);
            if given, each one must have exactly one prediction.
        """
        if isinstance(preds, pd.DataFrame):
            df, name = preds, "predictions"
        else:
            name = str(preds)
            df = pd.read_parquet(name) if name.endswith(".parquet") else pd.read_csv(name)

        for c in cls.required:
            if c not in df.columns:
                raise ValueError(f"missing required column: {c}")

        if df["marginal_price_pred"].isna().any():
            raise ValueError("NaN in marginal_price_pred")
        if (df["marginal_price_pred"] <= 0).any():
            raise ValueError("marginal_price_pred <= 0")
        if df["auction_id"].duplicated().any():
            dup = df.loc[df["auction_id"].duplicated(), "auction_id"].unique()[:5]
            raise ValueError(f"duplicated auction_id (e.g. {list(dup)})")

        if all(c in df.columns for c in cls.recommended):
            if (df["marginal_price_p10"] > df["marginal_price_pred"]).any():
                raise ValueError("p10 > pred on some rows")
            if (df["marginal_price_pred"] > df["marginal_price_p90"]).any():
                raise ValueError("pred > p90 on some rows")

        if expected_ids is not None:
            expected = pd.Index(expected_ids)
            missing = expected.difference(df["auction_id"])
            if len(missing):
                raise ValueError(
                    f"{len(missing)} of {len(expected)} test auctions have no "
                    f"prediction (e.g. {list(missing[:5])})"
                )

        print(f"[OK] {name}: {len(df)} rows, "
              f"{df['model'].nunique()} model(s), "
              f"{df['auction_id'].nunique()} unique auctions")
