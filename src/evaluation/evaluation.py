from dataclasses import dataclass
from typing import Iterator

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_pinball_loss
import numpy as np
import pandas as pd


class Metrics:
    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)

    @staticmethod
    def pinball(y_true, y_pred, q):
        return mean_pinball_loss(y_true, y_pred, alpha=q)

    @staticmethod
    def coverage(y_true, lo, hi):
        """coverage = (number of auctions where y_true ∈ [p10, p90]) / (total number of auctions)
        the target should be 80% : <80 model too confident, >80 model to careful on pred"""

        y_true, lo, hi = np.asarray(y_true), np.asarray(lo), np.asarray(hi)
        return float(np.mean((y_true >= lo) & (y_true <= hi)))


@dataclass
class WalkForwardSplitter:
    """Separate test and train in walkforward fashion according to frequency """

    start_test: str = "2025-01-01"
    end_test: str | None = None  # None = until last line
    freq: str = "M"              # "M" monthly, "W" weekly, "Q" quarterly
    avail_col: str = "avail"  # availability of results of auction
    gc_col: str = "gc"  # limit date to pulish offer

    def _to_utc(self, s):
        ts = pd.Timestamp(s)
        return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

    def splits(self, df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate (train, test) for each test period"""
        t0 = self._to_utc(self.start_test)  # beg test period
        t_end = self._to_utc(
            self.end_test) if self.end_test else df[self.gc_col].max()  # end of test period, if None, put last line

        # Generate periods according to freq
        periods = pd.period_range(t0, t_end, freq=self.freq)
        for period in periods:
            # period is a pd.Period taht we convert in naive timestamps then utc
            p_start = period.start_time.tz_localize("UTC")
            # avoid 23:59:59.999999999
            p_end = (period + 1).start_time.tz_localize("UTC")

            train = df.loc[df[self.avail_col] < p_start, :]
            test_mask = (df[self.gc_col] >= p_start) & (
                df[self.gc_col] < p_end)
            test = df.loc[test_mask, :]

            if test.empty:
                continue
            yield train, test


@dataclass
class LeakageSafeLag:
    def __init__(self, series_col="series", time_col="gc",
                 avail_col="avail", target_col="y"):
        self.series_col = series_col
        self.time_col = time_col
        self.avail_col = avail_col
        self.target_col = target_col

    def compute(self, auctions: pd.DataFrame) -> pd.DataFrame:
        # gauche : enchères à enrichir, triées par gc (clé de fusion)
        left = auctions.sort_values(self.time_col).reset_index(drop=True)

        # droite : enchères "connues" (avec cible), triées par avail
        right = (
            auctions.dropna(subset=[self.target_col])
                    .sort_values(self.avail_col)
                    .reset_index(drop=True)
        )

        lag = pd.merge_asof(
            left[["auction_id", self.series_col, self.time_col]],
            right[[self.series_col, self.avail_col, self.target_col]]
            .rename(columns={self.target_col: "y_lag1"}),
            left_on=self.time_col,      # gc de l'enchère à prédire
            right_on=self.avail_col,    # disponibilité des enchères passées
            by=self.series_col,         # même série uniquement
            direction="backward",       # la plus récente avec avail <= gc
        )
        # merge_asof ne garantit pas l'ordre de sortie → on réinjecte par id
        return auctions.merge(
            lag[["auction_id", "y_lag1"]], on="auction_id", how="left"
        )
