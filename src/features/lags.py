from dataclasses import dataclass

import pandas as pd


@dataclass
class LeakageSafeLag:
    """Add y_lag1: the y of the most recent KNOWN auction of the same series.

    "Known" means avail <= gc of the auction to predict, so auctions closing
    in the same sitting never see each other (avail = gc + 2h).
    No-buy auctions (y NaN) are skipped: the lag is the last auction with a price.
    """

    series_col: str = "series"
    time_col: str = "gc"
    avail_col: str = "avail"
    target_col: str = "y"

    def compute(self, auctions: pd.DataFrame) -> pd.DataFrame:
        # left: auctions to enrich, sorted on gc (merge key)
        left = auctions.sort_values(self.time_col).reset_index(drop=True)

        # right: known auctions (with a target), sorted on avail
        right = (
            auctions.dropna(subset=[self.target_col])
                    .sort_values(self.avail_col)
                    .reset_index(drop=True)
        )

        lag = pd.merge_asof(
            left[["auction_id", self.series_col, self.time_col]],
            right[[self.series_col, self.avail_col, self.target_col]]
            .rename(columns={self.target_col: "y_lag1"}),
            left_on=self.time_col,      # gc of the auction to predict
            right_on=self.avail_col,    # availability of past auctions
            by=self.series_col,         # same series only
            direction="backward",       # most recent with avail <= gc
        )
        # merge_asof does not keep the input order: merge back on auction_id
        return auctions.merge(
            lag[["auction_id", "y_lag1"]], on="auction_id", how="left"
        )
