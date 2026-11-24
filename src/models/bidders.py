"""Simple bidders for the weekly mFRR auctions (1 MW, one bid per auction).

Every bidder only knows marginal prices already published at the auction's
gate closure (avail <= gc), and treats upwards and downwards separately.

Years (ISO years, as in src.evaluation.revenue):
    2019-2024  learn
    2025       compare versions and choose settings
    2026       final test, scored only once, at the end

Usage:
    weekly = add_last_price(select_weekly(build_auctions()))
    bidder = DiscountBidder().fit(weekly)
    bids = bidder.bid(weekly[weekly["year"] == 2025])   # auction_id, bid
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.revenue import score_bids, summarise
from src.features.lags import LeakageSafeLag

LAST_LEARN_YEAR = 2024
CHOOSE_YEAR = 2025
FINAL_YEAR = 2026


def add_last_price(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add year and last_price, the latest marginal price known at gate closure.

    Pass ALL weekly auctions: the lag needs the history. The first auction of
    each direction has no last price (NaN).
    """
    # lag the price itself, not log10: 10**log10(x) can land just below x, and a
    # bid equal to the marginal price would then wrongly win
    out = LeakageSafeLag(target_col="marginal").compute(weekly).rename(
        columns={"y_lag1": "last_price"})
    out["year"] = out["date"].dt.isocalendar().year.astype(int)
    return out


class LastPriceBidder:
    """Bid the last known marginal price."""

    name = "last_price"

    def fit(self, weekly: pd.DataFrame) -> LastPriceBidder:
        return self  # nothing to learn

    def bid(self, auctions: pd.DataFrame) -> pd.DataFrame:
        return _as_bids(auctions, auctions["last_price"])


class DiscountBidder:
    """Bid factor x last known price, with one factor per direction.

    fit() picks the factor with the best average yearly % of the best possible
    revenue over the learning years. Each year counts the same, so the very
    expensive years do not decide alone. Later years are ignored by fit().

    Set by fit: factors_ ({direction: factor})
    """

    name = "discount"

    def __init__(self, grid=None, last_learn_year: int = LAST_LEARN_YEAR):
        self.grid = np.round(np.arange(0.30, 1.1001, 0.01), 2) if grid is None else grid
        self.last_learn_year = last_learn_year

    def fit(self, weekly: pd.DataFrame) -> DiscountBidder:
        learn = weekly[(weekly["year"] <= self.last_learn_year) & weekly["last_price"].notna()]
        if learn.empty:
            raise ValueError("no auction to learn from")
        self.factors_ = {}
        for direction, g in learn.groupby("direction"):
            quality = [self._mean_yearly_pct(g, f) for f in self.grid]
            self.factors_[direction] = float(self.grid[int(np.argmax(quality))])
        return self

    def bid(self, auctions: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "factors_"):
            raise RuntimeError("fit() was not called")
        return _as_bids(auctions, auctions["last_price"] * auctions["direction"].map(self.factors_))

    @staticmethod
    def _mean_yearly_pct(auctions: pd.DataFrame, factor: float) -> float:
        bids = _as_bids(auctions, factor * auctions["last_price"])
        return summarise(score_bids(auctions, bids))["pct_of_best"].mean()


def _as_bids(auctions: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    if prices.isna().any():
        raise ValueError("some auctions have no known last price or no factor for their direction")
    return pd.DataFrame({"auction_id": auctions["auction_id"], "bid": prices}).reset_index(drop=True)
