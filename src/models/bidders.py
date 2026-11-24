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


class ExpectedRevenueBidder:
    """Each week, bid the price with the highest (price x chance of winning).

    The chance of winning with factor f is estimated from past weeks of the same
    direction, published before the deadline:
        P(win) = share of past weeks where  marginal / last_price > f
    Weeks that came after a similar price move (last_price against the one before)
    count more: a fall is followed differently than a jump.

    Bid = best factor x last price. Nothing is frozen: every new published week
    joins the history, so fit() only stores it. `similarity` was chosen by
    walk-forward on the learning years 2021-2024.
    """

    name = "expected_revenue"

    def __init__(self, similarity: float = 0.1, min_history: int = 26, grid=None):
        self.similarity = similarity      # width, in log price move; small = only very similar weeks
        self.min_history = min_history    # past weeks needed before the first bid
        self.grid = np.round(np.arange(0.30, 1.5001, 0.01), 2) if grid is None else np.asarray(grid)

    def fit(self, weekly: pd.DataFrame) -> ExpectedRevenueBidder:
        """Store the history. Later weeks may be in it: bid() only reads avail <= gc."""
        history = weekly.sort_values("gc").copy()
        before = history.groupby("direction")["last_price"].shift(1)
        history["move"] = np.log(history["last_price"] / before)
        history["ratio"] = history["marginal"] / history["last_price"]
        self.history_ = history
        return self

    def bid(self, auctions: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "history_"):
            raise RuntimeError("fit() was not called")
        unknown = set(auctions["auction_id"]) - set(self.history_["auction_id"])
        if unknown:
            raise ValueError(f"{len(unknown)} auctions were not given to fit()")
        by_id = self.history_.set_index("auction_id")
        prices = [self._bid_one(by_id.loc[a]) for a in auctions["auction_id"]]
        return _as_bids(auctions, pd.Series(prices, index=auctions.index, dtype=float))

    def _bid_one(self, auction: pd.Series) -> float:
        h = self.history_
        past = h[(h["direction"] == auction["direction"]) & (h["avail"] <= auction["gc"])
                 & h["ratio"].notna()]
        if len(past) < self.min_history:
            raise ValueError(f"{auction.name}: only {len(past)} known past weeks, "
                             f"{self.min_history} needed")
        distance = (past["move"] - auction["move"]) / self.similarity
        # the small constant keeps every past week in play when none is similar
        weight = np.exp(-0.5 * distance**2).fillna(0.0).to_numpy() + 1e-3
        wins = past["ratio"].to_numpy()[:, None] > self.grid
        p_win = (wins * weight[:, None]).sum(axis=0) / weight.sum()
        return float(self.grid[np.argmax(self.grid * p_win)] * auction["last_price"])


def _as_bids(auctions: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    if prices.isna().any():
        raise ValueError("some auctions have no known last price or no factor for their direction")
    return pd.DataFrame({"auction_id": auctions["auction_id"], "bid": prices}).reset_index(drop=True)
