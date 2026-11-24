"""Yearly revenue of one bidder in the weekly mFRR auctions.

The bidder submits exactly one bid of 1 MW in every weekly auction:
    bid <  marginal price  -> wins and earns its own bid (CHF, 1 MW for the week)
    bid >= marginal price  -> loses and earns 0

This only scores bids against the real marginal prices. It does not simulate
the other bidders, and the bid does not move the marginal price.

Usage:
    weekly = select_weekly(build_auctions())
    scored = score_bids(weekly, bids)     # bids: columns auction_id, bid
    print(summarise(scored))              # one row per year and direction
"""

import numpy as np
import pandas as pd


def select_weekly(auctions: pd.DataFrame) -> pd.DataFrame:
    """Weekly auctions, one per week and direction (2019 correction auction dropped)."""
    weekly = auctions[(auctions["freq"] == "weekly")
                      & ~auctions["auction_id"].str.contains("KORR")]
    return weekly.sort_values(["gc", "direction"]).reset_index(drop=True)


def score_bids(auctions: pd.DataFrame, bids: pd.DataFrame) -> pd.DataFrame:
    """Score one bid per auction.

    auctions : auctions to score (auction_id, date, direction, marginal)
    bids     : columns auction_id and bid (CHF/MW)

    Returns one row per auction with: won, revenue, best_possible.
    Every auction needs exactly one valid bid: a forgotten bid raises an error
    instead of silently counting as a lost auction.
    """
    _check(auctions, bids)
    scored = auctions[["auction_id", "date", "direction", "marginal"]].merge(
        bids[["auction_id", "bid"]], on="auction_id")
    # ISO year, as in Swissgrid's auction_id: the week of 30 Dec 2024 is week 1 of 2025
    scored["year"] = scored["date"].dt.isocalendar().year.astype(int)
    scored["won"] = scored["bid"] < scored["marginal"]
    scored["revenue"] = scored["bid"].where(scored["won"], 0.0)
    # ceiling: a bid can get as close as it wants to the marginal, never reach it
    scored["best_possible"] = scored["marginal"]
    return scored


def summarise(scored: pd.DataFrame, by: tuple[str, ...] = ("year", "direction")) -> pd.DataFrame:
    """Revenue, win rate and % of the best possible revenue.

    Directions are never mixed: "direction" must be in `by`.
    An unfinished year has fewer auctions, so compare years with pct_of_best.
    """
    if "direction" not in by:
        raise ValueError("upwards and downwards must be scored separately")
    table = scored.groupby(list(by)).agg(
        auctions=("auction_id", "size"), wins=("won", "sum"),
        win_rate=("won", "mean"), revenue=("revenue", "sum"),
        best_possible=("best_possible", "sum")).reset_index()
    table["pct_of_best"] = 100 * table["revenue"] / table["best_possible"]
    return table


def _check(auctions: pd.DataFrame, bids: pd.DataFrame) -> None:
    if auctions["auction_id"].duplicated().any():
        raise ValueError("duplicated auction_id in auctions")
    if auctions["marginal"].isna().any():
        raise ValueError("some auctions have no marginal price and cannot be scored")
    if bids["auction_id"].duplicated().any():
        raise ValueError("more than one bid for the same auction")
    missing = set(auctions["auction_id"]) - set(bids["auction_id"])
    if missing:
        raise ValueError(f"{len(missing)} auctions have no bid, e.g. {sorted(missing)[:3]}")
    unknown = set(bids["auction_id"]) - set(auctions["auction_id"])
    if unknown:
        raise ValueError(f"{len(unknown)} bids match no auction, e.g. {sorted(unknown)[:3]}")
    if not (np.isfinite(bids["bid"]) & (bids["bid"] > 0)).all():
        raise ValueError("every bid must be a positive number")
