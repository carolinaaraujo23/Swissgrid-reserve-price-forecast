"""Tests of the simple weekly bidders (src/models/bidders.py).

Run with: uv run pytest tests/test_bidders.py
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.revenue import score_bids, select_weekly
from src.models.bidders import (
    CHOOSE_YEAR,
    FINAL_YEAR,
    LAST_LEARN_YEAR,
    DiscountBidder,
    ExpectedRevenueBidder,
    LastPriceBidder,
    add_last_price,
)


def make_weekly(prices, direction="upwards", start="2023-01-02") -> pd.DataFrame:
    """One weekly auction per price, in order. Gate closure: Tuesday 12:00 UTC of the week before."""
    dates = pd.date_range(start, periods=len(prices), freq="7D")
    gc = (dates - pd.Timedelta(days=6) + pd.Timedelta(hours=12)).tz_localize("UTC")
    return pd.DataFrame({
        "auction_id": [f"{direction}_{i}" for i in range(len(prices))],
        "date": dates, "direction": direction, "freq": "weekly",
        "series": f"weekly|{direction}|week", "gc": gc,
        "avail": gc + pd.Timedelta(hours=2), "marginal": [float(p) for p in prices]})


# --- last known price -------------------------------------------------------------

def test_last_price_is_the_previous_week():
    out = add_last_price(make_weekly([100, 90, 120]))
    assert np.isnan(out["last_price"].iloc[0])
    assert out["last_price"].iloc[1:].tolist() == [100.0, 90.0]


def test_last_price_never_crosses_directions():
    both = pd.concat([make_weekly([100, 90]), make_weekly([1000, 900], "downwards")],
                     ignore_index=True)
    out = add_last_price(both).set_index("auction_id")["last_price"]
    assert out["upwards_1"] == 100.0
    assert out["downwards_1"] == 1000.0


def test_last_price_not_published_yet_is_skipped():
    weekly = make_weekly([100, 90, 120])
    weekly.loc[1, "avail"] = weekly.loc[2, "gc"] + pd.Timedelta(seconds=1)  # one second too late
    assert add_last_price(weekly)["last_price"].iloc[2] == 100.0


@pytest.mark.parametrize("price", [3.0, 264.4, 399.0, 913.07, 5013.33])
def test_unchanged_price_loses_for_last_price_bidder(price):
    """The bid equals the marginal exactly: no rounding (log10) may turn it into a win."""
    out = add_last_price(make_weekly([price, price])).iloc[[1]]
    assert not score_bids(out, LastPriceBidder().bid(out))["won"].any()


def test_bidding_without_a_last_price_raises():
    out = add_last_price(make_weekly([100, 90]))
    with pytest.raises(ValueError, match="no known last price"):
        LastPriceBidder().bid(out)


# --- discount bidder ----------------------------------------------------------------

def test_discount_bids_factor_times_last_price():
    weekly = add_last_price(make_weekly([100, 90, 120])).iloc[1:]
    bidder = DiscountBidder()
    bidder.factors_ = {"upwards": 0.8}
    assert bidder.bid(weekly)["bid"].tolist() == pytest.approx([80.0, 72.0])


def test_fit_finds_the_best_factor_per_direction():
    # price always falls by 10 % (up) or 30 % (down): the best grid factor is just below
    up = make_weekly(1000 * 0.9 ** np.arange(20))
    down = make_weekly(1000 * 0.7 ** np.arange(20), "downwards")
    weekly = add_last_price(pd.concat([up, down], ignore_index=True))
    bidder = DiscountBidder(grid=np.array([0.5, 0.65, 0.69, 0.71, 0.85, 0.89, 0.91])).fit(weekly)
    assert bidder.factors_ == {"upwards": 0.89, "downwards": 0.69}


def test_every_year_counts_the_same():
    # 2022: cheap, falls 50 % a week. 2023: 1000x more expensive, falls 10 % a week.
    cheap = make_weekly(100 * 0.5 ** np.arange(10), start="2022-03-07")
    dear = make_weekly(1e5 * 0.9 ** np.arange(10), start="2023-03-06")
    dear["auction_id"] += "_dear"
    weekly = add_last_price(pd.concat([cheap, dear], ignore_index=True))
    weekly = weekly[~weekly["auction_id"].isin(["upwards_0_dear"])]
    # 0.49 wins in both years (98 % and ~54 %), 0.89 only in the expensive one (0 % and ~99 %)
    bidder = DiscountBidder(grid=np.array([0.49, 0.89])).fit(weekly)
    assert bidder.factors_ == {"upwards": 0.49}


def test_bidding_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        DiscountBidder().bid(add_last_price(make_weekly([100, 90])).iloc[1:])


def test_fit_without_learning_years_raises():
    weekly = add_last_price(make_weekly([100, 90, 80], start="2025-03-03"))
    with pytest.raises(ValueError, match="no auction to learn from"):
        DiscountBidder().fit(weekly)


# --- expected-revenue bidder ----------------------------------------------------------

def last_bid(prices, **kwargs) -> float:
    """Bid for the last week of a hand-made price path, as a factor of the last price."""
    weekly = add_last_price(make_weekly(prices))
    target = weekly.iloc[[-1]]
    bid = ExpectedRevenueBidder(min_history=3, **kwargs).fit(weekly).bid(target)["bid"].iloc[0]
    return bid / target["last_price"].iloc[0]


def test_picks_highest_price_times_chance_of_winning():
    # past ratios: 0.9 four times out of five, 0.5 once (always the same move before: ignored here)
    ratios = [0.9, 0.9, 0.9, 0.9, 0.5] * 6
    prices = 1e9 * np.cumprod([1.0] + ratios + [0.9])
    # 0.89 wins 80 % of the time -> 0.71 expected; 0.49 always wins -> 0.49; 1.0 never wins
    assert last_bid(prices, similarity=1e6, grid=[0.49, 0.89, 1.0]) == pytest.approx(0.89)


def test_bids_low_when_high_prices_rarely_win():
    ratios = [0.9, 0.5, 0.5, 0.5, 0.5] * 6
    prices = 1e9 * np.cumprod([1.0] + ratios + [0.9])
    # 0.89 wins 20 % -> 0.18; 0.49 always wins -> 0.49
    assert last_bid(prices, similarity=1e6, grid=[0.49, 0.89, 1.0]) == pytest.approx(0.49)


def test_similar_weeks_count_more():
    # a jump (x2) is always followed by a fall (x0.5), and a fall by a jump
    up_then_down = [2.0, 0.5] * 10
    after_jump = 100 * np.cumprod([1.0] + up_then_down + [2.0, 0.5])
    after_fall = 100 * np.cumprod([1.0] + up_then_down + [2.0])
    assert last_bid(after_jump, grid=[0.49, 1.49]) == pytest.approx(0.49)
    assert last_bid(after_fall, grid=[0.49, 1.49]) == pytest.approx(1.49)


def test_expected_revenue_never_learns_from_the_other_direction():
    up = make_weekly(1000 * 0.9 ** np.arange(20))
    down = make_weekly(1000 * 0.5 ** np.arange(20), "downwards")
    alone = add_last_price(up)
    both = add_last_price(pd.concat([up, down], ignore_index=True))
    target = alone.iloc[[-1]]
    # similarity off: otherwise the other direction's weeks would hardly count anyway
    bids = [ExpectedRevenueBidder(similarity=1e6, min_history=3).fit(table).bid(target)["bid"].iloc[0]
            for table in (alone, both)]
    assert bids[0] == bids[1]


def test_expected_revenue_needs_enough_history():
    weekly = add_last_price(make_weekly([100, 90, 80, 70]))
    with pytest.raises(ValueError, match="needed"):
        ExpectedRevenueBidder(min_history=26).fit(weekly).bid(weekly.iloc[[-1]])


def test_expected_revenue_before_fit_or_unknown_auction_raises():
    weekly = add_last_price(make_weekly([100, 90, 80, 70, 60]))
    with pytest.raises(RuntimeError, match="fit"):
        ExpectedRevenueBidder().bid(weekly.iloc[[-1]])
    with pytest.raises(ValueError, match="not given to fit"):
        ExpectedRevenueBidder(min_history=2).fit(weekly.iloc[:-1]).bid(weekly.iloc[[-1]])


def test_bids_do_not_depend_on_which_auctions_are_asked_together(mfrr_auctions):
    table = add_last_price(select_weekly(mfrr_auctions))
    due = table[table["year"] == CHOOSE_YEAR]
    bidder = ExpectedRevenueBidder().fit(table)
    together = bidder.bid(due).set_index("auction_id")["bid"]
    alone = bidder.bid(due.iloc[[30]]).set_index("auction_id")["bid"]
    assert together[alone.index[0]] == alone.iloc[0]


# --- no future information (real data) --------------------------------------------------

COARSE = np.array([0.6, 0.7, 0.8, 0.9, 1.0])   # enough to detect a leak, and fast

@pytest.fixture(scope="module")
def weekly(mfrr_auctions):
    return select_weekly(mfrr_auctions)


def scramble_unpublished(weekly: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Replace every price not yet published at `cutoff` with random numbers."""
    out = weekly.copy()
    hidden = out["avail"] > cutoff
    out.loc[hidden, "marginal"] = np.random.default_rng(0).uniform(1, 1e5, hidden.sum())
    return out


def bids_until(weekly: pd.DataFrame, cutoff: pd.Timestamp, year: int = CHOOSE_YEAR) -> pd.DataFrame:
    """Bids of all bidders, learning included, for auctions of `year` closing by `cutoff`."""
    table = add_last_price(weekly)
    due = table[(table["year"] == year) & (table["gc"] <= cutoff)]
    bids = [b.fit(table).bid(due) for b in (LastPriceBidder(), DiscountBidder(COARSE), ExpectedRevenueBidder())]
    return pd.concat(bids, ignore_index=True)


@pytest.mark.parametrize("week", [0, 1, 10, 25, 51])
def test_bids_do_not_use_the_future(weekly, week):
    closings = weekly.loc[weekly["date"].dt.isocalendar().year == CHOOSE_YEAR, "gc"].unique()
    cutoff = pd.Series(closings).sort_values().iloc[week]
    before = bids_until(weekly, cutoff)
    after = bids_until(scramble_unpublished(weekly, cutoff), cutoff)
    assert len(before) == 3 * 2 * (week + 1)   # three bidders, two directions
    pd.testing.assert_frame_equal(before, after)


def test_leak_check_catches_a_cheating_bidder(weekly):
    """Results treated as known at gate closure: the bidder sees its own auction's price."""
    cheating = weekly.assign(avail=weekly["gc"])
    cutoff = weekly.loc[weekly["date"].dt.isocalendar().year == CHOOSE_YEAR, "gc"].iloc[0]
    scrambled = scramble_unpublished(weekly, cutoff).assign(avail=weekly["gc"])
    assert not bids_until(cheating, cutoff).equals(bids_until(scrambled, cutoff))


def test_learning_ignores_the_choosing_and_final_years(weekly):
    table = add_last_price(weekly)
    changed = table.copy()
    later = changed["year"] > LAST_LEARN_YEAR
    changed.loc[later, ["marginal", "last_price"]] *= 7.0
    assert DiscountBidder(COARSE).fit(table).factors_ == DiscountBidder(COARSE).fit(changed).factors_


def test_split_years():
    assert LAST_LEARN_YEAR < CHOOSE_YEAR < FINAL_YEAR
    assert (LAST_LEARN_YEAR, CHOOSE_YEAR, FINAL_YEAR) == (2024, 2025, 2026)


def test_only_the_first_auction_of_each_direction_has_no_last_price(weekly):
    table = add_last_price(weekly)
    missing = table[table["last_price"].isna()]
    assert sorted(missing["direction"]) == ["downwards", "upwards"]
    assert missing["gc"].nunique() == 1
    assert missing["gc"].iloc[0] == weekly["gc"].min()
