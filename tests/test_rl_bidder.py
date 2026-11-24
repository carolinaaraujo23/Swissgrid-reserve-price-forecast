"""Tests of the reinforcement-learning weekly bidder (src/models/rl_bidder.py).

Run with: uv run pytest tests/test_rl_bidder.py
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.revenue import score_bids, select_weekly
from src.models.bidders import CHOOSE_YEAR, add_last_price
from src.models.rl_bidder import RLBidder

FEEDBACK = ["outcome", "published"]


def make_weekly(prices, direction="upwards", start="2023-01-02") -> pd.DataFrame:
    """One weekly auction per price, in order. Gate closure: Tuesday 12:00 UTC of the week before."""
    dates = pd.date_range(start, periods=len(prices), freq="7D")
    gc = (dates - pd.Timedelta(days=6) + pd.Timedelta(hours=12)).tz_localize("UTC")
    return pd.DataFrame({
        "auction_id": [f"{direction}_{i}" for i in range(len(prices))],
        "date": dates, "direction": direction, "freq": "weekly",
        "series": f"weekly|{direction}|week", "gc": gc,
        "avail": gc + pd.Timedelta(hours=2), "marginal": [float(p) for p in prices]})


def factors_played(weekly: pd.DataFrame, bidder: RLBidder) -> pd.Series:
    """Factor (bid / last price) of every auction after the learning years."""
    table = add_last_price(weekly)
    later = table[table["year"] > bidder.last_learn_year].set_index("auction_id")
    bids = bidder.fit(table).bid(later.reset_index()).set_index("auction_id")["bid"]
    return bids / later["last_price"]


# --- does it learn? -----------------------------------------------------------------

@pytest.mark.parametrize("feedback", FEEDBACK)
def test_learns_the_best_factor(feedback):
    # price always falls 10 %: 0.85 always wins, 0.95 never, 0.5 wins but earns little
    weekly = make_weekly(1e9 * 0.9 ** np.arange(60), start="2024-01-01")
    bidder = RLBidder(feedback=feedback, factors=[0.5, 0.85, 0.95])
    assert factors_played(weekly, bidder).round(6).eq(0.85).all()


@pytest.mark.parametrize("feedback", FEEDBACK)
def test_uses_how_the_price_moved(feedback):
    # a jump (x2) is always followed by a halving, and a halving by a jump
    prices = 100 * np.cumprod([1.0] + [2.0, 0.5] * 35)
    weekly = make_weekly(prices, start="2024-01-01")
    played = factors_played(weekly, RLBidder(feedback=feedback, factors=[0.45, 1.5]))
    moves = add_last_price(weekly).set_index("auction_id")["last_price"].pct_change()
    assert played[moves[played.index] > 0].round(6).eq(0.45).all()   # after a jump: bid low
    assert played[moves[played.index] < 0].round(6).eq(1.5).all()    # after a fall: bid high


@pytest.mark.parametrize("feedback", FEEDBACK)
def test_directions_learn_separately(feedback):
    up = make_weekly(1e9 * 0.9 ** np.arange(60), start="2024-01-01")
    down = make_weekly(1e9 * 0.6 ** np.arange(60), "downwards", start="2024-01-01")
    played = factors_played(pd.concat([up, down], ignore_index=True),
                            RLBidder(feedback=feedback, factors=[0.55, 0.85]))
    assert played[played.index.str.startswith("upwards")].round(6).eq(0.85).all()
    assert played[played.index.str.startswith("downwards")].round(6).eq(0.55).all()


def test_equal_to_the_marginal_earns_nothing():
    bidder = RLBidder(feedback="published", factors=[0.8, 0.9, 1.0])
    q, count = np.zeros((3, 3)), np.zeros((3, 3))
    bidder._update(q, count, 0, 0, ratio=0.9)        # marginal = 0.9 x last price
    assert q[0].tolist() == [0.8, 0.0, 0.0]


def test_outcome_learns_only_from_its_own_bid():
    bidder = RLBidder(feedback="outcome", factors=[0.8, 0.9, 1.0])
    q, count = np.zeros((3, 3)), np.zeros((3, 3))
    bidder._update(q, count, 0, 1, ratio=0.95)
    assert q[0].tolist() == [0.0, 0.9, 0.0]
    assert count[0].tolist() == [0, 1, 0]


def test_average_and_fixed_step():
    average, fixed = RLBidder(factors=[0.8]), RLBidder(factors=[0.8], step=0.5)
    for bidder, expected in ((average, 0.8 * 2 / 3), (fixed, 0.3)):
        q, count = np.zeros((1, 1)), np.zeros((1, 1))
        for ratio in (1.0, 1.0, 0.5):                # win, win, lose
            bidder._update(q, count, 0, 0, ratio)
        assert q[0, 0] == pytest.approx(expected)


# --- wrong use ------------------------------------------------------------------------

def test_wrong_use_raises():
    weekly = add_last_price(make_weekly(100 * 0.95 ** np.arange(60), start="2024-01-01"))
    with pytest.raises(ValueError, match="feedback"):
        RLBidder(feedback="magic")
    with pytest.raises(RuntimeError, match="fit"):
        RLBidder().bid(weekly)
    with pytest.raises(ValueError, match="not played"):
        RLBidder().fit(weekly).bid(weekly.iloc[[10]])            # a learning-year auction
    with pytest.raises(ValueError, match="no auction to learn from"):
        RLBidder(last_learn_year=2000).fit(weekly)


# --- real data: reproducible, no future information ----------------------------------------

@pytest.fixture(scope="module")
def weekly(mfrr_auctions):
    return select_weekly(mfrr_auctions)


def bids_until(weekly: pd.DataFrame, cutoff: pd.Timestamp, feedback: str, passes: int = 5) -> pd.DataFrame:
    table = add_last_price(weekly)
    due = table[(table["year"] == CHOOSE_YEAR) & (table["gc"] <= cutoff)]
    return RLBidder(feedback=feedback, passes=passes).fit(table).bid(due)


def closing(weekly: pd.DataFrame, week: int) -> pd.Timestamp:
    closings = weekly.loc[weekly["date"].dt.isocalendar().year == CHOOSE_YEAR, "gc"].unique()
    return pd.Series(closings).sort_values().iloc[week]


def scramble_unpublished(weekly: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    out = weekly.copy()
    hidden = out["avail"] > cutoff
    out.loc[hidden, "marginal"] = np.random.default_rng(0).uniform(1, 1e5, hidden.sum())
    return out


@pytest.mark.parametrize("feedback", FEEDBACK)
@pytest.mark.parametrize("week", [0, 10, 51])
def test_bids_do_not_use_the_future(weekly, feedback, week):
    cutoff = closing(weekly, week)
    before = bids_until(weekly, cutoff, feedback)
    assert len(before) == 2 * (week + 1)
    pd.testing.assert_frame_equal(
        before, bids_until(scramble_unpublished(weekly, cutoff), cutoff, feedback))


def test_result_is_not_used_before_it_is_published():
    """Each result comes out one second after the next deadline: the agent must wait for it."""
    prices = np.r_[1e6 * 0.9 ** np.arange(53), 1e6 * 0.9 ** 52 * 3.0 ** np.arange(1, 8)]
    table = add_last_price(make_weekly(prices, start="2024-01-01"))
    late = table.copy()
    late["avail"] = (late["gc"].shift(-1, fill_value=late["gc"].max() + pd.Timedelta(weeks=1))
                     + pd.Timedelta(seconds=1))

    def first_high_bid(weekly):
        # large fixed step: one seen result (price x3) is enough to raise the next bid
        agent = RLBidder("published", n_states=1, step=0.9, factors=[0.85, 2.5]).fit(weekly)
        last_price = weekly.set_index("auction_id")["last_price"]
        return next(k for k, (a, bid) in enumerate(agent.bids_.items()) if bid > last_price[a])

    assert first_high_bid(late) == first_high_bid(table) + 1


def test_leak_check_catches_a_cheating_agent(weekly):
    cutoff = closing(weekly, 0)
    cheating = weekly.assign(avail=weekly["gc"])
    scrambled = scramble_unpublished(weekly, cutoff).assign(avail=weekly["gc"])
    assert not bids_until(cheating, cutoff, "published").equals(bids_until(scrambled, cutoff, "published"))


@pytest.mark.parametrize("feedback", FEEDBACK)
def test_practice_ignores_the_choosing_and_final_years(weekly, feedback):
    table = add_last_price(weekly)
    changed = table.copy()
    changed.loc[changed["year"] >= CHOOSE_YEAR, "marginal"] *= 7.0
    a, b = (RLBidder(feedback=feedback, passes=5).fit(t).q_ for t in (table, changed))
    for direction in a:
        np.testing.assert_array_equal(a[direction], b[direction])


def test_same_seed_same_bids(weekly):
    cutoff = closing(weekly, 51)
    pd.testing.assert_frame_equal(bids_until(weekly, cutoff, "outcome"),
                                  bids_until(weekly, cutoff, "outcome"))


@pytest.mark.parametrize("feedback", FEEDBACK)
def test_beats_bidding_the_last_price(weekly, feedback):
    table = add_last_price(weekly)
    due = table[table["year"] == CHOOSE_YEAR]
    agent = score_bids(due, RLBidder(feedback=feedback).fit(table).bid(due))
    naive = score_bids(due, due[["auction_id"]].assign(bid=due["last_price"]))
    for direction in ("upwards", "downwards"):
        assert (agent.loc[agent["direction"] == direction, "revenue"].sum()
                > naive.loc[naive["direction"] == direction, "revenue"].sum())
