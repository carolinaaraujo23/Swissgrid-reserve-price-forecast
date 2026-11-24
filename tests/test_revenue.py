"""Tests of the weekly bid revenue scorer (src/evaluation/revenue.py).

Run with: uv run pytest tests/test_revenue.py
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.revenue import score_bids, select_weekly, summarise


def make_auctions(rows) -> pd.DataFrame:
    """rows: list of (auction_id, delivery date, direction, marginal price)"""
    df = pd.DataFrame(rows, columns=["auction_id", "date", "direction", "marginal"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def make_bids(**bids) -> pd.DataFrame:
    return pd.DataFrame({"auction_id": list(bids), "bid": list(bids.values())})


@pytest.fixture
def three_weeks() -> pd.DataFrame:
    return make_auctions([
        ("a", "2025-01-06", "upwards", 100.0),
        ("b", "2025-01-13", "upwards", 100.0),
        ("c", "2025-01-20", "upwards", 80.0),
    ])


# --- winning rule ---------------------------------------------------------------

@pytest.mark.parametrize("bid, won, revenue", [
    (90.0, True, 90.0),      # below: wins and earns its own bid, not the marginal price
    (99.99, True, 99.99),    # just below still wins
    (100.0, False, 0.0),     # equal: loses
    (100.01, False, 0.0),    # just above loses
    (500.0, False, 0.0),
])
def test_wins_only_strictly_below_marginal(bid, won, revenue):
    auctions = make_auctions([("a", "2025-01-06", "upwards", 100.0)])
    scored = score_bids(auctions, make_bids(a=bid)).iloc[0]
    assert scored["won"] == won
    assert scored["revenue"] == revenue


def test_yearly_summary_of_hand_made_example(three_weeks):
    scored = score_bids(three_weeks, make_bids(a=90.0, b=100.0, c=70.0))
    row = summarise(scored).iloc[0]
    assert (row["auctions"], row["wins"]) == (3, 2)
    assert row["win_rate"] == pytest.approx(2 / 3)
    assert row["revenue"] == 160.0
    assert row["best_possible"] == 280.0
    assert row["pct_of_best"] == pytest.approx(100 * 160 / 280)


def test_bidding_the_marginal_price_every_week_earns_nothing(three_weeks):
    scored = score_bids(three_weeks, make_bids(a=100.0, b=100.0, c=80.0))
    assert summarise(scored)["revenue"].iloc[0] == 0.0


def test_order_of_bids_does_not_matter(three_weeks):
    bids = make_bids(a=90.0, b=95.0, c=70.0)
    shuffled = bids.iloc[[2, 0, 1]]
    pd.testing.assert_frame_equal(score_bids(three_weeks, bids),
                                  score_bids(three_weeks, shuffled))


# --- directions and years ---------------------------------------------------------

def test_directions_are_scored_separately():
    auctions = make_auctions([
        ("up", "2025-01-06", "upwards", 100.0),
        ("down", "2025-01-06", "downwards", 1000.0),
    ])
    table = summarise(score_bids(auctions, make_bids(up=50.0, down=2000.0)))
    revenue = table.set_index("direction")["revenue"].to_dict()
    assert revenue == {"upwards": 50.0, "downwards": 0.0}


def test_mixing_directions_raises(three_weeks):
    scored = score_bids(three_weeks, make_bids(a=90.0, b=90.0, c=70.0))
    with pytest.raises(ValueError, match="separately"):
        summarise(scored, by=("year",))


@pytest.mark.parametrize("date, year", [
    ("2019-12-30", 2020),
    ("2024-12-23", 2024),
    ("2024-12-30", 2025),    # week 1 of 2025, called TRL_25_KW01 by Swissgrid
    ("2025-12-29", 2026),
])
def test_week_crossing_new_year_belongs_to_one_year(date, year):
    auctions = make_auctions([("a", date, "upwards", 100.0)])
    assert score_bids(auctions, make_bids(a=90.0))["year"].iloc[0] == year


# --- wrong inputs must raise, never count as a lost auction ---------------------------

def test_missing_bid_raises(three_weeks):
    with pytest.raises(ValueError, match="no bid"):
        score_bids(three_weeks, make_bids(a=90.0, b=90.0))


def test_duplicate_bid_raises(three_weeks):
    bids = pd.concat([make_bids(a=90.0, b=90.0, c=70.0), make_bids(a=80.0)])
    with pytest.raises(ValueError, match="more than one bid"):
        score_bids(three_weeks, bids)


def test_bid_for_unknown_auction_raises(three_weeks):
    with pytest.raises(ValueError, match="match no auction"):
        score_bids(three_weeks, make_bids(a=90.0, b=90.0, c=70.0, z=10.0))


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -5.0])
def test_invalid_bid_raises(three_weeks, bad):
    with pytest.raises(ValueError, match="positive number"):
        score_bids(three_weeks, make_bids(a=90.0, b=bad, c=70.0))


def test_auction_without_marginal_price_raises():
    auctions = make_auctions([("a", "2025-01-06", "upwards", np.nan)])
    with pytest.raises(ValueError, match="no marginal price"):
        score_bids(auctions, make_bids(a=90.0))


def test_duplicated_auction_raises(three_weeks):
    twice = pd.concat([three_weeks, three_weeks.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicated auction_id"):
        score_bids(twice, make_bids(a=90.0, b=90.0, c=70.0))


# --- real data ----------------------------------------------------------------------

def test_select_weekly_keeps_one_auction_per_week_and_direction(mfrr_auctions):
    weekly = select_weekly(mfrr_auctions)
    assert weekly["direction"].value_counts().to_dict() == {"upwards": 410, "downwards": 410}
    assert not weekly.duplicated(subset=["date", "direction"]).any()
    assert not weekly["auction_id"].str.contains("KORR").any()
    assert weekly["marginal"].notna().all()


def test_year_matches_swissgrid_auction_id(mfrr_auctions):
    weekly = select_weekly(mfrr_auctions)
    scored = score_bids(weekly, weekly[["auction_id"]].assign(bid=1.0))
    id_year = 2000 + scored["auction_id"].str.extract(r"^TRL[+-]?_(\d\d)_")[0].astype(int)
    assert (scored["year"] == id_year).all()


def test_bidding_90_percent_of_the_true_price_earns_90_percent(mfrr_auctions):
    weekly = select_weekly(mfrr_auctions)
    table = summarise(score_bids(
        weekly, weekly[["auction_id"]].assign(bid=0.9 * weekly["marginal"])))
    np.testing.assert_allclose(table["pct_of_best"], 90.0)
    assert (table["win_rate"] == 1.0).all()
