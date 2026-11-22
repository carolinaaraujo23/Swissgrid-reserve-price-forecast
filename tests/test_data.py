"""Checks of the mFRR auction table against the numbers in the project brief.

Run with: uv run pytest tests/test_data.py
"""

import numpy as np
import pandas as pd
import pytest

N_AUCTIONS = 34_895
N_NO_BUY = 5_425
COUNTS = {
    ("daily", "downwards"): 17_040,
    ("daily", "upwards"): 17_034,
    ("weekly", "downwards"): 410,
    ("weekly", "upwards"): 411,
}

# Gate closure (Zurich time) by weekday of the delivery date, from the brief.
DAILY_GATE_CLOSURE = {
    0: "Fri 14:30",  # Monday
    1: "Fri 15:30",  # Tuesday
    2: "Mon 14:30",  # Wednesday
    3: "Tue 14:30",  # Thursday
    4: "Wed 14:30",  # Friday
    5: "Thu 14:30",  # Saturday
    6: "Thu 15:30",  # Sunday
}
WEEKLY_GATE_CLOSURE = "Tue 13:00"


# --- counts -----------------------------------------------------------------

def test_total_auction_count(mfrr_auctions):
    assert len(mfrr_auctions) == N_AUCTIONS


def test_counts_by_frequency_and_direction(mfrr_auctions):
    counts = mfrr_auctions.groupby(["freq", "direction"]).size().to_dict()
    assert counts == COUNTS


def test_no_anticipated_auctions(mfrr_offers, mfrr_auctions):
    assert "weekly_anticipated" not in set(mfrr_auctions["freq"])
    assert not mfrr_offers["is_anticipated"].any()


def test_daily_series_have_six_blocks(mfrr_auctions):
    daily = mfrr_auctions[mfrr_auctions["freq"] == "daily"]
    assert set(daily["block"]) == {"00_04", "04_08", "08_12", "12_16", "16_20", "20_24"}


# --- duplicates ---------------------------------------------------------------

def test_auction_id_unique(mfrr_auctions):
    assert mfrr_auctions["auction_id"].is_unique


def test_one_auction_per_series_and_date(mfrr_auctions):
    """No two auctions share frequency, direction, block and delivery date,
    except one known correction auction: week 42 of 2019, upwards, has a
    regular auction (TRL_19_KW42_up, 36 bids) and a 'KORR' correction auction
    (TRL_19_KW42_KORR_up, 1 bid of 5 MW) with the same gate closure. The brief's
    count of 411 weekly upwards auctions includes it."""
    key = ["freq", "direction", "block", "date"]
    dup_ids = set(mfrr_auctions.loc[mfrr_auctions.duplicated(subset=key, keep=False), "auction_id"])
    assert dup_ids == {"TRL_19_KW42_up", "TRL_19_KW42_KORR_up"}


def test_auction_attributes_constant_within_auction(mfrr_offers):
    """Every bid of an auction carries the same date, direction, block and gate closure."""
    cols = ["date", "direction", "auction_frequency", "time_block", "gate_closure_time_utc"]
    n_unique = mfrr_offers.groupby("auction_id")[cols].nunique(dropna=False)
    assert (n_unique == 1).all().all()


# --- time zones and gate closures --------------------------------------------

def test_gate_closure_is_utc(mfrr_auctions):
    assert str(mfrr_auctions["gc"].dt.tz) == "UTC"
    assert mfrr_auctions["gc"].notna().all()


def test_gate_closure_before_delivery(mfrr_auctions):
    delivery_start = mfrr_auctions["date"].dt.tz_localize("Europe/Zurich").dt.tz_convert("UTC")
    assert (mfrr_auctions["gc"] < delivery_start).all()


def test_daily_gate_closure_schedule(mfrr_auctions):
    daily = mfrr_auctions[mfrr_auctions["freq"] == "daily"]
    local = daily["gc"].dt.tz_convert("Europe/Zurich").dt.strftime("%a %H:%M")
    expected = daily["date"].dt.day_of_week.map(DAILY_GATE_CLOSURE)
    assert (local == expected).all()


def test_weekly_gate_closure_schedule(mfrr_auctions):
    weekly = mfrr_auctions[mfrr_auctions["freq"] == "weekly"]
    local = weekly["gc"].dt.tz_convert("Europe/Zurich").dt.strftime("%a %H:%M")
    assert (local == WEEKLY_GATE_CLOSURE).all()
    assert (weekly["date"].dt.day_of_week == 0).all()  # delivery starts on Monday


def test_avail_is_gate_closure_plus_two_hours(mfrr_auctions):
    delta = mfrr_auctions["avail"] - mfrr_auctions["gc"]
    assert (delta == pd.Timedelta(hours=2)).all()


def test_table_sorted_by_gate_closure(mfrr_auctions):
    assert mfrr_auctions["gc"].is_monotonic_increasing


# --- target and no-buy auctions ----------------------------------------------

def test_no_buy_count(mfrr_auctions):
    assert mfrr_auctions["y"].isna().sum() == N_NO_BUY


def test_no_buy_auctions_have_nothing_awarded(mfrr_auctions):
    no_buy = mfrr_auctions[mfrr_auctions["y"].isna()]
    assert (no_buy["awarded"] == 0).all()
    assert no_buy["marginal"].isna().all()


def test_auctions_with_target_have_awarded_volume(mfrr_auctions):
    bought = mfrr_auctions[mfrr_auctions["y"].notna()]
    assert (bought["awarded"] > 0).all()


def test_accepted_matches_awarded_volume(mfrr_offers):
    assert (mfrr_offers["accepted"] == (mfrr_offers["awarded_volume"] > 0)).all()
    assert (mfrr_offers["awarded_volume"] <= mfrr_offers["offered_volume"]).all()


def test_target_is_log10_of_max_accepted_price(mfrr_offers, mfrr_auctions):
    expected = mfrr_offers[mfrr_offers["accepted"]].groupby("auction_id")["power_price"].max()
    got = mfrr_auctions.set_index("auction_id")["marginal"].dropna()
    pd.testing.assert_series_equal(
        got.sort_index(), expected.sort_index(), check_names=False
    )
    bought = mfrr_auctions.dropna(subset=["y"])
    np.testing.assert_allclose(bought["y"], np.log10(bought["marginal"]))


def test_prices_positive_and_single_unit(mfrr_offers):
    assert (mfrr_offers["power_price"] > 0).all()
    assert set(mfrr_offers["power_price_unit"].dropna()) == {"CHF/MW"}


# --- known deviation from the README ------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "README says weekly series use block='week', but time_block is '' (not NaN) "
    "for weekly auctions, so fillna('week') never fires and the key is 'weekly|<dir>|'"
))
def test_weekly_series_key_matches_readme(mfrr_auctions):
    weekly = mfrr_auctions[mfrr_auctions["freq"] == "weekly"]
    assert weekly["series"].str.endswith("|week").all()
