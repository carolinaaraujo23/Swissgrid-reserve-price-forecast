"""Leakage checks for the previous-known-auction lag (LeakageSafeLag).

Idea: pick a cutoff time t, overwrite the target of every auction whose result
is only available after t with random values, and recompute the lags. Lags of
auctions that close at or before t must not change. If they do, the lag used
information that was not yet published.

Run with: uv run pytest tests/test_lag_leakage.py
"""

import numpy as np
import pandas as pd
import pytest

from src.features.lags import LeakageSafeLag


def compute_lags(auctions: pd.DataFrame, avail_col: str = "avail") -> pd.Series:
    lag = LeakageSafeLag(avail_col=avail_col).compute(auctions)
    return lag.set_index("auction_id")["y_lag1"]


def leaks(auctions: pd.DataFrame, cutoff: pd.Timestamp, avail_col: str = "avail") -> bool:
    """True if lags of auctions closing at or before `cutoff` depend on results
    that only become available after `cutoff`."""
    base = compute_lags(auctions, avail_col)
    perturbed = auctions.copy()
    future = (perturbed["avail"] > cutoff) & perturbed["y"].notna()
    rng = np.random.default_rng(0)
    perturbed.loc[future, "y"] = rng.uniform(-3, 6, future.sum())
    after = compute_lags(perturbed, avail_col)

    targets = auctions.loc[auctions["gc"] <= cutoff, "auction_id"]
    b, a = base.loc[targets], after.loc[targets]
    same = (b == a) | (b.isna() & a.isna())
    return not same.all()


def _first_daily_gc(auctions: pd.DataFrame, weekday: int, year: int) -> pd.Timestamp:
    daily = auctions[(auctions["freq"] == "daily") & (auctions["date"].dt.year == year)]
    return daily.loc[daily["date"].dt.day_of_week == weekday, "gc"].iloc[0]


# --- same-sitting and close-in-time edge cases --------------------------------

def test_monday_and_tuesday_close_one_hour_apart(mfrr_auctions):
    """Brief: Monday's blocks close Friday 14:30, Tuesday's Friday 15:30. With
    the 2 h rule, Monday's result is NOT known when Tuesday closes."""
    mon = _first_daily_gc(mfrr_auctions, weekday=0, year=2025)
    tue = _first_daily_gc(mfrr_auctions, weekday=1, year=2025)
    assert tue - mon == pd.Timedelta(hours=1)


def test_tuesday_lag_does_not_use_monday(mfrr_auctions):
    """Cutoff = Tuesday's gate closure: Monday's result (available Fri 16:30)
    must not influence any lag of an auction closing by then."""
    for year in (2019, 2022, 2025):
        cutoff = _first_daily_gc(mfrr_auctions, weekday=1, year=year)
        assert not leaks(mfrr_auctions, cutoff), year


def test_sunday_lag_does_not_use_saturday(mfrr_auctions):
    """Saturday and Sunday blocks both close on Thursday (14:30 and 15:30):
    they must not see each other's results."""
    for year in (2019, 2022, 2025):
        cutoff = _first_daily_gc(mfrr_auctions, weekday=6, year=year)
        assert not leaks(mfrr_auctions, cutoff), year


def test_daily_lag_does_not_use_weekly_of_same_tuesday(mfrr_auctions):
    """Weekly auctions close Tuesday 13:00, Thursday daily blocks Tuesday 14:30."""
    weekly = mfrr_auctions[mfrr_auctions["freq"] == "weekly"]
    cutoff = weekly.loc[weekly["date"].dt.year == 2025, "gc"].iloc[0]
    assert not leaks(mfrr_auctions, cutoff)


@pytest.mark.parametrize("seed", range(5))
def test_random_cutoffs_do_not_leak(mfrr_auctions, seed):
    rng = np.random.default_rng(seed)
    test_period = mfrr_auctions[mfrr_auctions["gc"] >= pd.Timestamp("2025-01-01", tz="UTC")]
    cutoff = test_period["gc"].iloc[rng.integers(len(test_period))]
    assert not leaks(mfrr_auctions, cutoff)


# --- the test itself must catch a leaky lag ------------------------------------

def test_leak_detector_catches_gc_based_lag(mfrr_auctions):
    """Sanity check of the harness: a lag that treats results as known at gate
    closure (no 2 h delay) must be flagged on Tuesday's cutoff."""
    cutoff = _first_daily_gc(mfrr_auctions, weekday=1, year=2025)
    assert leaks(mfrr_auctions, cutoff, avail_col="gc")


# --- exactness on a sample -------------------------------------------------------

def test_lag_equals_latest_known_auction_of_series(mfrr_auctions):
    """Brute-force reference on a sample: y_lag1 is the target of the latest
    auction of the same series whose result was available at gate closure."""
    lags = compute_lags(mfrr_auctions)
    known = mfrr_auctions.dropna(subset=["y"])
    sample = mfrr_auctions.sample(400, random_state=0)
    for row in sample.itertuples():
        cand = known[(known["series"] == row.series) & (known["avail"] <= row.gc)]
        expected = cand.sort_values("avail")["y"].iloc[-1] if len(cand) else np.nan
        got = lags.loc[row.auction_id]
        if np.isnan(expected):
            assert np.isnan(got), row.auction_id
        else:
            assert got == pytest.approx(expected), row.auction_id
