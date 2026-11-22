"""Leakage tests: a feature for auction A may only use results known at A's gate closure.

Rule (statement, section 4): an auction counts as known only 2 hours after its
own gate closure (avail = gc + 2h), so auctions of the same sitting never see
each other. Tested on LeakageSafeLag (src/features/lags.py), the lag behind the
baseline, and on WalkForwardSplitter.

Times below are UTC. In winter Zurich = UTC+1, so "Friday 14:30" is 13:30 UTC.
Run with: uv run pytest tests/test_leakage.py
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import WalkForwardSplitter
from src.features.lags import LeakageSafeLag


def make_auctions(rows):
    """rows: list of (auction_id, series, gate_closure_utc, marginal_price).

    marginal_price None = no-buy auction. Columns follow src.data.data.build_auctions.
    """
    df = pd.DataFrame(rows, columns=["auction_id", "series", "gc", "marginal"])
    df["gc"] = pd.to_datetime(df["gc"], utc=True)
    df["marginal"] = df["marginal"].astype(float)
    df["y"] = np.log10(df["marginal"])
    df["avail"] = df["gc"] + pd.Timedelta(hours=2)
    return df


def lag_of(auctions, auction_id):
    out = LeakageSafeLag().compute(auctions).set_index("auction_id")
    return out.loc[auction_id, "y_lag1"]


UP = "daily|upwards|08_12"   # one series used in most tests below


# ---------------------------------------------------------------------------
# Edge cases from the statement, on hand-made tables
# ---------------------------------------------------------------------------

def test_monday_not_known_when_tuesday_closes():
    # Monday's blocks close Friday 14:30, Tuesday's Friday 15:30: only 1 h apart
    a = make_auctions([
        ("prev_week", UP, "2025-01-03 13:30", 1000.0),   # long known
        ("monday",    UP, "2025-01-10 13:30", 10.0),     # known Friday 15:30 UTC
        ("tuesday",   UP, "2025-01-10 14:30", 1.0),
    ])
    assert lag_of(a, "tuesday") == pytest.approx(3.0)    # prev_week, not monday


def test_same_sitting_auctions_do_not_see_each_other():
    # Saturday's and Sunday's blocks both close on Thursday
    a = make_auctions([
        ("friday",   UP, "2025-01-08 13:30", 100.0),     # Wednesday sitting
        ("saturday", UP, "2025-01-09 13:30", 10.0),      # Thursday sitting
        ("sunday",   UP, "2025-01-09 14:30", 1.0),       # Thursday sitting
    ])
    assert lag_of(a, "sunday") == pytest.approx(2.0)     # friday, not saturday


def test_result_known_exactly_at_gate_closure_is_usable():
    a = make_auctions([
        ("h", UP, "2025-01-10 10:00:00", 100.0),         # known 12:00:00
        ("a", UP, "2025-01-10 12:00:00", 1.0),
    ])
    assert lag_of(a, "a") == pytest.approx(2.0)


def test_result_known_one_second_late_is_not_usable():
    a = make_auctions([
        ("h", UP, "2025-01-10 10:00:01", 100.0),         # known 12:00:01
        ("a", UP, "2025-01-10 12:00:00", 1.0),
    ])
    assert np.isnan(lag_of(a, "a"))


def test_auction_never_sees_its_own_result():
    a = make_auctions([("a", UP, "2025-01-10 12:00", 10.0)])
    assert np.isnan(lag_of(a, "a"))


def test_only_the_same_series_is_used():
    a = make_auctions([
        ("same",        UP,                     "2025-01-01 13:30", 100.0),
        ("other_block", "daily|upwards|12_16",  "2025-01-05 13:30", 10.0),
        ("other_dir",   "daily|downwards|08_12", "2025-01-05 13:30", 10.0),
        ("other_freq",  "weekly|upwards|",       "2025-01-05 13:30", 10.0),
        ("a",           UP,                     "2025-01-10 13:30", 1.0),
    ])
    assert lag_of(a, "a") == pytest.approx(2.0)


def test_most_recent_known_auction_wins():
    a = make_auctions([
        ("older", UP, "2025-01-01 13:30", 1000.0),
        ("newer", UP, "2025-01-05 13:30", 100.0),
        ("a",     UP, "2025-01-10 13:30", 1.0),
    ])
    assert lag_of(a, "a") == pytest.approx(2.0)


def test_no_buy_auctions_are_skipped():
    a = make_auctions([
        ("priced", UP, "2025-01-01 13:30", 100.0),
        ("no_buy", UP, "2025-01-05 13:30", None),
        ("a",      UP, "2025-01-10 13:30", 1.0),
    ])
    assert lag_of(a, "a") == pytest.approx(2.0)


def test_first_auction_of_a_series_has_no_lag():
    a = make_auctions([
        ("first", UP, "2025-01-01 13:30", 100.0),
        ("other", "daily|downwards|08_12", "2024-12-01 13:30", 10.0),
    ])
    assert np.isnan(lag_of(a, "first"))


def test_compute_keeps_every_auction_once():
    a = make_auctions([
        ("x", UP, "2025-01-05 13:30", 100.0),
        ("y", UP, "2025-01-01 13:30", 10.0),
        ("z", "daily|downwards|08_12", "2025-01-03 13:30", None),
    ])
    out = LeakageSafeLag().compute(a)
    assert len(out) == len(a)
    assert sorted(out["auction_id"]) == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Changing the future must not change the past (real mFRR data)
# ---------------------------------------------------------------------------

# Midnight cutoffs test the far future. The 15:00 UTC ones fall right after a
# sitting (gate closures at 13:30 / 14:30 UTC in winter): auctions closing there
# are past but their results are not known yet, which tests the 2-hour rule.
CUTOFFS = ["2022-06-01", "2025-01-01", "2025-07-01",
           "2025-01-09 15:00", "2025-02-14 15:00"]


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_shuffling_future_results_does_not_change_past_lags(mfrr_auctions, cutoff):
    t0 = pd.Timestamp(cutoff, tz="UTC")
    before = LeakageSafeLag().compute(mfrr_auctions).set_index("auction_id")

    perturbed = mfrr_auctions.copy()
    future = perturbed["avail"] > t0                   # results unknown at t0
    rng = np.random.default_rng(0)
    perturbed.loc[future, "y"] = rng.permutation(perturbed.loc[future, "y"].to_numpy())
    after = LeakageSafeLag().compute(perturbed).set_index("auction_id")

    past = before.index[before["gc"] <= t0]
    assert future.sum() > 0 and len(past) > 0
    pd.testing.assert_series_equal(before.loc[past, "y_lag1"], after.loc[past, "y_lag1"])


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_deleting_future_rows_does_not_change_past_lags(mfrr_auctions, cutoff):
    """Rebuild the table as it looked at t0, and compare."""
    t0 = pd.Timestamp(cutoff, tz="UTC")
    full = LeakageSafeLag().compute(mfrr_auctions).set_index("auction_id")

    # at t0: auctions not closed yet do not exist; closed ones whose result
    # is not published yet (gc <= t0 < avail) exist but have no known price
    as_of_t0 = mfrr_auctions[mfrr_auctions["gc"] <= t0].copy()
    as_of_t0.loc[as_of_t0["avail"] > t0, "y"] = np.nan
    truncated = LeakageSafeLag().compute(as_of_t0).set_index("auction_id")

    past = full.index[full["gc"] <= t0]
    pd.testing.assert_series_equal(full.loc[past, "y_lag1"], truncated.loc[past, "y_lag1"])


def test_walk_forward_train_is_known_before_each_test_month(mfrr_auctions):
    n_splits = 0
    for train, test in WalkForwardSplitter().splits(mfrr_auctions):
        n_splits += 1
        assert (train["avail"] < test["gc"].min()).all()
    assert n_splits > 0
