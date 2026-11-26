import numpy as np
import pandas as pd

from src.features.external import add_day_ahead_features, day_ahead_daily


def make_prices(first_day, n_days, price_of_day):
    """Hourly day-ahead prices; every hour of delivery day i costs price_of_day(i) + hour / 100."""
    start = pd.Timestamp(first_day, tz="Europe/Zurich")
    local = pd.date_range(start, start + pd.DateOffset(days=n_days), freq="h", inclusive="left")
    day = (local.normalize() - local[0].normalize()).days
    return pd.DataFrame({
        "timestamp_utc": local.tz_convert("UTC"),
        "price_eur_mwh": [price_of_day(d) + h / 100 for d, h in zip(day, local.hour)],
    })


def make_auction(auction_id, series, gc_local):
    gc = pd.Timestamp(gc_local, tz="Europe/Zurich").tz_convert("UTC")
    return {"auction_id": auction_id, "series": series, "gc": gc}


def test_delivery_day_available_at_13h_the_day_before():
    prices = make_prices("2024-03-01", 10, lambda d: 100 * (d + 1))   # day i costs ~100*(i+1)
    features = pd.DataFrame([
        make_auction("before", "daily|upwards|08_12", "2024-03-05 12:59"),   # D = 6 mar not yet public
        make_auction("at", "daily|upwards|08_12", "2024-03-05 13:00"),       # D = 6 mar just public
    ])
    f = add_day_ahead_features(features, prices).set_index("auction_id")
    assert round(f.loc["before", "da_ch_last"], -2) == 500   # 5 mar (day index 4)
    assert round(f.loc["at", "da_ch_last"], -2) == 600       # 6 mar (day index 5)


def test_block_price_uses_the_right_hours():
    prices = make_prices("2024-03-01", 5, lambda d: 0)       # price = hour / 100
    features = pd.DataFrame([
        make_auction("D", "daily|upwards|08_12", "2024-03-04 14:30"),
        make_auction("W", "weekly|upwards|week", "2024-03-04 14:30"),
    ])
    f = add_day_ahead_features(features, prices).set_index("auction_id")
    assert np.isclose(f.loc["D", "da_ch_block"], np.mean([8, 9, 10, 11]) / 100)
    assert np.isnan(f.loc["W", "da_ch_block"])
    assert np.isclose(f.loc["W", "da_ch_last"], np.mean(range(24)) / 100)


def test_dst_day_has_23_hours_and_still_works():
    prices = make_prices("2024-03-29", 4, lambda d: 0)       # 31 mar 2024: clocks go forward
    daily = day_ahead_daily(prices)
    assert len(daily) == 4
    assert daily["avail"].dt.tz is not None
    # 13:00 Zurich the day before = 11:00 UTC in summer time, 12:00 UTC in winter time
    assert daily["avail"].dt.hour.tolist() == [12, 12, 12, 11]   # published on 28, 29, 30 (winter) and 31 mar (summer)


def test_day_ahead_features_do_not_change_when_future_is_perturbed():
    prices = make_prices("2024-03-01", 20, lambda d: 50 + d)
    features = pd.DataFrame([
        make_auction(f"A{i}", "daily|downwards|12_16", f"2024-03-{i:02d} 14:30") for i in range(3, 20)
    ])
    t = pd.Timestamp("2024-03-10 14:30", tz="Europe/Zurich").tz_convert("UTC")

    daily = day_ahead_daily(prices)
    future_days = daily.loc[daily["avail"] > t, "delivery_date"]
    local_day = prices["timestamp_utc"].dt.tz_convert("Europe/Zurich").dt.normalize().dt.tz_localize(None)
    perturbed = prices.copy()
    perturbed.loc[local_day.isin(future_days), "price_eur_mwh"] += 1000

    cols = ["da_ch_last", "da_ch_mean7", "da_ch_block"]
    before = add_day_ahead_features(features, prices).set_index("auction_id")
    after = add_day_ahead_features(features, perturbed).set_index("auction_id")
    past = before["gc"] <= t
    pd.testing.assert_frame_equal(before.loc[past, cols], after.loc[past, cols])
    assert not before.loc[~past, "da_ch_last"].equals(after.loc[~past, "da_ch_last"])


# ----------------------------------------------------------------------- reservoirs
from src.features.external import add_reservoir_features, load_reservoirs, reservoir_weekly_features


def make_reservoir_csv(tmp_path, n_weeks, fill_of_week):
    """BFE-like csv: date, 5 levels, 5 capacities (capacity 1000 GWh everywhere)."""
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="7D")      # Mondays
    rows = []
    for i, d in enumerate(dates):
        level = 1000 * fill_of_week(i)
        rows.append([d.strftime("%Y-%m-%d")] + [level] * 5 + [1000] * 5)
    cols = ["Datum", "Wallis", "Graubuenden", "Tessin", "UebrigCH", "TotalCH",
            "Wallis_max", "Graubuenden_max", "Tessin_max", "UebrigCH_max", "TotalCH_max"]
    path = tmp_path / "reservoirs_ch.csv"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


def test_reservoir_week_available_seven_days_later(tmp_path):
    path = make_reservoir_csv(tmp_path, 10, lambda i: i / 10)          # week i: fill i/10
    res = load_reservoirs(path)
    week3 = pd.Timestamp("2020-01-27", tz="Europe/Zurich")            # date of week index 3
    features = pd.DataFrame([
        make_auction("before", "weekly|upwards|week", "2020-02-02 23:59"),   # < 7 days after: week 2
        make_auction("at", "weekly|upwards|week", "2020-02-03 00:00"),       # exactly 7 days: week 3
    ])
    f = add_reservoir_features(features, res).set_index("auction_id")
    assert np.isclose(f.loc["before", "res_fill"], 0.2)
    assert np.isclose(f.loc["at", "res_fill"], 0.3)
    assert res.loc[3, "avail"] == (week3 + pd.Timedelta(days=7)).tz_convert("UTC")


def test_reservoir_anomaly_uses_only_previous_years(tmp_path):
    # same fill every year except the last one: the anomaly of the first year is NaN,
    # and later years are compared with the past only
    path = make_reservoir_csv(tmp_path, 52 * 3, lambda i: 0.5 if i < 104 else 0.8)
    w = reservoir_weekly_features(load_reservoirs(path))
    assert w.loc[:51, "res_anomaly"].isna().all()                      # no past year yet
    second = w.loc[52:103, "res_anomaly"]
    assert second.notna().sum() >= 50                                  # (a week missing the year before stays NaN)
    assert np.allclose(second.dropna(), 0.0)                           # same as the past
    assert np.allclose(w.loc[104:, "res_anomaly"].dropna(), 0.3)       # 0.8 vs past mean 0.5


def test_reservoir_features_do_not_change_when_future_is_perturbed(tmp_path):
    path = make_reservoir_csv(tmp_path, 60, lambda i: 0.3 + 0.01 * i)
    res = load_reservoirs(path)
    t = pd.Timestamp("2020-06-01 13:00", tz="UTC")
    perturbed = res.copy()
    future = perturbed["avail"] > t
    perturbed.loc[future, "total_fill"] += 0.5

    features = pd.DataFrame([
        make_auction(f"A{i}", "daily|upwards|08_12", d.strftime("%Y-%m-%d 14:30"))
        for i, d in enumerate(pd.date_range("2020-02-01", "2020-12-31", freq="5D"))
    ])
    cols = ["res_fill", "res_change4", "res_anomaly"]
    before = add_reservoir_features(features, res).set_index("auction_id")
    after = add_reservoir_features(features, perturbed).set_index("auction_id")
    past = before["gc"] <= t
    pd.testing.assert_frame_equal(before.loc[past, cols], after.loc[past, cols])
    assert not before.loc[~past, "res_fill"].equals(after.loc[~past, "res_fill"])