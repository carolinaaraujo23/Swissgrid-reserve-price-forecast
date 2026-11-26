import numpy as np
import pandas as pd

TZ = "Europe/Zurich"
DA_CLOSE_HOUR = 11      # day-ahead gate closure on D-1, local time
DELAY_HOURS = 2         # project rule: known 2 h after the gate closure


def load_day_ahead(path):
    prices = pd.read_parquet(path)
    prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
    return prices


def day_ahead_daily(prices):
    local = prices["timestamp_utc"].dt.tz_convert(TZ)
    df = pd.DataFrame({
        "delivery_date": local.dt.normalize().dt.tz_localize(None),
        "block": [f"{4 * (h // 4):02d}_{4 * (h // 4) + 4:02d}" for h in local.dt.hour],
        "price": prices["price_eur_mwh"].to_numpy(),
    })
    base = df.groupby("delivery_date")["price"].mean().rename("base")
    blocks = df.pivot_table(index="delivery_date", columns="block", values="price", aggfunc="mean")
    daily = pd.concat([base, blocks], axis=1).reset_index()

    publication = daily["delivery_date"] - pd.Timedelta(days=1) + pd.Timedelta(hours=DA_CLOSE_HOUR + DELAY_HOURS)
    daily["avail"] = publication.dt.tz_localize(TZ).dt.tz_convert("UTC")
    return daily.sort_values("avail").reset_index(drop=True)


def add_day_ahead_features(features, prices, zone="ch"):
    p = f"da_{zone}"
    daily = day_ahead_daily(prices)
    daily[f"{p}_mean7"] = daily["base"].rolling(7, min_periods=1).mean()

    out = features.copy()
    out["_block"] = out["series"].str.split("|").str[-1]
    out = pd.merge_asof(
        out.sort_values("gc"),
        daily.rename(columns={"avail": "_da_avail", "base": f"{p}_last"}),
        left_on="gc",
        right_on="_da_avail",
        direction="backward",
        allow_exact_matches=True,
    )

    block_cols = [c for c in daily.columns if c[:2].isdigit()]
    out[f"{p}_block"] = np.nan
    for block in block_cols:
        rows = out["_block"] == block
        out.loc[rows, f"{p}_block"] = out.loc[rows, block]

    return out.drop(columns=["_block", "_da_avail", "delivery_date", *block_cols])


def day_ahead_feature_names(zone="ch"):
    return [f"da_{zone}_last", f"da_{zone}_mean7", f"da_{zone}_block"]


DAY_AHEAD_FEATURES = day_ahead_feature_names("ch")


# ---------------------------------------------------------------------------------------------
# Hydro reservoirs (BFE, dataset ogd17, weekly, "Sunday 24h"; free use, reference required)
# ---------------------------------------------------------------------------------------------
# The weekly report is published "the following week" (no fixed day): each row is treated as
# available 7 days after its reference time (conservative).
RESERVOIR_DELAY = pd.Timedelta(days=7)
REGIONS = ["valais", "grisons", "ticino", "rest", "total"]


def load_reservoirs(path):
    raw = pd.read_csv(path)
    if raw.shape[1] != 11:
        raise ValueError(f"expected 11 columns in {path}, got {raw.shape[1]}: {list(raw.columns)}")
    out = pd.DataFrame({"date": pd.to_datetime(raw.iloc[:, 0])})
    for i, region in enumerate(REGIONS):
        level = pd.to_numeric(raw.iloc[:, 1 + i], errors="coerce")
        capacity = pd.to_numeric(raw.iloc[:, 6 + i], errors="coerce")
        out[f"{region}_gwh"] = level
        out[f"{region}_fill"] = level / capacity
    out["avail"] = (out["date"] + RESERVOIR_DELAY).dt.tz_localize(TZ).dt.tz_convert("UTC")
    return out.sort_values("avail").reset_index(drop=True)


def reservoir_weekly_features(reservoirs):
    w = reservoirs[["date", "avail", "total_fill"]].rename(columns={"total_fill": "res_fill"}).copy()
    w["res_change4"] = w["res_fill"] - w["res_fill"].shift(4)
    iso = w["date"].dt.isocalendar()
    w["_week"] = iso["week"].clip(upper=52).astype(int)            # week 53 joins week 52
    w["_year"] = iso["year"].astype(int)

    # mean of each (week, year), then for each year the mean of the same week in the years BEFORE it
    by_year = w.groupby(["_week", "_year"])["res_fill"].mean().reset_index()
    by_year["_past_mean"] = by_year.groupby("_week")["res_fill"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    w = w.merge(by_year[["_week", "_year", "_past_mean"]], on=["_week", "_year"], how="left")
    w["res_anomaly"] = w["res_fill"] - w["_past_mean"]
    return w.drop(columns=["_week", "_year", "_past_mean"])


def add_reservoir_features(features, reservoirs):
    weekly = reservoir_weekly_features(reservoirs).rename(columns={"avail": "_res_avail"})
    out = pd.merge_asof(
        features.sort_values("gc"),
        weekly.drop(columns="date"),
        left_on="gc",
        right_on="_res_avail",
        direction="backward",
        allow_exact_matches=True,
    )
    return out.drop(columns="_res_avail")


RESERVOIR_FEATURES = ["res_fill", "res_change4", "res_anomaly"]