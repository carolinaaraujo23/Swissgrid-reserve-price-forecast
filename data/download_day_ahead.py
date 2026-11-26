"""Download Swiss day-ahead electricity prices (EPEX CH auction) from the Energy-Charts API.

Source: https://api.energy-charts.info (Fraunhofer ISE), no registration needed.
Data license: CC BY 4.0, attribution "energy-charts.info".

Output: data/external/day_ahead_ch.parquet with one row per delivery interval
(hourly, 15-minute since the Swiss switch to 15-minute products):
    timestamp_utc   start of the delivery interval (UTC)
    price_eur_mwh   day-ahead price (EUR/MWh, can be negative)

Run from the repo root:
    uv run python -m src.data.download_day_ahead
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "day_ahead_ch.parquet"
URL = "https://api.energy-charts.info/price"


def get_json(url, attempts=6, wait=10):
    """GET with retries: on HTTP 429 (too many requests) wait and try again, longer each time."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == attempts - 1:
                raise
            delay = int(error.headers.get("Retry-After") or wait * 2 ** attempt)
            print(f"  too many requests, waiting {delay} s")
            time.sleep(delay)


def fetch_month(start, end, bzn="CH"):
    """Prices for delivery days start..end (inclusive), as a DataFrame."""
    query = urllib.parse.urlencode({"bzn": bzn, "start": start, "end": end})
    data = get_json(f"{URL}?{query}")
    if "unix_seconds" not in data or "price" not in data:
        raise ValueError(f"unexpected answer for {start}..{end}: keys {list(data)}")
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(data["unix_seconds"], unit="s", utc=True),
        "price_eur_mwh": data["price"],
    })


def download(first="2019-01-01", last=None, bzn="CH", done=None, pause=3):
    """Download month by month. Months already in `done` (a DataFrame) are skipped,
    so a run that was interrupted can be resumed."""
    last = last or pd.Timestamp.today().strftime("%Y-%m-%d")
    have = set()
    if done is not None and len(done):
        local = done["timestamp_utc"].dt.tz_convert("Europe/Zurich")
        counts = local.dt.strftime("%Y-%m").value_counts()
        have = {str(m) for m, n in counts.items() if n >= 24 * 28}   # month looks complete
    parts = [done] if done is not None else []
    for month_start in pd.date_range(first, last, freq="MS"):
        month = month_start.strftime("%Y-%m")
        if month in have:
            continue
        month_end = min(month_start + pd.offsets.MonthEnd(0), pd.Timestamp(last))
        start, end = month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
        try:
            part = fetch_month(start, end, bzn)
        except Exception as error:
            print(f"{month}: FAILED ({error})")
            continue
        print(f"{month}: {len(part)} rows")
        parts.append(part)
        time.sleep(pause)                   # be gentle with a free API
    prices = pd.concat(parts).drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
    return prices.dropna(subset=["price_eur_mwh"]).reset_index(drop=True)


if __name__ == "__main__":
    done = pd.read_parquet(OUT) if OUT.exists() else None   # resume: keep what we already have
    prices = download(done=done)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(OUT, index=False)
    print(f"saved {len(prices)} rows to {OUT}")
    print(prices.head())
    print(prices.describe())