import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path


def build_auctions(offers_path: str = "data/raw/offers", product: str = "mFRR", delay_hours: float = 2) -> pd.DataFrame:
    table = pq.read_table(offers_path, filters=[
                          ("auction_type", "==", product)])
    offers = table.to_pandas()
    offers = offers[~offers["is_anticipated"]].drop(columns=["is_anticipated"])
    accepted_offers = offers[offers["accepted"]]
    auctions = offers.groupby('auction_id').agg(
        date=('date', 'first'), direction=('direction', 'first'),
        freq=('auction_frequency', 'first'), block=('time_block', 'first'),
        gc=('gate_closure_time_utc', 'first'), n_bids=('power_price', 'size'),
        offered=('offered_volume', 'sum'), awarded=('awarded_volume', 'sum'))

    auctions['marginal'] = accepted_offers.groupby(
        "auction_id")["power_price"].max()  # regroups bids accepted per auction

    auctions['y'] = np.log10(auctions.marginal)
    auctions['avail'] = auctions.gc + pd.Timedelta(hours=delay_hours)
    auctions['series'] = auctions.freq + '|' + \
        auctions.direction + '|' + auctions.block.fillna('week')

    return auctions.reset_index().sort_values('gc').reset_index(drop=True)  # sort on gc


def save_auctions(
    auctions: pd.DataFrame,
    out_path: str = "data/processed/auctions.parquet",
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    auctions.to_parquet(out, index=False)


def load_auctions(path: str = "data/processed/auctions.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
