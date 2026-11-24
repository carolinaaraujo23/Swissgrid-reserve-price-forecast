import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_OFFERS = ROOT / "data" / "raw" / "offers"
AUCTIONS_PATH = ROOT / "data" / "processed" / "auctions_mFRR.parquet"


def build_auctions(offers_path: str | Path = RAW_OFFERS, product: str = "mFRR", delay_hours: float = 2) -> pd.DataFrame:
    table = pq.read_table(offers_path, filters=[
                          ("auction_type", "==", product)])
    offers = table.to_pandas()
    offers = offers[~offers["is_anticipated"]].drop(columns=["is_anticipated"])
    accepted_offers = offers[offers["accepted"]]
    auctions = offers.groupby('auction_id').agg(
        date=('date', 'first'), direction=('direction', 'first'),
        freq=('auction_frequency', 'first'), block=('time_block', 'first'),
        gc=('gate_closure_time_utc', 'first'), n_bids=('power_price', 'size'),
        offered=('offered_volume', 'sum'), awarded=('awarded_volume', 'sum'),
        holiday_share=('holiday_share', 'first'), day_of_week=('day_of_week', 'first'),
        day_type=('day_type', 'first'))

    # number of accepted bids: 0 for no-buy auctions
    auctions['n_accepted'] = accepted_offers.groupby('auction_id').size().reindex(
        auctions.index, fill_value=0)
    # offered / awarded volume; NaN (not inf) when nothing was bought
    auctions['supply_ratio'] = auctions.offered / auctions.awarded.where(auctions.awarded > 0)
    auctions['marginal'] = accepted_offers.groupby(
        # regroups bids accepted per auction
        "auction_id")["power_price"].max()
    

    auctions['y'] = np.log10(auctions.marginal)
    auctions['avail'] = auctions.gc + pd.Timedelta(hours=delay_hours)
    auctions['series'] = auctions.freq + '|' + \
        auctions.direction + '|' + \
        auctions.block.where(auctions.freq != 'weekly',
                             'week')  # not supposed to have nans but never know

    return auctions.reset_index().sort_values('gc').reset_index(drop=True)  # sort on gc


def save_auctions(
    auctions: pd.DataFrame,
    out_path: str | Path = AUCTIONS_PATH,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    auctions.to_parquet(out, index=False)


def load_auctions(path: str | Path = AUCTIONS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    a = build_auctions()
    save_auctions(a)
