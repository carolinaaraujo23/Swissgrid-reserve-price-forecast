from pathlib import Path

import pandas as pd
import pytest

from src.data.data import build_auctions

OFFERS_PATH = Path("data/raw/offers")

requires_data = pytest.mark.skipif(
    not OFFERS_PATH.exists(),
    reason="raw offers data not found in data/raw/offers (see README)",
)


@pytest.fixture(scope="session")
def mfrr_offers() -> pd.DataFrame:
    """Raw mFRR bids, anticipated auctions removed."""
    if not OFFERS_PATH.exists():
        pytest.skip("raw offers data not found in data/raw/offers")
    offers = pd.read_parquet(OFFERS_PATH, filters=[("auction_type", "==", "mFRR")])
    return offers[~offers["is_anticipated"]].reset_index(drop=True)


@pytest.fixture(scope="session")
def mfrr_auctions() -> pd.DataFrame:
    """One row per mFRR auction, as built by src.data.data.build_auctions."""
    if not OFFERS_PATH.exists():
        pytest.skip("raw offers data not found in data/raw/offers")
    return build_auctions(str(OFFERS_PATH), product="mFRR", delay_hours=2)
