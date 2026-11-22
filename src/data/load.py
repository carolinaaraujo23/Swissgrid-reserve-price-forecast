import pandas as pd  
import numpy as np   



def load_offers(path, product, frequency):
    """Filters the data based on the product and frequency"""
    df = pd.read_parquet(path, 
                        columns=["auction_id",
                                        "date",
                                        "direction",
                                        "gate_closure_time_utc",
                                        "holiday_share", 
                                        "offered_volume",
                                        "awarded_volume",
                                        "power_price",
                                        "accepted",
                                        "year"],
                        filters=[[("auction_frequency", "==", frequency),
                                  ("is_anticipated", "==", False),
                                  ("auction_type", "==", product)]]

                        )
    df = df[~df["auction_id"].str.contains("KORR")]
    return df

def build_auctions(offers: pd.DataFrame):
    accepted = offers[offers["accepted"]]
    marginal_price = accepted.groupby("auction_id")["power_price"].max()

    other_cols = offers.groupby("auction_id")[["direction", "date", "gate_closure_time_utc"]].first()
    auctions = other_cols.join(marginal_price.rename("marginal_price"))
    auctions["date"] = pd.to_datetime(auctions["date"])

    auctions["n_bids"] = offers.groupby("auction_id").size()
    auctions["n_accepted"] = offers[offers["accepted"]].groupby("auction_id").size()
    auctions["total_offered_volume"] = offers.groupby("auction_id")["offered_volume"].sum()
    auctions["total_awarded_volume"] = offers.groupby("auction_id")["awarded_volume"].sum()
    auctions["supply_ratio"] = auctions["total_offered_volume"]/auctions["total_awarded_volume"]

    auctions["log10_price"] = np.log10(auctions["marginal_price"])
    auctions["available_time_utc"] = auctions["gate_closure_time_utc"] + pd.Timedelta(hours=2)
    auctions = auctions.sort_values(["gate_closure_time_utc", "direction"]).reset_index()
    
    return auctions