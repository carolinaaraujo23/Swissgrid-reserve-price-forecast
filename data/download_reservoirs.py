"""Download the weekly filling level of the Swiss hydro reservoirs (BFE, dataset ogd17).

Source: Bundesamt für Energie BFE, "Füllungsgrad der Speicherseen, Sonntag 24h,
Wochenbericht Speicherinhalt" (opendata.swiss, terms: free use, reference required).

One row per week (the date is the Monday 00:00 = Sunday 24h), 11 columns: the date,
the stored energy in GWh for Valais, Grisons, Ticino, rest of Switzerland and the
Swiss total, and the maximum capacity of the same five regions.

The report of a week is published "the following week": for the features, treat each
row as available only 7 days after its date (conservative).

Run from the repo root:
    uv run python -m src.data.download_reservoirs
"""

import urllib.request

from src.data.data import ROOT

URL = "https://www.uvek-gis.admin.ch/BFE/ogd/17/ogd17_fuellungsgrad_speicherseen.csv"
OUT = ROOT / "data" / "raw" / "external" / "reservoirs_ch.csv"


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        OUT.write_bytes(response.read())
    print(f"saved to {OUT}")
    print(OUT.read_text(encoding="utf-8").splitlines()[:4])
