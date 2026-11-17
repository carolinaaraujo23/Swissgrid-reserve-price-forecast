# Swissgrid reserve price forecast

Forecasting the marginal clearing price of Swissgrid's aFRR and mFRR control-power auctions before gate closure. CS-433 ML4Science project (EPFL) in collaboration with Carina Power.

The marginal price of an auction is the price of the most expensive accepted bid (CHF/MW). We predict it for the **daily** (4-hour blocks) and **weekly** auctions, **upwards and downwards**, for both products:

- **aFRR** — Swissgrid name SRL, secondary control power (only accepted bids are published)
- **mFRR** — Swissgrid name TRL, tertiary control power (accepted and rejected bids are published)

## Setup

Requires [uv](https://docs.astral.sh/uv/). The Python version is pinned in `.python-version` and uv installs it automatically.

```bash
# install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh                  # macOS / Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"     # Windows

git clone https://github.com/carolinaaraujo23/Swissgrid-reserve-price-forecast.git
cd Swissgrid-reserve-price-forecast
uv sync                     # creates .venv with the exact versions from uv.lock
```

Run everything through uv (no need to activate the venv):

```bash
uv run jupyter lab
uv run pytest
uv run python -m src.<module>
```

Add a dependency with `uv add <package>` (or `uv add --dev <package>` for tools) and commit both `pyproject.toml` and `uv.lock`.

## Data

The data is **not** stored in git. Download the parquet files with the link in the project description (ask Carina Power for a new link if it has expired) and unpack them so that the layout is:

```
data/raw/offers/auction_type=mFRR/year=2019/data.parquet
...
data/raw/offers/auction_type=aFRR/year=2026/data.parquet
```

Read the whole tree in one call; product and year come as columns:

```python
import pandas as pd
offers = pd.read_parquet("data/raw/offers", filters=[("auction_type", "==", "mFRR")])
```

Sanity check (without anticipated auctions): mFRR 34,895 auctions, 5,425 of them without any accepted bid; aFRR 5,643 auctions.

## Repository structure

```
data/
  raw/            original data, read-only (git-ignored)
  processed/      derived tables, rebuilt by code (git-ignored)
src/
  data/           loading, cleaning, target construction
  features/       feature engineering
  evaluation/     walk-forward splits, metrics, leakage checks
  models/         baseline, supervised models, agent-based / RL models
notebooks/        exploration; import code from src/
tests/            unit tests (pytest)
results/          prediction files and metric tables
figures/          plots
reports/          final report
```

## Project rules

These come from the project description and apply to every model:

- **Scope.** Daily and weekly auctions, both directions. Drop the anticipated auctions (`auction_frequency == "weekly_anticipated"` / `is_anticipated == True`).
- **Target.** `marginal_price` = highest `power_price` among accepted bids of an `auction_id`. mFRR auctions with no accepted bid have no target and are excluded. Work in `log10` of the price (very skewed distribution).
- **No leakage.** A prediction for auction A may only use data published before A's `gate_closure_time_utc`. An auction's result counts as known only **2 hours after its own gate closure**, so auctions closing in the same sitting never see each other.
- **Walk-forward evaluation.** Test period: 2025-01-01 to the end of the data. Before each test month, retrain only on data already known.
- **Baseline first.** Previous auction of the same series (product, direction, frequency, time block). Report it next to every model.
- **Metrics** by frequency and direction: MAE and RMSE of `log10(pred) - log10(actual)`; for quantile forecasts, p10–p90 coverage (target ≈ 80 %) and pinball loss.
- Daily and weekly prices are not directly comparable: a weekly price covers 42 four-hour blocks.

## Deliverable

One row per test auction, saved in `results/` (parquet or CSV):

| column | meaning |
|---|---|
| `auction_id` | as in the data |
| `marginal_price_pred` | point forecast (CHF/MW) |
| `marginal_price_p10`, `marginal_price_p90` | 10 % and 90 % quantiles (optional) |
| `model` | short model name |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch names, commit messages and pull-request reviews.