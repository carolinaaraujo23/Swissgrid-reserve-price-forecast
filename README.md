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

Each row of the raw data is one **bid**. `src/data/data.py` aggregates them into one row per **auction**, with the target and the time at which the result becomes known. Build the mFRR table once:

```bash
uv run python -m src.data.data      # writes data/processed/auctions_mFRR.parquet
```

or from Python (default paths are relative to the project root, so this works from any folder, notebooks included):

```python
from src.data.data import build_auctions, save_auctions, load_auctions

auctions = build_auctions(product="mFRR", delay_hours=2)   # reads data/raw/offers
save_auctions(auctions)                                     # data/processed/auctions_mFRR.parquet
auctions = load_auctions()
```

`build_auctions` reads only the requested product (`"mFRR"` or `"aFRR"`), drops the anticipated auctions and returns a table sorted by gate closure time. **These column names are the team convention**: use them in every module and notebook.

| column | meaning |
|---|---|
| `auction_id` | as in the raw data |
| `date`, `direction`, `freq`, `block` | auction date, up/down, `daily`/`weekly`, 4-hour time block (empty string `''` for weekly) |
| `gc` | gate closure time (UTC) |
| `n_bids`, `n_accepted` | number of bids submitted / accepted (`n_accepted = 0` when nothing was bought) |
| `offered`, `awarded` | total offered and awarded volume (MW) |
| `supply_ratio` | `offered / awarded`; `NaN` when nothing was bought |
| `holiday_share`, `day_of_week`, `day_type` | calendar columns of the raw data |
| `marginal` | marginal price = highest `power_price` among accepted bids (CHF/MW) |
| `y` | `log10(marginal)`, the modelling target |
| `avail` | `gc + delay_hours`: the time from which this auction's result may be used as a feature |
| `series` | `freq\|direction\|block`, with `block` = `week` for weekly auctions (e.g. `daily\|upwards\|08_12`, `weekly\|upwards\|week`): the same product repeated over time. mFRR has 14 series (12 daily, 2 weekly). |

mFRR auctions with no accepted bid are kept, with `marginal` and `y` set to `NaN`: they must still get a prediction, but they are not scored.

Sanity check (asserted in `tests/test_data.py`): mFRR 34,895 auctions (17,040 daily down, 17,034 daily up, 410 weekly down, 411 weekly up), 5,425 of them without any accepted bid.

`TRL_19_KW42_KORR_up` is a second (correction) auction for the same week, direction and gate closure as `TRL_19_KW42_up`. It is kept in the table, since the statement counts include it, but notebooks that need one auction per week and direction drop it. Open team question (`questions.txt`).

## Evaluation

`src/evaluation/evaluation.py` is the harness every model is scored with, so that results are comparable. All metrics are on log10 prices.

```python
from src.evaluation import Metrics, WalkForwardSplitter, Evaluator, PredictionValidator

# walk-forward: one test month at a time from 2025-01-01
for train, test in WalkForwardSplitter(start_test="2025-01-01", freq="M").splits(auctions):
    ...   # train = results known before the month (avail < month start), test = gc in the month

PredictionValidator.validate(preds, expected_ids=test_auction_ids)   # contract of results/PREDICTION_CONTRACT.md
report = Evaluator(actuals=auctions).evaluate(preds)                  # one row per model x freq x direction
```

- `Metrics`: MAE, RMSE, pinball loss, p10–p90 coverage.
- `WalkForwardSplitter`: train only contains results known before each test month (`avail < month start`); test = auctions whose `gc` falls in the month. Train may contain no-buy auctions: drop them before fitting.
- `PredictionValidator`: required columns, no NaN, prices > 0, no duplicate, p10 ≤ pred ≤ p90; with `expected_ids`, also checks that every test auction (no-buy included) has a prediction.
- `Evaluator`: takes the full auction table and a prediction file, keeps the test period (`gc ≥ start_test`) and the auctions with a price, and reports MAE / RMSE (plus pinball and coverage when `marginal_price_p10` and `marginal_price_p90` are given) by `freq × direction`. A scorable test auction without a prediction raises an error (`allow_missing=True` to score a partial file). To score weekly only, pass weekly rows on both sides.

**Metrics file.** Every model saves its scores as `results/metrics_<model>.parquet`, where `<model>` is the value of the `model` column of its prediction file. So that all files can be stacked into one comparison table:

- produce it **only with `Evaluator.evaluate`**, never by hand;
- one row per `model × freq × direction`, with the columns of `EvaluationRow`:

| column | meaning |
|---|---|
| `model`, `freq`, `direction` | group |
| `n` | number of scored auctions (test period, with a price) |
| `mae`, `rmse` | error of `log10(pred) - log10(actual)` |
| `pinball_q10`, `pinball_q50`, `pinball_q90` | pinball loss in log10; empty without `marginal_price_p10` / `marginal_price_p90` |
| `coverage_80` | share of actual prices inside [p10, p90] (target ≈ 0.80); empty without p10 / p90 |

`*.parquet` is git-ignored: metrics files stay local until the team settles the results format.

**Leakage-safe lag.** `src/features/lags.py` provides `LeakageSafeLag`: for each auction, `y_lag1` is the `y` of the most recent auction of the same `series` whose result was known at its gate closure (`avail ≤ gc`), skipping no-buy auctions. Auctions closing in the same sitting never see each other. `series_col` sets the grouping column.

## Baseline

`src/models/baseline.py`, class `PreviousKnownAuctionBaseline` (sklearn-like `fit` / `predict` / `evaluate` / `save`): the forecast for an auction is the marginal price of the latest **known** auction of the same series (`10 ** y_lag1`), not simply the previous delivery date. `fit` only stores the history, since the baseline learns nothing; `predict` never uses a result published after the auction's gate closure.

```bash
uv run python -m src.models.baseline   # build table, predict, validate, evaluate, save
```

Outputs in `results/` (git-ignored, as all `*.parquet`): `predictions_baseline.parquet` (8,316 mFRR test auctions, no-buy included) and `metrics_baseline.parquet`.

On the mFRR test period (2025-01-01 → end of data), every auction has a known price in its series: no fallback rule is needed (0 of 8,316).

| freq | direction | n scored | MAE (log10) | RMSE (log10) |
|---|---|---|---|---|
| daily | downwards | 4,013 | 0.149 | 0.244 |
| daily | upwards | 4,055 | 0.182 | 0.315 |
| weekly | downwards | 96 | 0.079 | 0.132 |
| weekly | upwards | 96 | 0.067 | 0.109 |

A MAE of 0.15 in log10 means a typical error by a factor 10^0.15 ≈ 1.4. Every model must report its scores next to these.

## Tests

```bash
uv run pytest                              # everything
uv run pytest tests/test_leakage.py        # one file
```

| file | what it checks |
|---|---|
| `test_data.py` | auction table against the statement counts, duplicates, UTC, no-buy handling, series keys |
| `test_evaluation.py` | metrics, walk-forward boundaries, evaluator and validator on hand-made tables |
| `test_leakage.py` | `LeakageSafeLag` on the statement's edge cases (same sitting, Monday/Tuesday, exact 2-hour boundary, same series only) and on real data (shuffling or deleting the future does not change past lags) |
| `test_features.py` | `build_features` and the Ridge prototype |

Tests that need the raw data are skipped when `data/raw/offers` is missing.

## Repository structure

```
data/
  raw/            original data, read-only (git-ignored)
  processed/      derived tables, rebuilt by code (git-ignored)
src/
  data/           data.py: auction table and target (team convention)
  features/       lags.py: leakage-safe lag; build_features.py: Ridge prototype features
  evaluation/     evaluation.py: walk-forward splits, metrics, evaluator, validator
  models/         baseline.py, ridge.py
notebooks/        exploration (EDA, Ridge prototype); import code from src/
tests/            unit tests (pytest)
results/          prediction files and metric tables; PREDICTION_CONTRACT.md
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