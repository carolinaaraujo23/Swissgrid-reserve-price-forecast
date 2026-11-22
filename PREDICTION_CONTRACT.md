File : predictions_<model_name>.parquet (or .csv)
A line per auction (daily + weekly, up + down) on test period : 2025-01-01 → end of data.
(If there is no marginal price, predict anyway but exclude from scoring)

Mandatory columns :
- auction_id          : str, just like in offers (ex. "TRL_26_11_03_08_12_up")
- marginal_price_pred : float, CHF/MW, > 0, no NaN
- model               : str, short name of model (ex. "baseline", "lgbm_q50", "agent_v1")

Recommended columns :
- marginal_price_p10  : float, CHF/MW
- marginal_price_p90  : float, CHF/MW

Validation rules (check by validator) :
- no NaN in marginal_price_pred
- marginal_price_pred > 0
- p10 ≤ pred ≤ p90
- no auction_id duplicated in a same file
- all auctions daily/weekly on test period are accounted for. 