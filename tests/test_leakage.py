from src.data.data import load_auctions


def test_no_leakage():
    """Shuffle future rows → predictions unchanged."""
    df = load_auctions()
    df_sorted = df.sort_values('gc')
    # construire les lags sur df_sorted
    # shuffle les lignes post-t0
    # reconstruire → mêmes lags