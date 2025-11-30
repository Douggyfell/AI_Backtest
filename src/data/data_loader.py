import yfinance as yf
import pandas as pd


def load_price_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:

    data = yf.download(ticker, period=period, interval=interval)

    # If yfinance returns a MultiIndex (e.g. ('Close', 'AAPL')), flatten it
    if isinstance(data.columns, pd.MultiIndex):
        # Use the first level (Price fields like Open/High/Low/Close/Volume)
        data.columns = data.columns.get_level_values(0)

    # Normalize column names to lowercase
    data = data.rename(columns=str.lower)

    # Keep only what we need
    cols = [c for c in data.columns if c in ["open", "high", "low", "close", "volume"]]
    data = data[cols]

    return data
