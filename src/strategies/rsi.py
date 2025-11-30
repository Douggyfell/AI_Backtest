import pandas as pd
from src.strategies.base import Strategy


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


class RSIStrategy(Strategy):
    """
    RSI mean reversion:
      - Buy when RSI < lower
      - Sell/flat when RSI > upper
    """

    def __init__(self, period: int = 14, lower: int = 30, upper: int = 70):
        self.period = period
        self.lower = lower
        self.upper = upper

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["rsi"] = compute_rsi(df["close"], period=self.period)

        df["signal"] = 0
        df.loc[df["rsi"] < self.lower, "signal"] = 1
        df.loc[df["rsi"] > self.upper, "signal"] = 0

        return df[["signal"]]
