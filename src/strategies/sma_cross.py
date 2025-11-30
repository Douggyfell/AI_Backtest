import pandas as pd
from src.strategies.base import Strategy


class SMACross(Strategy):
    """
    Simple Moving Average Crossover Strategy:
      - Go long when fast SMA > slow SMA
      - Go flat when fast SMA <= slow SMA
    """

    def __init__(self, fast: int = 10, slow: int = 20):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["sma_fast"] = df["close"].rolling(self.fast).mean()
        df["sma_slow"] = df["close"].rolling(self.slow).mean()

        df["signal"] = 0
        df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
        df.loc[df["sma_fast"] <= df["sma_slow"], "signal"] = 0

        return df[["signal"]]
