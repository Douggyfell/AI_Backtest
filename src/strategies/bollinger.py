import pandas as pd
from src.strategies.base import Strategy


class BollingerReversion(Strategy):


    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        rolling_mean = df["close"].rolling(self.window).mean()
        rolling_std = df["close"].rolling(self.window).std()

        df["bb_mid"] = rolling_mean
        df["bb_upper"] = rolling_mean + self.num_std * rolling_std
        df["bb_lower"] = rolling_mean - self.num_std * rolling_std

        df["signal"] = 0
        df.loc[df["close"] < df["bb_lower"], "signal"] = 1
        df.loc[df["close"] > df["bb_upper"], "signal"] = 0

        return df[["signal"]]
