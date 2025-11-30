import pandas as pd
from src.backtest.portfolio import Portfolio
from src.strategies.base import Strategy


class BacktestEngine:
    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 10000.0):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital

    def run(self):

        df = self.data.copy()

        # Generate signals
        signals = self.strategy.generate_signals(df)
        df["signal"] = signals["signal"]

        # Run portfolio simulation
        portfolio = Portfolio(df, initial_capital=self.initial_capital)
        results = portfolio.run()
        trades = portfolio.generate_trades()

        return results, trades
