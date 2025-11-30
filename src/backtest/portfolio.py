import pandas as pd


class Portfolio:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0):
        self.data = data
        self.initial_capital = initial_capital
        self._results = None

    def run(self) -> pd.DataFrame:
        df = self.data.copy()

        # Use prior day's signal as today's position
        df["position"] = df["signal"].shift(1).fillna(0)

        # Asset returns
        df["return"] = df["close"].pct_change().fillna(0.0)

        # Strategy returns
        df["strategy_return"] = df["position"] * df["return"]

        # Strategy equity
        df["equity"] = (1 + df["strategy_return"]).cumprod() * self.initial_capital

        # Buy & hold benchmark
        first_close = df["close"].iloc[0]
        shares_bought = self.initial_capital / first_close
        df["bh_equity"] = df["close"] * shares_bought
        df["bh_return"] = df["bh_equity"].pct_change().fillna(0.0)

        self._results = df
        return df

    def generate_trades(self) -> pd.DataFrame:

        if self._results is None:
            raise ValueError("Run portfolio.run() before generate_trades().")

        df = self._results
        pos = df["position"].fillna(0)
        prices = df["close"]

        trades = []
        current_pos = 0
        entry_idx = None
        entry_price = None

        for idx, (p, price) in enumerate(zip(pos, prices)):
            date = prices.index[idx]

            # Opening a new position
            if current_pos == 0 and p != 0:
                current_pos = p
                entry_idx = date
                entry_price = price

            # Closing or reversing a position
            elif current_pos != 0 and p != current_pos:
                # Close current
                exit_idx = date
                exit_price = price
                direction = current_pos  # 1 for long, -1 for short

                ret_pct = (exit_price / entry_price - 1) * direction
                pnl = ret_pct * self.initial_capital  # assuming full capital per trade

                trades.append(
                    {
                        "direction": "long" if direction > 0 else "short",
                        "entry_date": entry_idx,
                        "exit_date": exit_idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": ret_pct,
                        "pnl": pnl,
                    }
                )

                # If new position is non-zero, open new trade
                if p != 0:
                    current_pos = p
                    entry_idx = date
                    entry_price = price
                else:
                    current_pos = 0
                    entry_idx = None
                    entry_price = None

        # Close any open trade at final bar
        if current_pos != 0 and entry_idx is not None:
            exit_idx = prices.index[-1]
            exit_price = prices.iloc[-1]
            direction = current_pos

            ret_pct = (exit_price / entry_price - 1) * direction
            pnl = ret_pct * self.initial_capital

            trades.append(
                {
                    "direction": "long" if direction > 0 else "short",
                    "entry_date": entry_idx,
                    "exit_date": exit_idx,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": ret_pct,
                    "pnl": pnl,
                }
            )

        trades_df = pd.DataFrame(trades)
        return trades_df
