import pandas as pd


def win_rate(trades: pd.DataFrame) -> float:

    if trades.empty:
        return 0.0

    wins = (trades["pnl"] > 0).sum()
    total = len(trades)
    return wins / total if total > 0 else 0.0


def profit_factor(trades: pd.DataFrame) -> float:

    if trades.empty:
        return 0.0

    gains = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = trades.loc[trades["pnl"] < 0, "pnl"].sum()

    if losses == 0:
        return float("inf") if gains > 0 else 0.0

    return gains / abs(losses)
