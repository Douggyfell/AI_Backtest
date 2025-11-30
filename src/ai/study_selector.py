import pandas as pd

from src.data import data_loader
from src.strategies.factory import create_strategy
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import sharpe_ratio, max_drawdown
from src.backtest.stats import win_rate, profit_factor


# Default parameter sets for each study / strategy
DEFAULT_STRATEGY_CONFIGS = {
    "sma": {"fast": 10, "slow": 30},
    "ema": {"fast": 12, "slow": 26},
    "rsi": {"period": 14, "lower": 30, "upper": 70},
    "bollinger": {"window": 20, "num_std": 2.0},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
}


def evaluate_strategies_for_ticker(ticker: str, period: str = "1y", initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Run all defined strategies on (ticker, period) and return
    a DataFrame of performance metrics for each.
    """
    data = data_loader.load_price_data(ticker, period=period)

    rows = []

    for stype, params in DEFAULT_STRATEGY_CONFIGS.items():
        config = {"type": stype, "params": params}

        try:
            strategy = create_strategy(config)
        except Exception as e:
            # If a strategy isn't implemented yet, skip it
            rows.append(
                {
                    "strategy": stype,
                    "status": f"ERROR: {e}",
                    "sharpe": None,
                    "max_drawdown": None,
                    "final_equity": None,
                    "num_trades": None,
                    "win_rate": None,
                    "profit_factor": None,
                }
            )
            continue

        engine = BacktestEngine(data, strategy, initial_capital=initial_capital)
        results, trades = engine.run()

        strat_ret = results["strategy_return"].dropna()
        strat_equity = results["equity"].dropna()

        metrics = {
            "strategy": stype,
            "status": "OK",
            "sharpe": sharpe_ratio(strat_ret),
            "max_drawdown": max_drawdown(strat_equity),
            "final_equity": float(strat_equity.iloc[-1]),
            "num_trades": int(len(trades)),
            "win_rate": win_rate(trades),
            "profit_factor": profit_factor(trades),
        }

        rows.append(metrics)

    df = pd.DataFrame(rows)
    return df


def rank_strategies(df: pd.DataFrame, risk_focus: str = "balanced") -> pd.DataFrame:

    df = df.copy()
    df = df[df["status"] == "OK"].dropna(subset=["sharpe", "max_drawdown", "final_equity"])

    if df.empty:
        return df

    # Build a simple composite score
    # Drawdown is negative, so we multiply by -1
    if risk_focus == "return":
        df["score"] = df["sharpe"] * 0.7 + (df["final_equity"] / df["final_equity"].mean()) * 0.3
    elif risk_focus == "defensive":
        df["score"] = df["sharpe"] * 0.5 + (-df["max_drawdown"]) * 0.5
    else:  # balanced
        df["score"] = df["sharpe"] * 0.5 + (-df["max_drawdown"]) * 0.3 + (df["final_equity"] / df["final_equity"].mean()) * 0.2

    df = df.sort_values("score", ascending=False)
    return df
