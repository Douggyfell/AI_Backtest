from typing import Dict, Any

from src.ai import nl_to_strategy
from src.data import data_loader
from src.strategies.sma_cross import SMACross
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import sharpe_ratio, max_drawdown
from src.backtest.stats import win_rate, profit_factor


def build_strategy_from_config(config: dict):
    stype = config.get("type")
    params = config.get("params", {}) or {}

    if stype == "sma":
        return SMACross(
            fast=params.get("fast", 10),
            slow=params.get("slow", 20),
        )
    else:
        raise ValueError(f"Unknown strategy type: {stype}")


def run_backtest_from_description(description: str, default_period: str = "1y") -> Dict[str, Any]:
    """
    1. Parse natural language into a config
    2. Load data
    3. Build strategy
    4. Run backtest
    5. Return metrics + results
    """
    config = nl_to_strategy.interpret_natural_language(description)

    ticker = config.get("ticker", "AAPL")
    period = config.get("period", default_period)

    data = data_loader.load_price_data(ticker, period=period)

    strategy = build_strategy_from_config(config)
    engine = BacktestEngine(data, strategy, initial_capital=10000.0)
    results, trades = engine.run()

    # Metrics
    strat_ret = results["strategy_return"].dropna()
    bh_ret = results["bh_return"].dropna()
    strat_equity = results["equity"].dropna()
    bh_equity = results["bh_equity"].dropna()

    metrics = {
        "ticker": ticker,
        "period": period,
        "config": config,
        "strategy_sharpe": sharpe_ratio(strat_ret),
        "strategy_max_dd": max_drawdown(strat_equity),
        "bh_sharpe": sharpe_ratio(bh_ret),
        "bh_max_dd": max_drawdown(bh_equity),
        "final_strategy_equity": float(strat_equity.iloc[-1]),
        "final_bh_equity": float(bh_equity.iloc[-1]),
        "num_trades": int(len(trades)),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
    }

    return {
        "metrics": metrics,
        "results": results,
        "trades": trades,
    }
