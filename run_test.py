print(">>> Starting backtest script...")

from src.data import data_loader
from src.strategies.factory import create_strategy
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import sharpe_ratio, max_drawdown
from src.backtest.stats import win_rate, profit_factor


def main():
    # === Choose stock & period ===
    raw_ticker = input("Stock (e.g. AAPL) [default: AAPL]: ").strip().upper()
    if raw_ticker == "":
        raw_ticker = "AAPL"

    raw_period = input("Time period (e.g. 6mo, 1y, 2y) [default: 1y]: ").strip()
    if raw_period == "":
        raw_period = "1y"

    # === Choose strategy type ===
    print("\nStrategy types:")
    print("  sma        - Simple moving average crossover")
    print("  ema        - Exponential moving average crossover")
    print("  rsi        - RSI mean reversion")
    print("  bollinger  - Bollinger band reversion")
    print("  macd       - MACD crossover")

    stype = input("Choose strategy [default: sma]: ").strip().lower()
    if stype == "":
        stype = "sma"

    # SIMPLE params for now
    if stype in ("sma", "ema", "macd"):
        fast = input("Fast length [default: 10]: ").strip()
        slow = input("Slow length [default: 20]: ").strip()
        fast = int(fast) if fast else 10
        slow = int(slow) if slow else 20
        params = {"fast": fast, "slow": slow}
    elif stype == "rsi":
        period = input("RSI period [default: 14]: ").strip()
        lower = input("Lower threshold [default: 30]: ").strip()
        upper = input("Upper threshold [default: 70]: ").strip()
        period = int(period) if period else 14
        lower = int(lower) if lower else 30
        upper = int(upper) if upper else 70
        params = {"period": period, "lower": lower, "upper": upper}
    elif stype == "bollinger":
        window = input("Window [default: 20]: ").strip()
        num_std = input("Std dev [default: 2.0]: ").strip()
        window = int(window) if window else 20
        num_std = float(num_std) if num_std else 2.0
        params = {"window": window, "num_std": num_std}
    else:
        raise ValueError(f"Unknown strategy type: {stype}")

    config = {"type": stype, "params": params}

    ticker = raw_ticker
    period = raw_period

    print(f"\n>>> Loading data for {ticker} over {period}...")
    data = data_loader.load_price_data(ticker, period=period)
    print(">>> Data loaded. Rows:", len(data))

    print(f">>> Creating strategy ({stype} with params {params})...")
    strategy = create_strategy(config)

    print(">>> Initializing engine...")
    engine = BacktestEngine(data, strategy, initial_capital=10000.0)

    print(">>> Running backtest...")
    results, trades = engine.run()

    print(">>> Backtest complete. Last 5 rows:")
    print(results.tail())

    print(">>> Trade log (first 10 trades):")
    print(trades.head(10))

    print(">>> Calculating metrics...")

    strat_ret = results["strategy_return"].dropna()
    bh_ret = results["bh_return"].dropna()

    strat_equity = results["equity"].dropna()
    bh_equity = results["bh_equity"].dropna()

    print("Strategy Sharpe:", sharpe_ratio(strat_ret))
    print("Strategy Max Drawdown:", max_drawdown(strat_equity))

    print("Buy & Hold Sharpe:", sharpe_ratio(bh_ret))
    print("Buy & Hold Max Drawdown:", max_drawdown(bh_equity))

    print("Final Strategy Equity:", float(strat_equity.iloc[-1]))
    print("Final Buy & Hold Equity:", float(bh_equity.iloc[-1]))

    print("Number of trades:", len(trades))
    print("Win rate:", win_rate(trades))
    print("Profit factor:", profit_factor(trades))


if __name__ == "__main__":
    main()
