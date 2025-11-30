print(">>> AI Backtester")

from src.ai.runner import run_backtest_from_description


def main():
    # === Step 1: Ask for stock and time, like normal backtest ===
    raw_ticker = input("Stock (e.g. AAPL) [default: AAPL]: ").strip().upper()
    if raw_ticker == "":
        raw_ticker = "AAPL"

    raw_period = input("Time period (e.g. 6mo, 1y, 2y) [default: 1y]: ").strip()
    if raw_period == "":
        raw_period = "1y"

    # === Step 2: Ask for strategy description (just the rule) ===
    # Example: "10/30 SMA crossover"
    rule_text = input("Strategy (e.g. '10/30 SMA crossover'): ").strip()
    if rule_text == "":
        rule_text = "10/20 SMA crossover"

    # Build a natural-language description that the parser understands
    description = f"Trade {raw_ticker} with {rule_text}"

    print("\n>>> Running AI backtest with:")
    print("    Stock:", raw_ticker)
    print("    Period:", raw_period)
    print("    Strategy description:", description)

    # === Step 3: Run the AI pipeline ===
    output = run_backtest_from_description(description, default_period=raw_period)

    metrics = output["metrics"]
    results = output["results"]
    trades = output["trades"]

    print("\n>>> Parsed config:", metrics["config"])
    print(">>> Ticker:", metrics["ticker"], "| Period:", metrics["period"])

    print(">>> Backtest complete. Last 5 rows:")
    print(results.tail())

    print(">>> Trade log (first 10 trades):")
    print(trades.head(10))

    print(">>> Metrics:")
    print("Strategy Sharpe:", metrics["strategy_sharpe"])
    print("Strategy Max Drawdown:", metrics["strategy_max_dd"])
    print("Buy & Hold Sharpe:", metrics["bh_sharpe"])
    print("Buy & Hold Max Drawdown:", metrics["bh_max_dd"])
    print("Final Strategy Equity:", metrics["final_strategy_equity"])
    print("Final Buy & Hold Equity:", metrics["final_bh_equity"])
    print("Number of trades:", metrics["num_trades"])
    print("Win rate:", metrics["win_rate"])
    print("Profit factor:", metrics["profit_factor"])


if __name__ == "__main__":
    main()
