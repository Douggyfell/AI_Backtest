print(">>> AI Study Recommender")

from src.ai.study_selector import evaluate_strategies_for_ticker, rank_strategies


def main():
    raw_ticker = input("Stock (e.g. AAPL) [default: AAPL]: ").strip().upper()
    if raw_ticker == "":
        raw_ticker = "AAPL"

    raw_period = input("Time period (e.g. 6mo, 1y, 2y) [default: 1y]: ").strip()
    if raw_period == "":
        raw_period = "1y"

    print("\nRisk focus options:")
    print("  return    - prioritize return and Sharpe")
    print("  defensive - prioritize lower drawdown")
    print("  balanced  - mix of both")

    risk_focus = input("Choose risk focus [default: balanced]: ").strip().lower()
    if risk_focus == "":
        risk_focus = "balanced"

    ticker = raw_ticker
    period = raw_period

    print(f"\n>>> Evaluating strategies for {ticker} over {period}...")
    df = evaluate_strategies_for_ticker(ticker, period=period)

    print("\n>>> Raw metrics for each study/strategy:")
    print(df)

    ranked = rank_strategies(df, risk_focus=risk_focus)

    if ranked.empty:
        print("\nNo valid strategies to rank (check implementation).")
        return

    print("\n>>> Ranked strategies (best to worst) based on", risk_focus, "objective:")
    print(
        ranked[
            [
                "strategy",
                "sharpe",
                "max_drawdown",
                "final_equity",
                "num_trades",
                "win_rate",
                "profit_factor",
                "score",
            ]
        ]
    )

    # Simple natural-language style recommendation
    top = ranked.iloc[0]
    print("\n>>> Recommendation:")
    print(
        f"For {ticker} over {period}, the {top['strategy']} study performed best "
        f"with Sharpe {top['sharpe']:.2f}, max drawdown {top['max_drawdown']:.2%}, "
        f"final equity ${top['final_equity']:.2f}, win rate {top['win_rate']:.0%}, "
        f"and profit factor {top['profit_factor']:.2f}."
    )


if __name__ == "__main__":
    main()
