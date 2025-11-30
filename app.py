import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.data import data_loader
from src.strategies.factory import create_strategy
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import sharpe_ratio, max_drawdown
from src.backtest.stats import win_rate, profit_factor
from src.ai.study_selector import evaluate_strategies_for_ticker, rank_strategies

# ---------- DEFAULTS ----------

DEFAULT_STRATEGY_CONFIGS = {
    "sma": {"fast": 10, "slow": 30},
    "ema": {"fast": 12, "slow": 26},
    "rsi": {"period": 14, "lower": 30, "upper": 70},
    "bollinger": {"window": 20, "num_std": 2.0},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
}

INITIAL_CAPITAL_DEFAULT = 10_000.0


# ---------- METRIC HELPERS ----------

def calc_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if equity.empty:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def calc_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def calc_sortino(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if downside.std() == 0 or np.isnan(downside.std()):
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / downside.std()


def monte_carlo_projection(
    returns: pd.Series,
    start_equity: float,
    years: int = 5,
    periods_per_year: int = 252,
    sims: int = 200,
) -> pd.DataFrame:
    """
    Simple Monte Carlo projection based on historical mean/std of strategy returns.
    Not financial advice, just a toy model.
    """
    if returns.empty:
        return pd.DataFrame()

    mu = returns.mean()
    sigma = returns.std()
    n_periods = years * periods_per_year

    simulated_paths = []
    for _ in range(sims):
        rand_rets = np.random.normal(mu, sigma, n_periods)
        equity = start_equity * (1 + pd.Series(rand_rets)).cumprod()
        simulated_paths.append(equity.values)

    idx = pd.RangeIndex(start=1, stop=n_periods + 1)
    sim_df = pd.DataFrame(np.column_stack(simulated_paths), index=idx)
    return sim_df


# ---------- STREAMLIT SETUP & THEME ----------

st.set_page_config(page_title="AI Backtester & Futuristic Evaluator", layout="wide")

# Dark / terminal-like CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #050608;
    }
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4 {
        font-family: "Menlo", "Consolas", monospace;
        color: #f5f5f5;
    }
    .headline {
        font-size: 32px;
        font-weight: 700;
        color: #ff914d;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .subheadline {
        font-size: 14px;
        color: #cccccc;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111318;
        border-radius: 3px 3px 0 0;
        padding-top: 4px;
        padding-bottom: 4px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #f5b400;
    }
    [data-testid="stMetricLabel"] {
        font-family: "Menlo", "Consolas", monospace;
        color: #aaaaaa;
    }
    [data-testid="stMetricValue"] {
        font-family: "Menlo", "Consolas", monospace;
        color: #f5b400;
    }
    .css-1y4p8pa {background-color: #050608;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Use a dark style for matplotlib as a base
plt.style.use("dark_background")

# ---------- HEADER ----------

st.markdown(
    '<div class="headline">AI-POWERED TRADING STRATEGY BACKTESTER</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subheadline">A modular backtesting engine with AI-driven study selection, performance analytics, and futuristic scenario analysis.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(
    [
        "Backtest & Deep Insights",
        "AI Study Advisor",
        "Futuristic Projections",
    ]
)

# ---------------- TAB 1: BACKTEST & INSIGHTS ----------------
with tab1:
    st.subheader("Backtest & Deep Insights")

    col_inputs = st.columns(4)
    with col_inputs[0]:
        ticker = st.text_input("Stock", value="AAPL").upper()
    with col_inputs[1]:
        period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1)
    with col_inputs[2]:
        stype = st.selectbox(
            "Strategy",
            ["sma", "ema", "rsi", "bollinger", "macd"],
            index=0,
        )
    with col_inputs[3]:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000.0,
            max_value=1_000_000.0,
            value=INITIAL_CAPITAL_DEFAULT,
            step=1000.0,
            format="%.2f",
        )

    params = DEFAULT_STRATEGY_CONFIGS[stype]

    run_bt = st.button("🚀 Run Backtest", type="primary")

    if run_bt:
        # Load data
        data = data_loader.load_price_data(ticker, period=period)

        # Build and run strategy
        config = {"type": stype, "params": params}
        strategy = create_strategy(config)
        engine = BacktestEngine(data, strategy, initial_capital=initial_capital)
        results, trades = engine.run()

        # Store in session_state for other tabs
        st.session_state["last_results"] = results
        st.session_state["last_trades"] = trades
        st.session_state["last_ticker"] = ticker
        st.session_state["last_period"] = period
        st.session_state["last_stype"] = stype
        st.session_state["last_initial_capital"] = initial_capital

        strat_ret = results["strategy_return"].dropna()
        bh_ret = results["bh_return"].dropna()
        strat_equity = results["equity"].dropna()
        bh_equity = results["bh_equity"].dropna()

        # ---- METRICS PANEL ----
        st.subheader("Performance Snapshot")

        strat_cagr = calc_cagr(strat_equity)
        bh_cagr = calc_cagr(bh_equity)
        strat_vol = calc_volatility(strat_ret)
        bh_vol = calc_volatility(bh_ret)
        strat_sortino = calc_sortino(strat_ret)
        bh_sortino = calc_sortino(bh_ret)

        col_top1, col_top2, col_top3, col_top4 = st.columns(4)
        with col_top1:
            st.metric("Strategy Sharpe", f"{sharpe_ratio(strat_ret):.2f}")
            st.metric("Strategy Sortino", f"{strat_sortino:.2f}")
        with col_top2:
            st.metric("Strategy Max Drawdown", f"{max_drawdown(strat_equity):.2%}")
            st.metric("Strategy Volatility", f"{strat_vol:.2%}")
        with col_top3:
            st.metric("Final Strategy Equity", f"${float(strat_equity.iloc[-1]):,.2f}")
            st.metric("Strategy CAGR", f"{strat_cagr:.2%}")
        with col_top4:
            st.metric("Buy & Hold Sharpe", f"{sharpe_ratio(bh_ret):.2f}")
            st.metric("Buy & Hold CAGR", f"{bh_cagr:.2%}")

        # ---- FACTS & INSIGHTS ----
        st.subheader("Key Facts & Insights")

        total_ret_strat = strat_equity.iloc[-1] / strat_equity.iloc[0] - 1
        total_ret_bh = bh_equity.iloc[-1] / bh_equity.iloc[0] - 1

        best_day = strat_ret.idxmax() if not strat_ret.empty else None
        worst_day = strat_ret.idxmin() if not strat_ret.empty else None
        best_ret = strat_ret.max() if not strat_ret.empty else 0.0
        worst_ret = strat_ret.min() if not strat_ret.empty else 0.0

        pos_days = (strat_ret > 0).sum()
        neg_days = (strat_ret < 0).sum()
        total_days = len(strat_ret)
        pct_pos = pos_days / total_days if total_days > 0 else 0.0

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(
                f"""
                **Over the selected period for `{ticker}`:**

                - Strategy total return: **{total_ret_strat:.2%}**  
                - Buy & hold total return: **{total_ret_bh:.2%}**  
                - Strategy had **{pos_days}** positive days and **{neg_days}** negative days  
                - % of positive days: **{pct_pos:.1%}**  
                """
            )
        with col_f2:
            st.markdown(
                f"""
                **Daily return extremes (strategy):**

                - Best day: **{best_ret:.2%}** {'on ' + best_day.strftime('%Y-%m-%d') if best_day is not None else ''}  
                - Worst day: **{worst_ret:.2%}** {'on ' + worst_day.strftime('%Y-%m-%d') if worst_day is not None else ''}  
                - Trade count: **{len(trades)}**  
                - Win rate: **{win_rate(trades):.1%}**  
                - Profit factor: **{profit_factor(trades):.2f}**  
                """
            )

        # ---- EQUITY CURVE ----
        st.subheader("Equity Curve (Strategy vs Buy & Hold)")
        fig_eq, ax_eq = plt.subplots(figsize=(8, 3), facecolor="#050608")
        ax_eq.set_facecolor("#050608")
        ax_eq.plot(
            strat_equity.index,
            strat_equity.values,
            label="Strategy",
            color="#00ff7f",
            linewidth=1.5,
        )
        ax_eq.plot(
            bh_equity.index,
            bh_equity.values,
            label="Buy & Hold",
            color="#f5b400",
            linewidth=1.2,
            linestyle="--",
        )
        ax_eq.set_xlabel("Date", color="#dddddd")
        ax_eq.set_ylabel("Equity ($)", color="#dddddd")
        ax_eq.tick_params(colors="#bbbbbb")
        ax_eq.legend(facecolor="#111111", edgecolor="#333333")
        ax_eq.grid(alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig_eq)

        # ---- RETURN DISTRIBUTION ----
        st.subheader("Distribution of Strategy Daily Returns")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 3), facecolor="#050608")
        ax_hist.set_facecolor("#050608")
        ax_hist.hist(strat_ret, bins=40, color="#00ff7f", alpha=0.85)
        ax_hist.set_xlabel("Daily Return", color="#dddddd")
        ax_hist.set_ylabel("Frequency", color="#dddddd")
        ax_hist.tick_params(colors="#bbbbbb")
        ax_hist.grid(alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig_hist)

        # ---- TRADE LOG ----
        with st.expander("🔍 View Trade Log"):
            st.dataframe(trades)

    else:
        st.info("Run a backtest to see detailed insights.")


# ---------------- TAB 2: AI STUDY ADVISOR ----------------
with tab2:
    st.subheader("AI Study Advisor – Which Indicators Work Best Here?")

    col1, col2 = st.columns(2)
    with col1:
        ticker2 = st.text_input("Stock (advisor)", value="AAPL").upper()
    with col2:
        period2 = st.selectbox("Period (advisor)", ["6mo", "1y", "2y"], index=1)

    risk_focus = st.selectbox(
        "Risk focus",
        ["balanced", "return", "defensive"],
        index=0,
        help="balanced: mix of return & risk • return: favor Sharpe & final equity • defensive: favor lower drawdown",
    )

    if st.button("🤖 Evaluate Studies"):
        df = evaluate_strategies_for_ticker(ticker2, period=period2)
        st.subheader("Raw Strategy Metrics")
        st.dataframe(df)

        ranked = rank_strategies(df, risk_focus=risk_focus)
        if ranked.empty:
            st.warning("No valid strategies to rank.")
        else:
            st.subheader("Ranked Strategies (Best → Worst)")
            st.dataframe(
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

            top = ranked.iloc[0]
            st.success(
                f"For **{ticker2}** over **{period2}**, the **{top['strategy']}** study performed best "
                f"with Sharpe **{top['sharpe']:.2f}**, max drawdown **{top['max_drawdown']:.2%}**, "
                f"final equity **${top['final_equity']:.2f}**, win rate **{top['win_rate']:.0%}**, "
                f"and profit factor **{top['profit_factor']:.2f}**.\n\n"
                f"In plain English: historically, `{top['strategy']}` has been one of the most informative "
                f"studies to base entry/exit decisions on for this stock and horizon."
            )


# ---------------- TAB 3: FUTURISTIC PROJECTIONS ----------------
with tab3:
    st.subheader("Futuristic Projections & Scenario Analysis")
    st.caption(
        "These projections are purely hypothetical, based on historical strategy returns. "
        "They are NOT predictions or financial advice."
    )

    if "last_results" not in st.session_state:
        st.info("Run a backtest in the first tab to unlock futuristic projections.")
    else:
        results = st.session_state["last_results"]
        ticker3 = st.session_state.get("last_ticker", "AAPL")
        period3 = st.session_state.get("last_period", "1y")
        stype3 = st.session_state.get("last_stype", "sma")

        strat_ret = results["strategy_return"].dropna()
        strat_equity = results["equity"].dropna()

        col_mc1, col_mc2 = st.columns(2)
        with col_mc1:
            years = st.slider("Projection horizon (years)", 1, 10, 5)
        with col_mc2:
            sims = st.slider("Number of Monte Carlo simulations", 50, 500, 200, step=50)

        sim_df = monte_carlo_projection(
            strat_ret, start_equity=float(strat_equity.iloc[-1]), years=years, sims=sims
        )
        if sim_df.empty:
            st.warning("Not enough data to generate projections.")
        else:
            st.subheader(f"Hypothetical Future Paths for {ticker3} ({stype3} strategy)")

            # Plot a subset of paths for readability
            fig_mc, ax_mc = plt.subplots(figsize=(8, 3), facecolor="#050608")
            ax_mc.set_facecolor("#050608")
            n_plot = min(20, sim_df.shape[1])
            ax_mc.plot(sim_df.index, sim_df.iloc[:, :n_plot], linewidth=0.7, alpha=0.7)
            ax_mc.set_xlabel("Days into the future", color="#dddddd")
            ax_mc.set_ylabel("Equity ($)", color="#dddddd")
            ax_mc.tick_params(colors="#bbbbbb")
            ax_mc.grid(alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig_mc)

            # Percentile bands
            final_values = sim_df.iloc[-1, :]
            p10 = np.percentile(final_values, 10)
            p50 = np.percentile(final_values, 50)
            p90 = np.percentile(final_values, 90)

            st.subheader("Future Equity Distribution (at end of horizon)")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric(f"10th percentile after {years}y", f"${p10:,.2f}")
            with col_p2:
                st.metric(f"Median after {years}y", f"${p50:,.2f}")
            with col_p3:
                st.metric(f"90th percentile after {years}y", f"${p90:,.2f}")

            st.markdown(
                f"""
                **Interpretation (purely hypothetical):**

                - If the past behavior of this strategy persisted, there is a wide range of plausible futures.  
                - In this toy model, **10%** of simulated outcomes end below **${p10:,.0f}**,  
                  while **10%** end above **${p90:,.0f}** after **{years}** years.  
                - The median simulated outcome is around **${p50:,.0f}**.
                """
            )
