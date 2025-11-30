# AI-Powered Trading Strategy Backtester  
*A modular backtesting engine with AI-driven strategy selection, study evaluation, and performance analytics — built for quantitative research workflows.*

---

## Overview

This project is a fully modular **algorithmic trading backtesting system** designed to evaluate rule-based strategies using historical equities data. It integrates:

- A high-quality **backtesting engine**  
- Multiple technical strategies (SMA, EMA, RSI, Bollinger Bands, MACD)  
- A trade simulator with equity curve analysis  
- A custom **AI Study Selector** that determines which indicators are most relevant for a given stock & timeframe  
- Monte Carlo forecasting for hypothetical long-term projections  

This system was built to merge **finance, AI, and quantitative research** into a single workflow suitable for professional analysis, interview demonstrations, and portfolio research.

---

## Features

### **1. Modular Backtesting Engine**
- Plug-and-play architecture  
- Strategy-independent engine  
- Handles:
  - Entry/exit signals  
  - Portfolio equity tracking  
  - P/L computation  
  - Position management  
  - Buy & Hold benchmarking  

### Metrics Included
- **Sharpe Ratio**  
- **Max Drawdown**  
- **Annualized Volatility**  
- **CAGR (Compounded Annual Growth Rate)**  
- **Win Rate**  
- **Profit Factor**  
- **Return Distribution**  

---

## 📈 **2. Supported Strategies**
| Strategy | Description |
|---------|-------------|
| SMA Crossover | Trend following via moving average crosses |
| EMA Crossover | Faster trend detection using exponential weighting |
| RSI Reversion | Overbought/oversold mean reversion |
| Bollinger Bands | Volatility breakout / reversion signals |
| MACD | Trend-following momentum strategy |

Each strategy inherits from a unified `Strategy` class, allowing easy extension and testing of new rule sets.

---

## **3. AI Study Selector**
The AI module evaluates a stock and returns:

- Best-performing indicators  
- Ranked by composite score:
  - Sharpe  
  - Stability  
  - Drawdown  
  - Consistency  
- A natural-language explanation of why those indicators performed best  

This transforms the system into a **quantitative research assistant**, not just a backtester.

Example output:

> “For AAPL (1-year), Bollinger Bands and RSI provide the highest predictive stability due to cyclical volatility and consistent mean reversion. SMA/EMA underperform due to chop-heavy periods.”

---

Launch the interface with:

streamlit run app.py
