import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:

    if returns.empty:
        return 0.0

    excess = returns - risk_free_rate / periods_per_year
    mean = excess.mean()
    std = excess.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return np.sqrt(periods_per_year) * mean / std


def max_drawdown(equity_curve: pd.Series) -> float:

    if equity_curve.empty:
        return 0.0

    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1.0
    return dd.min()
