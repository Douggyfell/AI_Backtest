from src.strategies.sma_cross import SMACross
from src.strategies.ema_cross import EMACross
from src.strategies.rsi import RSIStrategy
from src.strategies.bollinger import BollingerReversion
from src.strategies.macd import MACDStrategy


def create_strategy(config: dict):

    stype = config.get("type")
    params = config.get("params", {}) or {}

    if stype == "sma":
        return SMACross(
            fast=params.get("fast", 10),
            slow=params.get("slow", 20),
        )
    elif stype == "ema":
        return EMACross(
            fast=params.get("fast", 12),
            slow=params.get("slow", 26),
        )
    elif stype == "rsi":
        return RSIStrategy(
            period=params.get("period", 14),
            lower=params.get("lower", 30),
            upper=params.get("upper", 70),
        )
    elif stype == "bollinger":
        return BollingerReversion(
            window=params.get("window", 20),
            num_std=params.get("num_std", 2.0),
        )
    elif stype == "macd":
        return MACDStrategy(
            fast=params.get("fast", 12),
            slow=params.get("slow", 26),
            signal_period=params.get("signal", 9),
        )
    else:
        raise ValueError(f"Unknown strategy type: {stype}")
