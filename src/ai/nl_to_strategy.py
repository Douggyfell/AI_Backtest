import re

def interpret_natural_language(text: str) -> dict:

    t = text.lower()

    # Ticker guess: first ALL-CAPS word with 1–5 letters
    ticker_match = re.search(r"\b([A-Z]{1,5})\b", text)
    ticker = ticker_match.group(1) if ticker_match else "AAPL"

    # --- SMA strategy ---
    if "sma" in t or "moving average" in t:
        nums = [int(n) for n in re.findall(r"\b\d+\b", text)]
        fast, slow = 10, 20
        if len(nums) >= 2:
            fast, slow = sorted(nums[:2])
        elif len(nums) == 1:
            fast = nums[0]
            slow = fast * 2

        return {
            "ticker": ticker,
            "type": "sma",
            "params": {"fast": fast, "slow": slow},
        }

    # --- RSI strategy ---
    if "rsi" in t:
        nums = [int(n) for n in re.findall(r"\b\d+\b", text)]
        period, lower, upper = 14, 30, 70

        if len(nums) >= 1:
            period = nums[0]
        if len(nums) >= 3:
            lower, upper = nums[1], nums[2]

        return {
            "ticker": ticker,
            "type": "rsi",
            "params": {"period": period, "lower": lower, "upper": upper},
        }

    # --- Fallback: default SMA on ticker ---
    return {
        "ticker": ticker,
        "type": "sma",
        "params": {"fast": 10, "slow": 20},
    }
