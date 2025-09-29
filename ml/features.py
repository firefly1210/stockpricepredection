from __future__ import annotations
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = pd.Series(loss, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    # Returns
    df["Return"] = close.pct_change()
    df["LogReturn"] = np.log(close).diff()

    # Moving averages
    for w in (5, 10, 20, 50, 100, 200):
        df[f"SMA_{w}"] = close.rolling(w).mean()
        df[f"EMA_{w}"] = ema(close, w)

    # RSI
    df["RSI_14"] = rsi(close, 14)

    # MACD
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = ema(df["MACD"], 9)
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    sma20 = df["SMA_20"]
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20

    # Volume features
    for w in (5, 10, 20):
        df[f"VOL_SMA_{w}"] = df["Volume"].rolling(w).mean()

    # Lags of close and return
    for lag in range(1, 6):
        df[f"Close_lag{lag}"] = close.shift(lag)
        df[f"Return_lag{lag}"] = df["Return"].shift(lag)

    # Calendar features
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month

    return df
