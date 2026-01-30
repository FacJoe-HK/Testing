import json
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = None  # defaults to today

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float

@dataclass
class StrategyResult:
    name: str
    win_rate: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    total_return: float
    trades: int


def compute_sharpe(returns: pd.Series):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * math.sqrt(252)


def compute_max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min() if len(drawdown) else 0.0


def dual_ma_with_atr_stop(df: pd.DataFrame, fast=20, slow=100, atr_len=14, atr_mult=2.5):
    df = df.copy()
    df["fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    df["tr"] = pd.concat(
        [
          (high - low),
          (high - close.shift(1)).abs(),
          (low - close.shift(1)).abs(),
        ], axis=1
    ).max(axis=1)
    df["atr"] = df["tr"].rolling(atr_len).mean()

    position = 0
    entry = 0.0
    equity = [1.0]
    returns = []
    trades: List[Trade] = []

    closes = df["Close"].values.flatten()
    fast_arr = df["fast"].values.flatten()
    slow_arr = df["slow"].values.flatten()
    atr_arr = df["atr"].values.flatten()

    for i in range(1, len(df)):
        price = float(closes[i])
        prev_price = float(closes[i - 1])
        fast_ma = fast_arr[i]
        slow_ma = slow_arr[i]
        atr_val = atr_arr[i]
        atr = float(atr_val) if not np.isnan(atr_val) else np.nan

        # signals
        if position == 0 and not np.isnan(fast_ma) and not np.isnan(slow_ma) and fast_ma > slow_ma:
            position = 1
            entry = price
            entry_date = df.index[i]
        elif position == 1:
            stop = entry - atr_mult * atr if not np.isnan(atr) else -np.inf
            if price < stop or (not np.isnan(fast_ma) and not np.isnan(slow_ma) and fast_ma < slow_ma):
                # exit
                pnl_pct = (price - entry) / entry
                trades.append(Trade(entry_date, df.index[i], entry, price, price - entry, pnl_pct))
                position = 0
                entry = 0

        # daily equity update
        if position == 1:
            ret = (price - prev_price) / prev_price
        else:
            ret = 0
        returns.append(ret)
        equity.append(equity[-1] * (1 + ret))

    returns = pd.Series(returns, index=df.index[1:])
    equity = pd.Series(equity[1:], index=df.index[1:])
    return returns, equity, trades


def rsi_reversion(df: pd.DataFrame, length=14, lower=30, upper=70, hold_days=5):
    df = df.copy()
    df["rsi"] = RSIIndicator(df["Close"], window=length).rsi()

    position = 0
    hold = 0
    entry = 0.0
    equity = [1.0]
    returns = []
    trades: List[Trade] = []

    for i in range(1, len(df)):
        price = df.iloc[i]["Close"]
        prev_price = df.iloc[i - 1]["Close"]
        rsi = df.iloc[i]["rsi"]

        if position == 0 and rsi < lower:
            position = 1
            entry = price
            entry_date = df.index[i]
            hold = 0
        elif position == 1:
            hold += 1
            if rsi > upper or hold >= hold_days:
                pnl_pct = (price - entry) / entry
                trades.append(Trade(entry_date, df.index[i], entry, price, price - entry, pnl_pct))
                position = 0
                entry = 0
                hold = 0

        if position == 1:
            ret = (price - prev_price) / prev_price
        else:
            ret = 0
        returns.append(ret)
        equity.append(equity[-1] * (1 + ret))

    returns = pd.Series(returns, index=df.index[1:])
    equity = pd.Series(equity[1:], index=df.index[1:])
    return returns, equity, trades


def donchian_breakout(df: pd.DataFrame, lookback=20, trail=10):
    df = df.copy()
    df["don_high"] = df["High"].rolling(lookback).max()
    df["don_low"] = df["Low"].rolling(lookback).min()

    position = 0
    entry = 0.0
    equity = [1.0]
    returns = []
    trades: List[Trade] = []

    trail_high = None

    for i in range(1, len(df)):
        price = df.iloc[i]["Close"]
        prev_price = df.iloc[i - 1]["Close"]
        dh = df.iloc[i]["don_high"]
        dl = df.iloc[i]["don_low"]

        if position == 0 and not math.isnan(dh) and price > dh:
            position = 1
            entry = price
            entry_date = df.index[i]
            trail_high = price
        elif position == 1:
            trail_high = max(trail_high, price)
            stop = trail_high * (1 - 0.01 * trail)
            if price < stop:
                pnl_pct = (price - entry) / entry
                trades.append(Trade(entry_date, df.index[i], entry, price, price - entry, pnl_pct))
                position = 0
                entry = 0
                trail_high = None

        if position == 1:
            ret = (price - prev_price) / prev_price
        else:
            ret = 0
        returns.append(ret)
        equity.append(equity[-1] * (1 + ret))

    returns = pd.Series(returns, index=df.index[1:])
    equity = pd.Series(equity[1:], index=df.index[1:])
    return returns, equity, trades


def summarize_strategy(name: str, returns: pd.Series, equity: pd.Series, trades: List[Trade]) -> StrategyResult:
    win_rate = (sum(1 for t in trades if t.pnl_pct > 0) / len(trades)) if trades else 0.0
    gains = sum(t.pnl for t in trades if t.pnl > 0)
    losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gains / losses if losses != 0 else float('inf') if gains > 0 else 0.0
    sharpe = compute_sharpe(returns) if len(returns) else 0.0
    max_dd = compute_max_drawdown(equity)
    total_return = (equity.iloc[-1] - 1.0) if len(equity) else 0.0
    return StrategyResult(name, win_rate, profit_factor, sharpe, max_dd, total_return, len(trades))


def run():
    results: Dict[str, Dict] = {}
    for ticker in UNIVERSE:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        if df.empty:
            continue

        # Normalize columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in df.columns:
                series = df[col]
                if isinstance(series, pd.DataFrame):
                    df[col] = series.iloc[:, 0]
                elif hasattr(series, "values") and len(series.values.shape) > 1:
                    df[col] = series.squeeze()

        # Strategies
        strat_outputs = {}

        r_returns, r_equity, r_trades = dual_ma_with_atr_stop(df)
        strat_outputs["trend"] = summarize_strategy("trend", r_returns, r_equity, r_trades)

        rsi_returns, rsi_equity, rsi_trades = rsi_reversion(df)
        strat_outputs["reversion"] = summarize_strategy("reversion", rsi_returns, rsi_equity, rsi_trades)

        don_returns, don_equity, don_trades = donchian_breakout(df)
        strat_outputs["breakout"] = summarize_strategy("breakout", don_returns, don_equity, don_trades)

        # Ensemble: simple average of signals via equity average
        equities = [r_equity, rsi_equity, don_equity]
        min_len = min(len(eq) for eq in equities)
        if min_len == 0:
            continue
        aligned = pd.concat([eq.iloc[-min_len:].reset_index(drop=True) for eq in equities], axis=1)
        ensemble_equity = aligned.mean(axis=1)
        ensemble_returns = ensemble_equity.pct_change().fillna(0)
        ensemble_result = summarize_strategy("ensemble", ensemble_returns, ensemble_equity, [])

        results[ticker] = {
            "strategies": {k: vars(v) for k, v in strat_outputs.items()},
            "ensemble": vars(ensemble_result),
        }

    output = {
        "universe": UNIVERSE,
        "start": START_DATE,
        "end": END_DATE,
        "results": results,
    }

    with open("../public/backtest.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Wrote ../public/backtest.json")


if __name__ == "__main__":
    run()
