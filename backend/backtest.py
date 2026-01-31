import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = None  # today

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
    params: Dict
    win_rate: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    total_return: float
    trades: int


def compute_sharpe(returns: pd.Series):
    if returns.std() == 0 or returns.std() is None:
        return 0.0
    return (returns.mean() / returns.std()) * math.sqrt(252)


def compute_max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min() if len(drawdown) else 0.0


def true_range(high, low, close):
    return pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)


def dual_ma_with_atr_stop(df: pd.DataFrame, fast=20, slow=100, atr_len=14, atr_mult=2.5):
    df = df.copy()
    df["fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    df["atr"] = true_range(df["High"], df["Low"], df["Close"]).rolling(atr_len).mean()

    closes = df["Close"].values.flatten()
    fast_arr = df["fast"].values.flatten()
    slow_arr = df["slow"].values.flatten()
    atr_arr = df["atr"].values.flatten()

    position = 0
    entry = 0.0
    entry_date = None
    equity = [1.0]
    returns = []
    trades: List[Trade] = []

    for i in range(1, len(df)):
        price = float(closes[i])
        prev_price = float(closes[i - 1])
        fast_ma = fast_arr[i]
        slow_ma = slow_arr[i]
        atr_val = atr_arr[i]
        atr = float(atr_val) if not np.isnan(atr_val) else np.nan

        if position == 0 and not np.isnan(fast_ma) and not np.isnan(slow_ma) and fast_ma > slow_ma:
            position = 1
            entry = price
            entry_date = df.index[i]
        elif position == 1:
            stop = entry - atr_mult * atr if not np.isnan(atr) else -np.inf
            if price < stop or (not np.isnan(fast_ma) and not np.isnan(slow_ma) and fast_ma < slow_ma):
                pnl_pct = (price - entry) / entry
                trades.append(Trade(entry_date, df.index[i], entry, price, price - entry, pnl_pct))
                position = 0
                entry = 0
                entry_date = None

        ret = (price - prev_price) / prev_price if position == 1 else 0
        returns.append(ret)
        equity.append(equity[-1] * (1 + ret))

    returns = pd.Series(returns, index=df.index[1:])
    equity = pd.Series(equity[1:], index=df.index[1:])
    return returns, equity, trades


def rsi_reversion(df: pd.DataFrame, length=14, lower=30, upper=70, hold_days=5):
    df = df.copy()
    df["rsi"] = RSIIndicator(df["Close"], window=length).rsi()

    closes = df["Close"].values.flatten()
    rsi_arr = df["rsi"].values.flatten()

    position = 0
    entry = 0.0
    entry_date = None
    hold = 0
    equity = [1.0]
    returns = []
    trades: List[Trade] = []

    for i in range(1, len(df)):
        price = float(closes[i])
        prev_price = float(closes[i - 1])
        rsi = rsi_arr[i]

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
                entry_date = None
                hold = 0

        ret = (price - prev_price) / prev_price if position == 1 else 0
        returns.append(ret)
        equity.append(equity[-1] * (1 + ret))

    returns = pd.Series(returns, index=df.index[1:])
    equity = pd.Series(equity[1:], index=df.index[1:])
    return returns, equity, trades


def donchian_breakout(df: pd.DataFrame, lookback=20, trail=10):
    df = df.copy()
    df["don_high"] = df["High"].rolling(lookback).max()
    df["don_low"] = df["Low"].rolling(lookback).min()

    closes = df["Close"].values.flatten()
    dh_arr = df["don_high"].values.flatten()
    dl_arr = df["don_low"].values.flatten()

    position = 0
    entry = 0.0
    entry_date = None
    equity = [1.0]
    returns = []
    trades: List[Trade] = []
    trail_high = None

    for i in range(1, len(df)):
        price = float(closes[i])
        prev_price = float(closes[i - 1])
        dh = dh_arr[i]
        dl = dl_arr[i]

        if position == 0 and not np.isnan(dh) and price > dh:
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
                entry_date = None
                trail_high = None

        ret = (price - prev_price) / prev_price if position == 1 else 0
        returns.append(ret)
        equity.append(equity[-1] * (1 + ret))

    returns = pd.Series(returns, index=df.index[1:])
    equity = pd.Series(equity[1:], index=df.index[1:])
    return returns, equity, trades


def summarize_strategy(name: str, params: Dict, returns: pd.Series, equity: pd.Series, trades: List[Trade]) -> StrategyResult:
    win_rate = (sum(1 for t in trades if t.pnl_pct > 0) / len(trades)) if trades else 0.0
    gains = sum(t.pnl for t in trades if t.pnl > 0)
    losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gains / losses if losses != 0 else float('inf') if gains > 0 else 0.0
    sharpe = compute_sharpe(returns) if len(returns) else 0.0
    max_dd = compute_max_drawdown(equity)
    total_return = (equity.iloc[-1] - 1.0) if len(equity) else 0.0
    return StrategyResult(name, params, win_rate, profit_factor, sharpe, max_dd, total_return, len(trades))


def bollinger_reversion(df: pd.DataFrame, length=20, num_std=2.0, exit_mid=True, time_stop=5):
    df = df.copy()
    mid = df["Close"].rolling(length).mean()
    std = df["Close"].rolling(length).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std

    closes = df["Close"].values.flatten()
    mid_arr = mid.values.flatten()
    upper_arr = upper.values.flatten()
    lower_arr = lower.values.flatten()

    position = 0
    entry = 0.0
    entry_date = None
    hold = 0
    equity = [1.0]
    returns = []
    trades: List[Trade] = []

    for i in range(1, len(df)):
        price = float(closes[i])
        prev_price = float(closes[i - 1])
        lo = lower_arr[i]
        up = upper_arr[i]
        m = mid_arr[i]

        if position == 0 and not np.isnan(lo) and price < lo:
            position = 1
            entry = price
            entry_date = df.index[i]
            hold = 0
        elif position == 1:
            hold += 1
            exit_cond = False
            if exit_mid and not np.isnan(m) and price >= m:
                exit_cond = True
            if hold >= time_stop:
                exit_cond = True
            if exit_cond:
                pnl_pct = (price - entry) / entry
                trades.append(Trade(entry_date, df.index[i], entry, price, price - entry, pnl_pct))
                position = 0
                entry = 0
                entry_date = None
                hold = 0

        ret = (price - prev_price) / prev_price if position == 1 else 0
        returns.append(ret)
        equity.append(equity[-1] * (1 + ret))

    returns = pd.Series(returns, index=df.index[1:])
    equity = pd.Series(equity[1:], index=df.index[1:])
    return returns, equity, trades


def run_strategy(df: pd.DataFrame, name: str, params: Dict):
    if name == "trend":
        return dual_ma_with_atr_stop(df, **params)
    if name == "reversion":
        return rsi_reversion(df, length=params["length"], lower=params["lower"], upper=params["upper"], hold_days=params["hold"])
    if name == "breakout":
        return donchian_breakout(df, lookback=params["lookback"], trail=params["trail"])
    if name == "bollinger":
        return bollinger_reversion(df, length=params["length"], num_std=params["std"], exit_mid=params["exit_mid"], time_stop=params["time_stop"])
    raise ValueError("unknown strategy")


def sweep_strategy(df_train: pd.DataFrame, df_val: pd.DataFrame, name: str) -> Tuple[StrategyResult, List[StrategyResult], StrategyResult]:
    candidates: List[StrategyResult] = []
    if name == "trend":
        grid = [
            (fast, slow, atr_len, atr_mult)
            for fast in [5, 10, 20]
            for slow in [50, 100, 150, 200]
            for atr_len in [10, 14, 20]
            for atr_mult in [1.5, 2.0, 2.5, 3.0]
            if fast < slow
        ]
        random.shuffle(grid)
        for (fast, slow, atr_len, atr_mult) in grid:
            r, e, t = dual_ma_with_atr_stop(df_train, fast=fast, slow=slow, atr_len=atr_len, atr_mult=atr_mult)
            res = summarize_strategy("trend", {"fast": fast, "slow": slow, "atr_len": atr_len, "atr_mult": atr_mult}, r, e, t)
            candidates.append(res)
    elif name == "reversion":
        grid = [
            (length, lower, upper, hold)
            for length in [7, 10, 14]
            for lower in [20, 25, 30]
            for upper in [60, 65, 70]
            for hold in [3, 5, 7]
        ]
        random.shuffle(grid)
        for (length, lower, upper, hold) in grid:
            r, e, t = rsi_reversion(df_train, length=length, lower=lower, upper=upper, hold_days=hold)
            res = summarize_strategy("reversion", {"length": length, "lower": lower, "upper": upper, "hold": hold}, r, e, t)
            candidates.append(res)
    elif name == "breakout":
        grid = [
            (lookback, trail)
            for lookback in [10, 20, 30, 55]
            for trail in [5, 8, 10, 12]
        ]
        random.shuffle(grid)
        for (lookback, trail) in grid:
            r, e, t = donchian_breakout(df_train, lookback=lookback, trail=trail)
            res = summarize_strategy("breakout", {"lookback": lookback, "trail": trail}, r, e, t)
            candidates.append(res)
    elif name == "bollinger":
        grid = [
            (length, std, exit_mid, time_stop)
            for length in [10, 14, 20]
            for std in [1.5, 2.0, 2.5]
            for exit_mid in [True, False]
            for time_stop in [3, 5, 7]
        ]
        random.shuffle(grid)
        for (length, std, exit_mid, time_stop) in grid:
            r, e, t = bollinger_reversion(df_train, length=length, num_std=std, exit_mid=exit_mid, time_stop=time_stop)
            res = summarize_strategy("bollinger", {"length": length, "std": std, "exit_mid": exit_mid, "time_stop": time_stop}, r, e, t)
            candidates.append(res)
    best = sorted(candidates, key=lambda x: (x.win_rate, x.profit_factor, -x.max_drawdown), reverse=True)[0] if candidates else None
    val_res = None
    if best and df_val is not None and len(df_val) > 20:
        r, e, t = run_strategy(df_val, name, best.params)
        val_res = summarize_strategy(f"{name}_val", best.params, r, e, t)
    return best, candidates, val_res


def run():
    results: Dict[str, Dict] = {}
    for ticker in UNIVERSE:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in df.columns:
                series = df[col]
                if isinstance(series, pd.DataFrame):
                    df[col] = series.iloc[:, 0]
                elif hasattr(series, "values") and len(series.values.shape) > 1:
                    df[col] = series.squeeze()

        split_idx = int(len(df) * 0.7)
        df_train = df.iloc[:split_idx]
        df_val = df.iloc[split_idx:] if split_idx < len(df) else None

        strat_outputs = {}
        all_candidates = {}

        for strat in ["trend", "reversion", "breakout", "bollinger"]:
            best, candidates, val_res = sweep_strategy(df_train, df_val, strat)
            if best:
                strat_outputs[strat] = {
                    "train": vars(best),
                    "validation": vars(val_res) if val_res else None,
                }
                all_candidates[strat] = [vars(c) for c in sorted(candidates, key=lambda x: x.win_rate, reverse=True)[:5]]

        # ensemble via equal weight of best equities on validation if available else train
        equities = []
        for strat in ["trend", "reversion", "breakout", "bollinger"]:
            if strat not in strat_outputs:
                continue
            params = strat_outputs[strat]["train"]["params"]
            target_df = df_val if df_val is not None and len(df_val) > 20 else df_train
            _, e, _ = run_strategy(target_df, strat, params)
            equities.append(e)

        if equities:
            min_len = min(len(eq) for eq in equities)
            aligned = pd.concat([eq.iloc[-min_len:].reset_index(drop=True) for eq in equities], axis=1)
            ensemble_equity = aligned.mean(axis=1)
            ensemble_returns = ensemble_equity.pct_change().fillna(0)
            ensemble_result = summarize_strategy("ensemble", {}, ensemble_returns, ensemble_equity, [])
        else:
            ensemble_result = None

        results[ticker] = {
            "strategies": strat_outputs,
            "top_candidates": all_candidates,
            "ensemble": vars(ensemble_result) if ensemble_result else None,
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
