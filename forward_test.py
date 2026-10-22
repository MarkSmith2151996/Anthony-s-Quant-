#!/usr/bin/env python3
"""
HTF Swing v3 — Forward Test Tracker

Pulls latest MNQ 1m data from Databento, runs the full 3-strategy
HTF Swing v3 system (RSI + IB Breakout + Momentum), and tracks
forward performance from a start date.

All 3 sub-strategies run in parallel (unlike TradingView's 1-position limit).

Usage:
    python3 forward_test.py              # Run with defaults (pulls latest data, shows dashboard)
    python3 forward_test.py --pull       # Force fresh data pull from Databento
    python3 forward_test.py --dashboard  # Just show dashboard from last run
    python3 forward_test.py --reset      # Reset forward test start date to today
"""

import json
import os
import sys
import time as _time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "processed" / "MNQ" / "1m"
STATE_FILE = ROOT / "forward_test_state.json"
REPORTS_DIR = ROOT / "reports" / "forward_test"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# COST MODEL (matches Berj's exactly)
# ═══════════════════════════════════════════════════════════════════

TICK_SIZE = 0.25
POINT_VALUE = 2.0       # MNQ
SLIP_TICKS = 2          # per side
COMM_PER_SIDE = 0.62
EXCH_PER_SIDE = 0.27
SLIP_PTS = SLIP_TICKS * TICK_SIZE  # 0.50

def rt_cost(c: int) -> float:
    """Round-trip cost for c contracts."""
    return (COMM_PER_SIDE + EXCH_PER_SIDE) * 2 * c + SLIP_TICKS * TICK_SIZE * POINT_VALUE * 2 * c

# ═══════════════════════════════════════════════════════════════════
# TRADE DATACLASS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    direction: int
    entry_px: float
    exit_px: float
    contracts: int
    net_pnl: float
    entry_time: object
    exit_time: object
    bars_held: int
    reason: str
    strategy: str

# ═══════════════════════════════════════════════════════════════════
# INDICATORS (vectorized, numpy — exact copies from Berj's code)
# ═══════════════════════════════════════════════════════════════════

def calc_vwap(df: pl.DataFrame) -> np.ndarray:
    """Daily VWAP (resets each date_et)."""
    tp = (df["high"].to_numpy() + df["low"].to_numpy() + df["close"].to_numpy()) / 3
    vol = df["volume"].to_numpy().astype(float)
    dates = df["date_et"].to_list()
    vwap = np.full(len(df), np.nan)
    cum_tpv = 0.0
    cum_vol = 0.0
    prev_date = None
    for i in range(len(df)):
        if dates[i] != prev_date:
            cum_tpv = 0.0
            cum_vol = 0.0
            prev_date = dates[i]
        cum_tpv += tp[i] * vol[i]
        cum_vol += vol[i]
        vwap[i] = cum_tpv / cum_vol if cum_vol > 0 else tp[i]
    return vwap


def calc_atr(highs, lows, closes, period=14):
    n = len(highs)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    tr[0] = highs[0] - lows[0]
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def calc_ema(data, period):
    n = len(data)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    ema[period-1] = np.mean(data[:period])
    k = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = data[i] * k + ema[i-1] * (1 - k)
    return ema


def calc_rsi(closes, period=14):
    n = len(closes)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = np.mean(gains[:period])
    avg_l = np.mean(losses[:period])
    if avg_l == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100 - 100 / (1 + avg_g / avg_l)
    for i in range(period, n - 1):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        if avg_l == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100 - 100 / (1 + avg_g / avg_l)
    return rsi


# ═══════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (exact copy from Berj)
# ═══════════════════════════════════════════════════════════════════

def backtest(
    opens, highs, lows, closes, timestamps, hhmm_arr,
    signals, sl_ticks, tp_ticks, max_hold, contracts,
    strategy_name="", flatten_time=1545,
) -> list[Trade]:
    """Honest bar-by-bar backtest. No same-bar exits. Conservative ordering."""
    n = len(opens)
    cost = rt_cost(contracts)
    trades = []
    in_pos = False
    direction = 0
    entry_px = 0.0
    entry_bar = 0
    stop_px = 0.0
    target_px = 0.0
    pending = 0

    for i in range(n):
        h = hhmm_arr[i] if i < len(hhmm_arr) else 0

        # Flatten near close
        if in_pos and h >= flatten_time and i > entry_bar:
            ex = closes[i] - direction * SLIP_PTS
            raw = (ex - entry_px) * direction * POINT_VALUE * contracts
            trades.append(Trade(direction, entry_px, ex, contracts, raw - cost,
                                timestamps[entry_bar], timestamps[i], i - entry_bar,
                                "time_exit", strategy_name))
            in_pos = False
            pending = 0
            continue

        # Execute pending
        if pending != 0 and not in_pos:
            if h >= flatten_time - 15:
                pending = 0
            else:
                entry_px = opens[i] + int(pending) * SLIP_PTS
                direction = int(pending)
                entry_bar = i
                stop_px = entry_px - direction * sl_ticks * TICK_SIZE
                target_px = entry_px + direction * tp_ticks * TICK_SIZE
                in_pos = True
                pending = 0

        # Manage (skip entry bar)
        if in_pos and i > entry_bar:
            bh = i - entry_bar
            ex = None
            reason = ""

            if direction == 1 and lows[i] <= stop_px:
                ex = stop_px; reason = "stop_loss"
            elif direction == -1 and highs[i] >= stop_px:
                ex = stop_px; reason = "stop_loss"

            if ex is None:
                if direction == 1 and highs[i] >= target_px:
                    ex = target_px; reason = "take_profit"
                elif direction == -1 and lows[i] <= target_px:
                    ex = target_px; reason = "take_profit"

            if ex is None and bh >= max_hold:
                ex = closes[i]; reason = "max_hold"

            if ex is not None:
                ex -= direction * SLIP_PTS
                raw = (ex - entry_px) * direction * POINT_VALUE * contracts
                trades.append(Trade(direction, entry_px, ex, contracts, raw - cost,
                                    timestamps[entry_bar], timestamps[i], bh,
                                    reason, strategy_name))
                in_pos = False

        # New signal
        if not in_pos and i < len(signals) and signals[i] != 0:
            pending = signals[i]

    return trades


# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATORS (exact copies from Berj)
# ═══════════════════════════════════════════════════════════════════

def sig_rsi_extreme(df, period=7, ob=70, os_=30):
    closes = df["close"].to_numpy()
    rsi = calc_rsi(closes, period)
    n = len(closes)
    sigs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if np.isnan(rsi[i]):
            continue
        if rsi[i] < os_:
            sigs[i] = 1
        elif rsi[i] > ob:
            sigs[i] = -1
    return sigs


def sig_ib_breakout(df, ib_range_filter=True):
    """IB breakout on 15m bars. IB = 9:30-10:00 = first 2 bars."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    hhmm = df["hhmm"].to_numpy()
    dates = df["date_et"].to_list()
    n = len(df)

    ib_data = {}
    for i in range(n):
        if hhmm[i] < 1000:
            d = dates[i]
            if d not in ib_data:
                ib_data[d] = {"high": highs[i], "low": lows[i]}
            else:
                ib_data[d]["high"] = max(ib_data[d]["high"], highs[i])
                ib_data[d]["low"] = min(ib_data[d]["low"], lows[i])

    ib_ranges = []
    ib_dates = sorted(ib_data.keys())
    for d in ib_dates:
        ib_ranges.append(ib_data[d]["high"] - ib_data[d]["low"])

    sigs = np.zeros(n, dtype=np.int8)
    traded = {}

    for i in range(n):
        if hhmm[i] < 1000 or hhmm[i] >= 1530:
            continue
        d = dates[i]
        if d not in ib_data or d in traded:
            continue

        ib_h = ib_data[d]["high"]
        ib_l = ib_data[d]["low"]
        ib_r = ib_h - ib_l

        if ib_r < 2.0:
            continue

        if ib_range_filter and len(ib_ranges) > 20:
            idx = ib_dates.index(d) if d in ib_dates else -1
            if idx > 20:
                recent = ib_ranges[max(0, idx-50):idx]
                p25, p75 = np.percentile(recent, 25), np.percentile(recent, 75)
                if ib_r < p25 or ib_r > p75:
                    continue

        if highs[i] > ib_h:
            sigs[i] = 1
            traded[d] = True
        elif lows[i] < ib_l:
            sigs[i] = -1
            traded[d] = True

    return sigs


def sig_momentum_bar(df, atr_mult=1.0, vol_mult=1.0):
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    volumes = df["volume"].to_numpy().astype(float)
    atr = calc_atr(highs, lows, closes, 14)
    ema21 = calc_ema(closes, 21)
    n = len(closes)

    avg_vol = np.full(n, np.nan)
    for i in range(20, n):
        avg_vol[i] = np.mean(volumes[i-20:i])

    sigs = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(atr[i]) or np.isnan(ema21[i]) or np.isnan(avg_vol[i]):
            continue
        bar_range = highs[i] - lows[i]
        if bar_range < atr[i] * atr_mult:
            continue
        if volumes[i] < avg_vol[i] * vol_mult:
            continue
        bar_dir = 1 if closes[i] > opens[i] else -1
        trend_dir = 1 if closes[i] > ema21[i] else -1
        if bar_dir == trend_dir:
            sigs[i] = bar_dir

    return sigs


def extract_arrays(df):
    return (df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
            df["close"].to_numpy(), df["timestamp"].to_list(),
            df["hhmm"].to_numpy() if "hhmm" in df.columns else np.zeros(len(df), dtype=np.int32))


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

_ET = ZoneInfo("US/Eastern")


def load_and_resample(df_1m: pl.DataFrame) -> pl.DataFrame:
    """Take 1m data, filter RTH, resample to 15m. Returns 15m DataFrame."""
    if "ts_et" not in df_1m.columns:
        ts = df_1m["timestamp"]
        if hasattr(ts.dtype, 'time_zone') and ts.dtype.time_zone:
            df_1m = df_1m.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        df_1m = df_1m.with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC").alias("_utc")
        )
        df_1m = df_1m.with_columns(
            pl.col("_utc").dt.convert_time_zone("US/Eastern").alias("ts_et")
        )
        df_1m = df_1m.drop("_utc")

    df_1m = df_1m.with_columns([
        pl.col("ts_et").dt.date().alias("date_et"),
        pl.col("ts_et").dt.hour().cast(pl.Int32).alias("h_et"),
        pl.col("ts_et").dt.minute().cast(pl.Int32).alias("m_et"),
    ])
    df_1m = df_1m.with_columns([(pl.col("h_et") * 100 + pl.col("m_et")).alias("hhmm")])

    # RTH only: 9:30-16:00 ET
    rth = df_1m.filter((pl.col("hhmm") >= 930) & (pl.col("hhmm") < 1600))

    # Resample to 15m
    r = (
        rth.group_by_dynamic("ts_et", every="15m")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("date_et").last(),
            pl.col("hhmm").last(),
        ])
        .filter(pl.col("open").is_not_null())
        .sort("ts_et")
    )
    r = r.rename({"ts_et": "timestamp"})
    return r


def pull_databento_data(start: str, end: str) -> pl.DataFrame | None:
    """Pull MNQ 1m data from Databento."""
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("  [!] DATABENTO_API_KEY not set. Skipping data pull.")
        return None

    try:
        import databento as db
        client = db.Historical(key=api_key)
        print(f"  Pulling MNQ 1m from Databento: {start} to {end} ...")
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=["MNQ.c.0"],
            schema="ohlcv-1m",
            stype_in="continuous",
            start=start,
            end=end,
        )
        pdf = data.to_df()
        if len(pdf) == 0:
            print("  [!] No data returned from Databento.")
            return None

        df = pl.DataFrame({
            "timestamp": pdf.index.values,
            "open": pdf["open"].values,
            "high": pdf["high"].values,
            "low": pdf["low"].values,
            "close": pdf["close"].values,
            "volume": pdf["volume"].values,
        })
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None))
        df = df.with_columns(
            (pl.col("volume") / 5).cast(pl.Int64).clip(lower_bound=1).alias("tick_count")
        )
        print(f"  Pulled {len(df):,} bars.")
        return df
    except Exception as e:
        print(f"  [!] Databento pull failed: {e}")
        return None


def load_all_1m_data() -> pl.DataFrame:
    """Load all available MNQ 1m data from disk."""
    # Primary files in order (non-overlapping date ranges)
    files = [
        DATA_DIR / "databento_8yr_ext.parquet",   # 2017-12 to 2021-12
        DATA_DIR / "databento_extended.parquet",   # 2022-01 to 2024-03
        DATA_DIR / "full_2yr.parquet",             # 2024-03 to 2026-03
    ]
    dfs = []
    for f in files:
        if f.exists():
            dfs.append(pl.read_parquet(f))

    # Also load any fresh forward-test data
    ft_data = DATA_DIR / "forward_test_latest.parquet"
    if ft_data.exists():
        dfs.append(pl.read_parquet(ft_data))

    if not dfs:
        raise FileNotFoundError(f"No MNQ 1m data found in {DATA_DIR}")

    # Normalize dtypes before concat (Databento returns UInt64 volume, historical has Int64)
    normalized = []
    for d in dfs:
        if "volume" in d.columns:
            d = d.with_columns(pl.col("volume").cast(pl.Int64))
        if "tick_count" in d.columns:
            d = d.with_columns(pl.col("tick_count").cast(pl.Int64))
        normalized.append(d)

    df = pl.concat(normalized)
    df = df.sort("timestamp").unique(subset=["timestamp"], keep="last")
    return df


# ═══════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS (HTF Swing v3 — RSI + IB + MOM)
# ═══════════════════════════════════════════════════════════════════

STRATS = {
    "RSI": {"sl": 60, "tp": 400, "hold": 5},    # SL 15pts, TP 100pts
    "IB":  {"sl": 80, "tp": 480, "hold": 10},   # SL 20pts, TP 120pts
    "MOM": {"sl": 60, "tp": 400, "hold": 5},    # SL 15pts, TP 100pts
}

CONTRACTS = 3  # aggressive sizing


def run_strategy(df_15m, name, contracts):
    """Run a single sub-strategy on 15m data."""
    s = STRATS[name]
    if name == "RSI":
        sigs = sig_rsi_extreme(df_15m, 7, 70, 30)
    elif name == "IB":
        sigs = sig_ib_breakout(df_15m)
    elif name == "MOM":
        sigs = sig_momentum_bar(df_15m, 1.0, 1.0)
    else:
        raise ValueError(f"Unknown strategy: {name}")

    o, h, l, c, ts, hm = extract_arrays(df_15m)
    return backtest(o, h, l, c, ts, hm, sigs, s["sl"], s["tp"], s["hold"], contracts, name)


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════

def print_dashboard(all_trades: list[Trade], forward_start: str, account_size: float = 50000.0):
    """Print a nice dashboard showing forward test results."""
    # Filter to only forward-period trades
    ft_trades = [t for t in all_trades if str(t.entry_time)[:10] >= forward_start]
    all_recent = all_trades  # for context

    print()
    print("=" * 70)
    print("  HTF SWING v3 — FORWARD TEST DASHBOARD")
    print("=" * 70)
    print(f"  Forward test start: {forward_start}")
    print(f"  Account size: ${account_size:,.0f}")
    print(f"  Sizing: {CONTRACTS} contracts per sub-strategy")
    print(f"  Sub-strategies: RSI + IB Breakout + Momentum")
    print()

    if not ft_trades:
        print("  No trades yet in forward test period.")
        print("  (Strategy needs RTH market hours to generate signals)")
        print("=" * 70)
        return {}

    # Per-strategy breakdown
    by_strat = defaultdict(list)
    for t in ft_trades:
        by_strat[t.strategy].append(t)

    print(f"  {'Strategy':<12} {'Trades':>7} {'Win%':>7} {'Net P&L':>12} {'Avg':>10}")
    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*12} {'─'*10}")
    for name in ["RSI", "IB", "MOM"]:
        trades = by_strat.get(name, [])
        if trades:
            pnls = [t.net_pnl for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            print(f"  {name:<12} {len(trades):>7} {wins/len(trades)*100:>6.1f}% ${sum(pnls):>+10,.0f} ${np.mean(pnls):>+9,.0f}")
        else:
            print(f"  {name:<12} {'—':>7} {'—':>7} {'—':>12} {'—':>10}")

    # Totals
    total_pnl = sum(t.net_pnl for t in ft_trades)
    total_wins = sum(1 for t in ft_trades if t.net_pnl > 0)
    total_n = len(ft_trades)

    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*12} {'─'*10}")
    print(f"  {'TOTAL':<12} {total_n:>7} {total_wins/total_n*100:>6.1f}% ${total_pnl:>+10,.0f} ${total_pnl/total_n:>+9,.0f}")

    # Daily P&L
    daily = defaultdict(float)
    for t in ft_trades:
        daily[str(t.entry_time)[:10]] += t.net_pnl

    # Monthly P&L
    monthly = defaultdict(float)
    for t in ft_trades:
        monthly[str(t.entry_time)[:7]] += t.net_pnl

    # Equity curve
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    balance = account_size
    for d in sorted(daily.keys()):
        cum += daily[d]
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)

    balance = account_size + total_pnl

    print()
    print(f"  ACCOUNT:")
    print(f"    Starting:    ${account_size:>10,.0f}")
    print(f"    Current:     ${balance:>10,.0f}  ({'+' if total_pnl >= 0 else ''}{total_pnl/account_size*100:.1f}%)")
    print(f"    Net P&L:     ${total_pnl:>+10,.0f}")
    print(f"    Max Drawdown:${mdd:>+10,.0f}")

    # Daily breakdown
    print()
    print(f"  DAILY P&L:")
    for d in sorted(daily.keys()):
        v = daily[d]
        bar = "+" * min(int(abs(v) / 100), 30) if v > 0 else "-" * min(int(abs(v) / 100), 30)
        emoji = "G" if v > 0 else "R" if v < 0 else " "
        print(f"    {d}: ${v:>+8,.0f}  {bar}")

    # Monthly if applicable
    if len(monthly) > 1 or (len(monthly) == 1 and list(monthly.values())[0] != total_pnl):
        print()
        print(f"  MONTHLY P&L:")
        for m in sorted(monthly.keys()):
            v = monthly[m]
            print(f"    {m}: ${v:>+10,.0f}")

    # Prop firm check (Topstep 150K rules from v3)
    print()
    print(f"  PROP FIRM CHECK (Topstep 150K):")
    worst_day = min(daily.values()) if daily else 0
    print(f"    Worst day:   ${worst_day:>+10,.0f} vs -$3,000 {'PASS' if worst_day > -3000 else 'FAIL'}")
    print(f"    Max DD:      ${mdd:>+10,.0f} vs -$4,500 {'PASS' if mdd > -4500 else 'FAIL'}")

    trading_days = len(daily)
    if total_pnl >= 9000:
        print(f"    Eval target: ${total_pnl:>+10,.0f} vs $9,000  PASSED in {trading_days} days!")
    else:
        remaining = 9000 - total_pnl
        if trading_days > 0:
            avg_daily = total_pnl / trading_days
            if avg_daily > 0:
                est_days = int(remaining / avg_daily) + 1
                print(f"    Eval target: ${total_pnl:>+10,.0f} vs $9,000  (~{est_days} more trading days at current pace)")
            else:
                print(f"    Eval target: ${total_pnl:>+10,.0f} vs $9,000  (need ${remaining:,.0f} more)")
        else:
            print(f"    Eval target: ${total_pnl:>+10,.0f} vs $9,000  (need ${remaining:,.0f} more)")

    # Recent trades
    print()
    last_n = min(10, len(ft_trades))
    print(f"  LAST {last_n} TRADES:")
    print(f"  {'Time':<20} {'Strat':<6} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'P&L':>10} {'Reason':<12}")
    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
    for t in ft_trades[-last_n:]:
        d = "LONG" if t.direction == 1 else "SHORT"
        print(f"  {str(t.entry_time)[:19]:<20} {t.strategy:<6} {d:<6} {t.entry_px:>10.2f} {t.exit_px:>10.2f} ${t.net_pnl:>+9,.0f} {t.reason:<12}")

    print()
    print("=" * 70)

    # Return summary for state
    return {
        "total_trades": total_n,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(total_wins / total_n * 100, 1),
        "max_dd": round(mdd, 2),
        "worst_day": round(worst_day, 2),
        "balance": round(balance, 2),
        "trading_days": trading_days,
        "by_strategy": {
            name: {
                "trades": len(by_strat.get(name, [])),
                "pnl": round(sum(t.net_pnl for t in by_strat.get(name, [])), 2),
            }
            for name in ["RSI", "IB", "MOM"]
        },
        "daily": {d: round(v, 2) for d, v in sorted(daily.items())},
        "monthly": {m: round(v, 2) for m, v in sorted(monthly.items())},
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HTF Swing v3 Forward Test Tracker")
    parser.add_argument("--pull", action="store_true", help="Force fresh data pull from Databento")
    parser.add_argument("--dashboard", action="store_true", help="Just show dashboard from last run")
    parser.add_argument("--reset", action="store_true", help="Reset forward test start to today")
    parser.add_argument("--start-date", type=str, help="Set forward test start date (YYYY-MM-DD)")
    parser.add_argument("--contracts", type=int, default=3, help="Contracts per sub-strategy (default: 3)")
    parser.add_argument("--account", type=float, default=50000.0, help="Starting account size (default: 50000)")
    args = parser.parse_args()

    global CONTRACTS
    CONTRACTS = args.contracts

    state = load_state()

    # Set forward test start date
    if args.reset or args.start_date:
        state["forward_start"] = args.start_date or datetime.now(_ET).strftime("%Y-%m-%d")
        save_state(state)
        print(f"  Forward test start set to: {state['forward_start']}")

    if "forward_start" not in state:
        state["forward_start"] = datetime.now(_ET).strftime("%Y-%m-%d")
        save_state(state)
        print(f"  Forward test start set to: {state['forward_start']}")

    forward_start = state["forward_start"]

    # Dashboard-only mode
    if args.dashboard:
        if "last_summary" in state:
            print(json.dumps(state["last_summary"], indent=2))
        else:
            print("  No previous run found. Run without --dashboard first.")
        return

    # Pull fresh data if requested
    if args.pull:
        # Pull from the day before forward_start to ensure we have indicator warmup
        pull_start = (datetime.strptime(forward_start, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        pull_end = datetime.now(_ET).strftime("%Y-%m-%d")
        new_data = pull_databento_data(pull_start, pull_end)
        if new_data is not None:
            out_path = DATA_DIR / "forward_test_latest.parquet"
            new_data.write_parquet(out_path)
            print(f"  Saved to {out_path}")

    # Load all data
    print("  Loading MNQ 1m data ...")
    df_1m = load_all_1m_data()
    print(f"  Loaded {len(df_1m):,} bars: {df_1m['timestamp'][0]} to {df_1m['timestamp'][-1]}")

    # Resample to 15m
    print("  Resampling to 15m RTH bars ...")
    df_15m = load_and_resample(df_1m)
    print(f"  {len(df_15m):,} 15m bars")

    # Run all 3 sub-strategies
    print(f"\n  Running HTF Swing v3 ({CONTRACTS} contracts each) ...")
    t0 = _time.time()
    all_trades = []
    for name in ["RSI", "IB", "MOM"]:
        trades = run_strategy(df_15m, name, CONTRACTS)
        all_trades.extend(trades)
        ft = [t for t in trades if str(t.entry_time)[:10] >= forward_start]
        print(f"    {name}: {len(trades)} total trades, {len(ft)} in forward period")

    all_trades.sort(key=lambda t: str(t.entry_time))
    elapsed = _time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Dashboard
    summary = print_dashboard(all_trades, forward_start, args.account)

    # Save state
    state["last_run"] = datetime.now().isoformat()
    state["last_summary"] = summary
    state["data_end"] = str(df_1m["timestamp"][-1])
    save_state(state)

    # Save detailed trades to CSV
    ft_trades = [t for t in all_trades if str(t.entry_time)[:10] >= forward_start]
    if ft_trades:
        trades_file = REPORTS_DIR / "forward_test_trades.csv"
        rows = []
        for t in ft_trades:
            rows.append({
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "strategy": t.strategy,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_px": t.entry_px,
                "exit_px": t.exit_px,
                "contracts": t.contracts,
                "net_pnl": round(t.net_pnl, 2),
                "bars_held": t.bars_held,
                "reason": t.reason,
            })
        pl.DataFrame(rows).write_csv(trades_file)
        print(f"\n  Trades saved to {trades_file}")

    # Save summary report
    report_file = REPORTS_DIR / "forward_test_report.json"
    with open(report_file, "w") as f:
        json.dump({
            "forward_start": forward_start,
            "last_run": state["last_run"],
            "data_range": f"{df_1m['timestamp'][0]} to {df_1m['timestamp'][-1]}",
            "summary": summary,
        }, f, indent=2, default=str)
    print(f"  Report saved to {report_file}")


if __name__ == "__main__":
    main()
