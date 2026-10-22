#!/usr/bin/env python3
"""
HTF Swing v3 — Live Paper Trader

Runs as a persistent background process. Connects to Databento live stream,
receives 1m MNQ bars, builds 15m bars, runs all 3 sub-strategies
(RSI + IB Breakout + Momentum) with full position management and prop firm
risk rules.

Usage:
    python3 paper_trader_live.py                  # Run live (default Topstep 150K)
    python3 paper_trader_live.py --account 50000  # Custom account size
    python3 paper_trader_live.py --status          # Show current status from state file
    python3 paper_trader_live.py --conservative    # 2 contracts instead of 3

Runs 24/5. Sleeps outside market hours. Start it and forget it.
"""

import json
import logging
import os
import signal
import sys
import time as _time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta, time
from pathlib import Path
from threading import Event
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent
STATE_FILE = ROOT / "paper_trader_state.json"
TRADE_LOG = ROOT / "reports" / "paper_trader" / "trades.csv"
DAILY_LOG = ROOT / "reports" / "paper_trader" / "daily.csv"
BAR_LOG = ROOT / "reports" / "paper_trader" / "bars_15m.csv"
LOG_FILE = ROOT / "reports" / "paper_trader" / "paper_trader.log"

for p in [TRADE_LOG.parent]:
    p.mkdir(parents=True, exist_ok=True)

ET = ZoneInfo("US/Eastern")

# MNQ cost model
TICK_SIZE = 0.25
POINT_VALUE = 2.0
SLIP_TICKS = 2
COMM_PER_SIDE = 0.62
EXCH_PER_SIDE = 0.27
SLIP_PTS = SLIP_TICKS * TICK_SIZE  # 0.50

def rt_cost(c: int) -> float:
    return (COMM_PER_SIDE + EXCH_PER_SIDE) * 2 * c + SLIP_TICKS * TICK_SIZE * POINT_VALUE * 2 * c

# RTH session
RTH_START = time(9, 30)
RTH_END = time(16, 0)
FLATTEN_TIME = time(15, 45)
NO_ENTRY_AFTER = time(15, 30)

# Prop firm rules (Topstep 150K)
DAILY_LOSS_LIMIT = -3000.0
MAX_TRAILING_DD = -4500.0
EVAL_TARGET = 9000.0

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger("paper_trader")

# ═══════════════════════════════════════════════════════════════════
# INDICATORS (exact copies from Berj's run_htf_swing.py)
# ═══════════════════════════════════════════════════════════════════

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


def calc_vwap_daily(highs, lows, closes, volumes, dates):
    """Daily VWAP (resets each day)."""
    tp = (highs + lows + closes) / 3
    n = len(highs)
    vwap = np.full(n, np.nan)
    cum_tpv = 0.0
    cum_vol = 0.0
    prev_date = None
    for i in range(n):
        if dates[i] != prev_date:
            cum_tpv = 0.0
            cum_vol = 0.0
            prev_date = dates[i]
        cum_tpv += tp[i] * volumes[i]
        cum_vol += volumes[i]
        vwap[i] = cum_tpv / cum_vol if cum_vol > 0 else tp[i]
    return vwap


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Position:
    strategy: str
    direction: int        # +1 long, -1 short
    entry_px: float
    entry_time: str
    entry_bar_idx: int
    contracts: int
    stop_px: float
    target_px: float
    max_hold: int

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Position(**d)


@dataclass
class TradeRecord:
    strategy: str
    direction: int
    entry_px: float
    exit_px: float
    entry_time: str
    exit_time: str
    contracts: int
    net_pnl: float
    bars_held: int
    reason: str

    def to_csv_row(self):
        d = "LONG" if self.direction == 1 else "SHORT"
        return f"{self.entry_time},{self.exit_time},{self.strategy},{d},{self.entry_px:.2f},{self.exit_px:.2f},{self.contracts},{self.net_pnl:.2f},{self.bars_held},{self.reason}"


# ═══════════════════════════════════════════════════════════════════
# PAPER TRADING ENGINE
# ═══════════════════════════════════════════════════════════════════

class PaperTrader:
    """Full paper trading engine with position management and risk rules."""

    def __init__(self, account_size: float = 150000.0, contracts: int = 3):
        self.initial_balance = account_size
        self.contracts = contracts

        # State
        self.balance = account_size
        self.peak_balance = account_size
        self.daily_pnl = 0.0
        self.current_date = None
        self.positions: dict[str, Position] = {}  # strategy_name -> Position
        self.pending_signals: dict[str, int] = {}  # strategy_name -> direction
        self.trades: list[TradeRecord] = []
        self.daily_summary: dict[str, dict] = {}
        self.bar_count = 0
        self.killed = False  # risk kill switch

        # IB tracking
        self.ib_data: dict = {}           # date -> {high, low}
        self.ib_ranges: list[float] = []  # rolling IB ranges for percentile
        self.ib_traded_today = False
        self.ib_complete_today = False

        # 15m bar history for indicators
        self.bar_history: list[dict] = []
        self.max_history = 500  # keep last 500 bars for indicator warmup

        # Strategy params (exact v3 settings)
        self.strat_params = {
            "RSI": {"sl_ticks": 60, "tp_ticks": 400, "max_hold": 5},
            "IB":  {"sl_ticks": 80, "tp_ticks": 480, "max_hold": 10},
            "MOM": {"sl_ticks": 60, "tp_ticks": 400, "max_hold": 5},
        }

    # ── State persistence ──────────────────────────────────────────

    def save_state(self):
        state = {
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "initial_balance": self.initial_balance,
            "contracts": self.contracts,
            "daily_pnl": self.daily_pnl,
            "current_date": self.current_date,
            "bar_count": self.bar_count,
            "killed": self.killed,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "pending_signals": self.pending_signals,
            "ib_data": {str(k): v for k, v in self.ib_data.items()},
            "ib_ranges": self.ib_ranges[-50:],
            "ib_traded_today": self.ib_traded_today,
            "ib_complete_today": self.ib_complete_today,
            "trades_count": len(self.trades),
            "daily_summary": self.daily_summary,
            "bar_history": self.bar_history[-self.max_history:],
            "last_save": datetime.now(ET).isoformat(),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self) -> bool:
        if not STATE_FILE.exists():
            return False
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.balance = state["balance"]
            self.peak_balance = state["peak_balance"]
            self.initial_balance = state.get("initial_balance", self.initial_balance)
            self.contracts = state.get("contracts", self.contracts)
            self.daily_pnl = state["daily_pnl"]
            self.current_date = state["current_date"]
            self.bar_count = state["bar_count"]
            self.killed = state.get("killed", False)
            self.positions = {k: Position.from_dict(v) for k, v in state.get("positions", {}).items()}
            self.pending_signals = state.get("pending_signals", {})
            self.ib_data = state.get("ib_data", {})
            self.ib_ranges = state.get("ib_ranges", [])
            self.ib_traded_today = state.get("ib_traded_today", False)
            self.ib_complete_today = state.get("ib_complete_today", False)
            self.daily_summary = state.get("daily_summary", {})
            self.bar_history = state.get("bar_history", [])
            log.info(f"Restored state: balance=${self.balance:,.0f}, {len(self.positions)} open positions, {self.bar_count} bars processed")
            return True
        except Exception as e:
            log.error(f"Failed to load state: {e}")
            return False

    # ── Risk checks ────────────────────────────────────────────────

    def check_risk(self) -> bool:
        """Check prop firm risk limits. Returns True if trading allowed."""
        if self.killed:
            return False

        # Daily loss limit
        if self.daily_pnl <= DAILY_LOSS_LIMIT:
            log.warning(f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:,.0f} <= ${DAILY_LOSS_LIMIT:,.0f}")
            self.killed = True
            return False

        # Trailing max drawdown
        dd = self.balance - self.peak_balance
        if dd <= MAX_TRAILING_DD:
            log.warning(f"MAX TRAILING DD HIT: ${dd:,.0f} <= ${MAX_TRAILING_DD:,.0f}")
            self.killed = True
            return False

        # Kill switch at 80% of daily limit
        if self.daily_pnl <= DAILY_LOSS_LIMIT * 0.8:
            log.warning(f"Kill switch: daily P&L ${self.daily_pnl:,.0f} hit 80% of limit")
            return False

        return True

    # ── New day reset ──────────────────────────────────────────────

    def new_day(self, date_str: str):
        """Reset daily state for a new trading day."""
        if self.current_date and self.current_date != date_str:
            # Save previous day summary
            self.daily_summary[self.current_date] = {
                "pnl": round(self.daily_pnl, 2),
                "balance": round(self.balance, 2),
                "trades": sum(1 for t in self.trades if t.entry_time[:10] == self.current_date),
            }
            log.info(f"Day {self.current_date} closed: P&L ${self.daily_pnl:+,.0f}, balance ${self.balance:,.0f}")

        self.current_date = date_str
        self.daily_pnl = 0.0
        self.ib_traded_today = False
        self.ib_complete_today = False
        self.killed = False  # reset daily kill
        # Reset IB data for today
        if date_str not in self.ib_data:
            self.ib_data[date_str] = None
        log.info(f"=== New trading day: {date_str} | Balance: ${self.balance:,.0f} | DD from peak: ${self.balance - self.peak_balance:+,.0f} ===")

    # ── Position management ────────────────────────────────────────

    def open_position(self, strategy: str, direction: int, fill_px: float, bar_time: str, bar_idx: int):
        """Open a new position."""
        params = self.strat_params[strategy]
        entry_px = fill_px + direction * SLIP_PTS  # slippage on entry

        pos = Position(
            strategy=strategy,
            direction=direction,
            entry_px=entry_px,
            entry_time=bar_time,
            entry_bar_idx=bar_idx,
            contracts=self.contracts,
            stop_px=entry_px - direction * params["sl_ticks"] * TICK_SIZE,
            target_px=entry_px + direction * params["tp_ticks"] * TICK_SIZE,
            max_hold=params["max_hold"],
        )
        self.positions[strategy] = pos
        d = "LONG" if direction == 1 else "SHORT"
        log.info(f"  OPEN {strategy} {d} @ {entry_px:.2f} | SL {pos.stop_px:.2f} TP {pos.target_px:.2f}")

    def close_position(self, strategy: str, exit_px: float, bar_time: str, bars_held: int, reason: str):
        """Close a position and record the trade."""
        pos = self.positions.pop(strategy)
        exit_px_adj = exit_px - pos.direction * SLIP_PTS  # slippage on exit
        cost = rt_cost(pos.contracts)
        raw_pnl = (exit_px_adj - pos.entry_px) * pos.direction * POINT_VALUE * pos.contracts
        net_pnl = raw_pnl - cost

        trade = TradeRecord(
            strategy=strategy,
            direction=pos.direction,
            entry_px=pos.entry_px,
            exit_px=exit_px_adj,
            entry_time=pos.entry_time,
            exit_time=bar_time,
            contracts=pos.contracts,
            net_pnl=round(net_pnl, 2),
            bars_held=bars_held,
            reason=reason,
        )
        self.trades.append(trade)
        self.balance += net_pnl
        self.daily_pnl += net_pnl
        self.peak_balance = max(self.peak_balance, self.balance)

        # Append to trade log CSV
        if not TRADE_LOG.exists():
            with open(TRADE_LOG, "w") as f:
                f.write("entry_time,exit_time,strategy,direction,entry_px,exit_px,contracts,net_pnl,bars_held,reason\n")
        with open(TRADE_LOG, "a") as f:
            f.write(trade.to_csv_row() + "\n")

        d = "LONG" if pos.direction == 1 else "SHORT"
        emoji = "W" if net_pnl > 0 else "L"
        log.info(f"  CLOSE {strategy} {d} @ {exit_px_adj:.2f} | P&L ${net_pnl:+,.0f} ({reason}) [{emoji}]")

    # ── Signal generation ──────────────────────────────────────────

    def compute_signals(self, bars: list[dict]) -> dict[str, int]:
        """Compute signals for all 3 strategies from bar history."""
        if len(bars) < 25:
            return {}

        closes = np.array([b["close"] for b in bars])
        opens = np.array([b["open"] for b in bars])
        highs = np.array([b["high"] for b in bars])
        lows = np.array([b["low"] for b in bars])
        volumes = np.array([b["volume"] for b in bars], dtype=float)
        hhmm = np.array([b["hhmm"] for b in bars])
        dates = [b["date_et"] for b in bars]

        signals = {}
        i = len(bars) - 1  # current bar index

        # ── RSI Extremes ──
        rsi = calc_rsi(closes, 7)
        if not np.isnan(rsi[i]):
            if rsi[i] < 30:
                signals["RSI"] = 1
            elif rsi[i] > 70:
                signals["RSI"] = -1

        # ── IB Breakout ──
        current_hhmm = hhmm[i]
        current_date = dates[i]

        # Build IB for today (bars before 10:00)
        if current_hhmm < 1000:
            if self.ib_data.get(current_date) is None:
                self.ib_data[current_date] = {"high": highs[i], "low": lows[i]}
            else:
                self.ib_data[current_date]["high"] = max(self.ib_data[current_date]["high"], float(highs[i]))
                self.ib_data[current_date]["low"] = min(self.ib_data[current_date]["low"], float(lows[i]))

        # Mark IB complete
        if current_hhmm >= 1000 and not self.ib_complete_today and self.ib_data.get(current_date):
            self.ib_complete_today = True
            ib = self.ib_data[current_date]
            ib_range = ib["high"] - ib["low"]
            if ib_range > 0:
                self.ib_ranges.append(ib_range)
                if len(self.ib_ranges) > 50:
                    self.ib_ranges = self.ib_ranges[-50:]

        # IB signal
        if (self.ib_complete_today and not self.ib_traded_today
                and 1000 <= current_hhmm < 1530 and self.ib_data.get(current_date)):
            ib = self.ib_data[current_date]
            ib_h = ib["high"]
            ib_l = ib["low"]
            ib_r = ib_h - ib_l

            if ib_r >= 2.0:
                range_ok = True
                if len(self.ib_ranges) > 20:
                    p25 = np.percentile(self.ib_ranges[:-1], 25)  # exclude current
                    p75 = np.percentile(self.ib_ranges[:-1], 75)
                    range_ok = p25 <= ib_r <= p75

                if range_ok:
                    if highs[i] > ib_h:
                        signals["IB"] = 1
                    elif lows[i] < ib_l:
                        signals["IB"] = -1

        # ── Momentum Bar ──
        atr = calc_atr(highs, lows, closes, 14)
        ema21 = calc_ema(closes, 21)

        if not np.isnan(atr[i]) and not np.isnan(ema21[i]):
            avg_vol = np.mean(volumes[max(0, i-20):i]) if i >= 20 else np.nan
            if not np.isnan(avg_vol):
                bar_range = highs[i] - lows[i]
                if bar_range >= atr[i] * 1.0 and volumes[i] >= avg_vol * 1.0:
                    bar_dir = 1 if closes[i] > opens[i] else -1
                    trend_dir = 1 if closes[i] > ema21[i] else -1
                    if bar_dir == trend_dir:
                        signals["MOM"] = bar_dir

        return signals

    # ── Process a new 15m bar ──────────────────────────────────────

    def process_bar(self, bar: dict):
        """Process a new 15-minute bar. This is the main entry point."""
        bar_time = bar["timestamp"]
        bar_date = bar["date_et"]
        bar_hhmm = bar["hhmm"]

        # New day?
        if self.current_date != bar_date:
            self.new_day(bar_date)

        # Add to history
        self.bar_history.append(bar)
        if len(self.bar_history) > self.max_history:
            self.bar_history = self.bar_history[-self.max_history:]
        self.bar_count += 1

        bar_open = bar["open"]
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]

        # ── 1. EOD flatten ──
        if bar_hhmm >= 1545:
            for strat in list(self.positions.keys()):
                pos = self.positions[strat]
                bars_held = self.bar_count - pos.entry_bar_idx
                self.close_position(strat, bar_close, bar_time, bars_held, "eod_flatten")
            self.pending_signals.clear()
            self._status_line(bar_time, bar_close)
            self.save_state()
            return

        # ── 2. Execute pending signals (fill at this bar's open) ──
        for strat, direction in list(self.pending_signals.items()):
            if strat in self.positions:
                continue  # already in position
            if bar_hhmm >= 1530:
                continue  # too close to EOD

            if not self.check_risk():
                log.warning(f"  Risk check failed, skipping {strat} entry")
                continue

            self.open_position(strat, direction, bar_open, bar_time, self.bar_count)
            if strat == "IB":
                self.ib_traded_today = True

        self.pending_signals.clear()

        # ── 3. Manage open positions ──
        for strat in list(self.positions.keys()):
            pos = self.positions[strat]
            bars_held = self.bar_count - pos.entry_bar_idx

            if bars_held == 0:
                continue  # skip entry bar

            exit_px = None
            reason = ""

            # Stop loss FIRST (conservative)
            if pos.direction == 1 and bar_low <= pos.stop_px:
                exit_px = pos.stop_px
                reason = "stop_loss"
            elif pos.direction == -1 and bar_high >= pos.stop_px:
                exit_px = pos.stop_px
                reason = "stop_loss"

            # Take profit (only if stop not hit)
            if exit_px is None:
                if pos.direction == 1 and bar_high >= pos.target_px:
                    exit_px = pos.target_px
                    reason = "take_profit"
                elif pos.direction == -1 and bar_low <= pos.target_px:
                    exit_px = pos.target_px
                    reason = "take_profit"

            # Max hold
            if exit_px is None and bars_held >= pos.max_hold:
                exit_px = bar_close
                reason = "max_hold"

            if exit_px is not None:
                self.close_position(strat, exit_px, bar_time, bars_held, reason)

        # ── 4. Generate new signals ──
        if bar_hhmm < 1530 and 930 <= bar_hhmm:
            signals = self.compute_signals(self.bar_history)
            for strat, direction in signals.items():
                if strat not in self.positions:
                    self.pending_signals[strat] = direction

        # ── 5. Status update ──
        self._status_line(bar_time, bar_close)
        self.save_state()

    def _status_line(self, bar_time: str, price: float):
        """Print one-line status."""
        pos_str = ", ".join(
            f"{s}:{'L' if p.direction == 1 else 'S'}"
            for s, p in self.positions.items()
        ) or "flat"
        pend_str = ", ".join(
            f"{s}:{'L' if d == 1 else 'S'}"
            for s, d in self.pending_signals.items()
        )
        pend = f" | pend: {pend_str}" if pend_str else ""
        dd = self.balance - self.peak_balance
        log.info(
            f"  BAR {bar_time} | MNQ {price:.2f} | "
            f"bal ${self.balance:,.0f} | day ${self.daily_pnl:+,.0f} | "
            f"dd ${dd:+,.0f} | pos: {pos_str}{pend}"
        )

    # ── Dashboard ──────────────────────────────────────────────────

    def print_dashboard(self):
        """Print full status dashboard."""
        print()
        print("=" * 70)
        print("  HTF SWING v3 — LIVE PAPER TRADER")
        print("=" * 70)
        print(f"  Status: {'KILLED - risk limit breached' if self.killed else 'ACTIVE'}")
        print(f"  Sizing: {self.contracts} contracts per sub-strategy")
        print(f"  Bars processed: {self.bar_count}")
        print()

        # Account
        total_pnl = self.balance - self.initial_balance
        dd = self.balance - self.peak_balance
        print(f"  ACCOUNT:")
        print(f"    Starting:     ${self.initial_balance:>10,.0f}")
        print(f"    Current:      ${self.balance:>10,.0f}  ({total_pnl/self.initial_balance*100:+.1f}%)")
        print(f"    Net P&L:      ${total_pnl:>+10,.0f}")
        print(f"    Peak:         ${self.peak_balance:>10,.0f}")
        print(f"    Drawdown:     ${dd:>+10,.0f}")
        print(f"    Today P&L:    ${self.daily_pnl:>+10,.0f}")

        # Open positions
        if self.positions:
            print()
            print(f"  OPEN POSITIONS:")
            for strat, pos in self.positions.items():
                d = "LONG" if pos.direction == 1 else "SHORT"
                print(f"    {strat}: {d} @ {pos.entry_px:.2f} | SL {pos.stop_px:.2f} TP {pos.target_px:.2f}")

        # Trades
        if self.trades:
            wins = sum(1 for t in self.trades if t.net_pnl > 0)
            total_n = len(self.trades)
            print()
            print(f"  TRADES: {total_n} total, {wins/total_n*100:.1f}% win rate")

            by_strat = defaultdict(list)
            for t in self.trades:
                by_strat[t.strategy].append(t)

            print(f"  {'Strategy':<10} {'N':>5} {'Win%':>7} {'P&L':>12}")
            print(f"  {'─'*10} {'─'*5} {'─'*7} {'─'*12}")
            for name in ["RSI", "IB", "MOM"]:
                trades = by_strat.get(name, [])
                if trades:
                    w = sum(1 for t in trades if t.net_pnl > 0)
                    pnl = sum(t.net_pnl for t in trades)
                    print(f"  {name:<10} {len(trades):>5} {w/len(trades)*100:>6.1f}% ${pnl:>+10,.0f}")

        # Daily summary
        if self.daily_summary:
            print()
            print(f"  DAILY HISTORY:")
            for d in sorted(self.daily_summary.keys())[-14:]:  # last 14 days
                s = self.daily_summary[d]
                pnl = s["pnl"]
                bar = "+" * min(int(abs(pnl) / 100), 30) if pnl > 0 else "-" * min(int(abs(pnl) / 100), 30)
                print(f"    {d}: ${pnl:>+8,.0f}  {bar}")

        # Prop firm
        print()
        worst_day = min((s["pnl"] for s in self.daily_summary.values()), default=0)
        print(f"  PROP FIRM (Topstep 150K):")
        print(f"    Worst day:    ${worst_day:>+10,.0f} vs -$3,000 {'PASS' if worst_day > -3000 else 'FAIL'}")
        print(f"    Max DD:       ${dd:>+10,.0f} vs -$4,500 {'PASS' if dd > -4500 else 'FAIL'}")
        if total_pnl >= EVAL_TARGET:
            print(f"    Eval target:  PASSED! ${total_pnl:>+10,.0f} >= $9,000")
        else:
            print(f"    Eval target:  ${total_pnl:>+10,.0f} / $9,000 ({total_pnl/EVAL_TARGET*100:.0f}%)")

        print()
        print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# 1m → 15m BAR BUILDER
# ═══════════════════════════════════════════════════════════════════

class BarBuilder:
    """Accumulates 1m bars into 15m bars."""

    def __init__(self):
        self.current_bucket = None  # datetime of current 15m bucket start
        self.bars_1m: list[dict] = []

    def add_1m_bar(self, bar: dict) -> dict | None:
        """Add a 1m bar. Returns completed 15m bar when bucket is full, else None."""
        ts = bar["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        # Convert to ET
        if ts.tzinfo is None:
            ts_et = ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(ET)
        else:
            ts_et = ts.astimezone(ET)

        # RTH filter: 9:30-16:00 ET
        t = ts_et.time()
        if t < RTH_START or t >= RTH_END:
            return None

        hhmm = ts_et.hour * 100 + ts_et.minute
        date_et = ts_et.strftime("%Y-%m-%d")

        # 15m bucket: floor to 15-min boundary
        minute_bucket = (ts_et.minute // 15) * 15
        bucket = ts_et.replace(minute=minute_bucket, second=0, microsecond=0)

        bar_data = {
            "timestamp_1m": ts_et.isoformat(),
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": int(bar.get("volume", 0)),
            "hhmm": hhmm,
            "date_et": date_et,
        }

        # New bucket?
        if self.current_bucket is not None and bucket != self.current_bucket:
            # Emit the completed 15m bar
            completed = self._build_15m()
            self.current_bucket = bucket
            self.bars_1m = [bar_data]
            return completed

        self.current_bucket = bucket
        self.bars_1m.append(bar_data)
        return None

    def _build_15m(self) -> dict | None:
        if not self.bars_1m:
            return None

        bar = {
            "timestamp": self.bars_1m[0]["timestamp_1m"],
            "open": self.bars_1m[0]["open"],
            "high": max(b["high"] for b in self.bars_1m),
            "low": min(b["low"] for b in self.bars_1m),
            "close": self.bars_1m[-1]["close"],
            "volume": sum(b["volume"] for b in self.bars_1m),
            "hhmm": self.bars_1m[-1]["hhmm"],
            "date_et": self.bars_1m[-1]["date_et"],
        }
        return bar

    def flush(self) -> dict | None:
        """Force emit current bucket (e.g., at EOD)."""
        if self.bars_1m:
            bar = self._build_15m()
            self.bars_1m = []
            self.current_bucket = None
            return bar
        return None


# ═══════════════════════════════════════════════════════════════════
# DATABENTO LIVE STREAM
# ═══════════════════════════════════════════════════════════════════

def run_live(trader: PaperTrader):
    """Main live loop using Databento streaming."""
    import databento as db

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        log.error("DATABENTO_API_KEY not set. Cannot start live stream.")
        sys.exit(1)

    builder = BarBuilder()
    shutdown = Event()

    def handle_signal(signum, frame):
        log.info("Shutdown signal received, saving state...")
        trader.save_state()
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not shutdown.is_set():
        now_et = datetime.now(ET)

        # Check if market day (Mon-Fri)
        if now_et.weekday() >= 5:
            next_monday = now_et + timedelta(days=(7 - now_et.weekday()))
            next_open = next_monday.replace(hour=9, minute=15, second=0, microsecond=0)
            wait = (next_open - now_et).total_seconds()
            log.info(f"Weekend. Sleeping until {next_open.strftime('%Y-%m-%d %H:%M ET')} ({wait/3600:.1f}h)")
            shutdown.wait(min(wait, 3600))  # check hourly
            continue

        # Before market open?
        market_open = now_et.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=15, second=0, microsecond=0)

        if now_et < market_open:
            wait = (market_open - now_et).total_seconds()
            log.info(f"Pre-market. Sleeping until {market_open.strftime('%H:%M ET')} ({wait/60:.0f}m)")
            shutdown.wait(min(wait, 600))  # check every 10 min
            continue

        if now_et > market_close:
            # Flush any remaining bars
            final_bar = builder.flush()
            if final_bar:
                trader.process_bar(final_bar)

            tomorrow = now_et + timedelta(days=1)
            next_open = tomorrow.replace(hour=9, minute=15, second=0, microsecond=0)
            # Skip to Monday if Friday
            if tomorrow.weekday() >= 5:
                days_until_monday = 7 - tomorrow.weekday()
                next_open = (tomorrow + timedelta(days=days_until_monday)).replace(hour=9, minute=15, second=0, microsecond=0)

            wait = (next_open - now_et).total_seconds()
            log.info(f"Market closed. Sleeping until {next_open.strftime('%Y-%m-%d %H:%M ET')} ({wait/3600:.1f}h)")
            trader.save_state()
            shutdown.wait(min(wait, 3600))
            continue

        # ── Market is open — connect to Databento live ──
        log.info("Connecting to Databento live stream...")
        try:
            client = db.Live(key=api_key, reconnect_policy="reconnect")
            client.subscribe(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
                symbols=["MNQ.c.0"],
                stype_in="continuous",
            )
            log.info("Subscribed to MNQ.c.0 ohlcv-1m. Waiting for bars...")

            for record in client:
                if shutdown.is_set():
                    break

                # Check if market is still open
                now_et = datetime.now(ET)
                if now_et.time() > time(16, 15):
                    log.info("Market closed, disconnecting stream.")
                    break

                # Parse the OHLCV record
                try:
                    bar_1m = {
                        "timestamp": record.ts_event if hasattr(record, "ts_event") else record.hd.ts_event,
                        "open": float(record.open) / 1e9 if record.open > 1e6 else float(record.open),
                        "high": float(record.high) / 1e9 if record.high > 1e6 else float(record.high),
                        "low": float(record.low) / 1e9 if record.low > 1e6 else float(record.low),
                        "close": float(record.close) / 1e9 if record.close > 1e6 else float(record.close),
                        "volume": int(record.volume),
                    }
                except AttributeError:
                    continue  # heartbeat or other non-OHLCV record

                # Build 15m bar
                bar_15m = builder.add_1m_bar(bar_1m)
                if bar_15m is not None:
                    trader.process_bar(bar_15m)

        except Exception as e:
            log.error(f"Stream error: {e}")
            log.error(traceback.format_exc())
            log.info("Reconnecting in 30 seconds...")
            shutdown.wait(30)
            continue

    log.info("Paper trader stopped.")
    trader.save_state()


# ═══════════════════════════════════════════════════════════════════
# POLLING FALLBACK (if live stream isn't available)
# ═══════════════════════════════════════════════════════════════════

def run_polling(trader: PaperTrader):
    """Fallback: poll Databento Historical API every 15 min."""
    import databento as db

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        log.error("DATABENTO_API_KEY not set.")
        sys.exit(1)

    shutdown = Event()

    def handle_signal(signum, frame):
        log.info("Shutdown signal received, saving state...")
        trader.save_state()
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    client = db.Historical(key=api_key)
    last_bar_time = None

    while not shutdown.is_set():
        now_et = datetime.now(ET)

        # Weekend check
        if now_et.weekday() >= 5:
            next_monday = now_et + timedelta(days=(7 - now_et.weekday()))
            next_open = next_monday.replace(hour=9, minute=15, second=0)
            wait = (next_open - now_et).total_seconds()
            log.info(f"Weekend. Sleeping until Monday {next_open.strftime('%H:%M ET')}")
            shutdown.wait(min(wait, 3600))
            continue

        # Pre-market
        if now_et.time() < time(9, 15):
            market_open = now_et.replace(hour=9, minute=15, second=0)
            wait = (market_open - now_et).total_seconds()
            log.info(f"Pre-market. Sleeping {wait/60:.0f}m")
            shutdown.wait(min(wait, 600))
            continue

        # Post-market
        if now_et.time() > time(16, 30):
            trader.save_state()
            tomorrow = now_et + timedelta(days=1)
            if tomorrow.weekday() >= 5:
                tomorrow += timedelta(days=(7 - tomorrow.weekday()))
            next_open = tomorrow.replace(hour=9, minute=15, second=0)
            wait = (next_open - now_et).total_seconds()
            log.info(f"Market closed. Next open: {next_open.strftime('%Y-%m-%d %H:%M ET')}")
            shutdown.wait(min(wait, 3600))
            continue

        # Pull last 30 min of 1m data
        try:
            start = (now_et - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S")
            end = now_et.strftime("%Y-%m-%dT%H:%M:%S")

            data = client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["MNQ.c.0"],
                schema="ohlcv-1m",
                stype_in="continuous",
                start=start,
                end=end,
            )
            pdf = data.to_df()

            if len(pdf) > 0:
                builder = BarBuilder()
                # Restore builder state from previous bars if needed
                for _, row in pdf.iterrows():
                    bar_1m = {
                        "timestamp": row.name,  # index is timestamp
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                    }
                    bar_15m = builder.add_1m_bar(bar_1m)
                    if bar_15m is not None:
                        bar_key = bar_15m["timestamp"]
                        if bar_key != last_bar_time:
                            trader.process_bar(bar_15m)
                            last_bar_time = bar_key

        except Exception as e:
            log.error(f"Poll error: {e}")

        # Sleep until next 15-min mark + 1 min buffer
        next_minute = ((now_et.minute // 15) + 1) * 15
        if next_minute >= 60:
            next_bar = (now_et + timedelta(hours=1)).replace(minute=next_minute - 60, second=30, microsecond=0)
        else:
            next_bar = now_et.replace(minute=next_minute, second=30, microsecond=0)

        wait = max((next_bar - datetime.now(ET)).total_seconds(), 10)
        log.info(f"Next poll at {next_bar.strftime('%H:%M:%S ET')} ({wait/60:.1f}m)")
        shutdown.wait(wait)

    trader.save_state()
    log.info("Paper trader stopped.")


# ═══════════════════════════════════════════════════════════════════
# WARMUP: Load historical bars to seed indicators
# ═══════════════════════════════════════════════════════════════════

def warmup_from_history(trader: PaperTrader):
    """Load recent 15m bars from disk to warm up indicators."""
    data_dir = ROOT / "data" / "processed" / "MNQ" / "1m"
    latest = data_dir / "full_2yr.parquet"
    if not latest.exists():
        log.warning("No historical data for warmup. Indicators will need ~50 bars to stabilize.")
        return

    log.info("Warming up indicators from historical data...")
    df = pl.read_parquet(latest)

    # Only need last ~100 bars worth of 1m data (100 * 15 = 1500 minutes = ~6 trading days)
    df = df.tail(10000)

    # Resample to 15m
    if "ts_et" not in df.columns:
        if hasattr(df["timestamp"].dtype, 'time_zone') and df["timestamp"].dtype.time_zone:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        df = df.with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC").alias("_utc")
        )
        df = df.with_columns(
            pl.col("_utc").dt.convert_time_zone("US/Eastern").alias("ts_et")
        )
        df = df.drop("_utc")

    df = df.with_columns([
        pl.col("ts_et").dt.date().alias("date_et"),
        pl.col("ts_et").dt.hour().cast(pl.Int32).alias("h_et"),
        pl.col("ts_et").dt.minute().cast(pl.Int32).alias("m_et"),
    ])
    df = df.with_columns([(pl.col("h_et") * 100 + pl.col("m_et")).alias("hhmm")])
    rth = df.filter((pl.col("hhmm") >= 930) & (pl.col("hhmm") < 1600))

    bars_15m = (
        rth.group_by_dynamic("ts_et", every="15m")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("date_et").last().cast(pl.Utf8),
            pl.col("hhmm").last(),
        ])
        .filter(pl.col("open").is_not_null())
        .sort("ts_et")
    )

    # Only take last 200 bars for warmup (don't replay as trades)
    warmup_bars = bars_15m.tail(200)
    for row in warmup_bars.iter_rows(named=True):
        bar = {
            "timestamp": str(row["ts_et"]),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "date_et": str(row["date_et"]),
            "hhmm": row["hhmm"],
        }
        trader.bar_history.append(bar)

    trader.bar_history = trader.bar_history[-trader.max_history:]
    log.info(f"Warmed up with {len(warmup_bars)} historical 15m bars")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HTF Swing v3 Live Paper Trader")
    parser.add_argument("--account", type=float, default=150000.0, help="Account size (default: 150000)")
    parser.add_argument("--contracts", type=int, default=3, help="Contracts per strategy (default: 3)")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    parser.add_argument("--conservative", action="store_true", help="Use 2 contracts (conservative)")
    parser.add_argument("--mode", choices=["live", "poll"], default="live", help="live=streaming, poll=every 15min")
    parser.add_argument("--reset", action="store_true", help="Reset state (fresh start)")
    args = parser.parse_args()

    contracts = 2 if args.conservative else args.contracts

    trader = PaperTrader(account_size=args.account, contracts=contracts)

    # Load existing state
    if not args.reset and STATE_FILE.exists():
        trader.load_state()

    if args.status:
        trader.print_dashboard()
        return

    if args.reset:
        log.info("State reset. Starting fresh.")
        if STATE_FILE.exists():
            STATE_FILE.unlink()

    # Warmup indicators if no bar history
    if not trader.bar_history:
        warmup_from_history(trader)

    log.info(f"HTF Swing v3 Paper Trader starting | mode={args.mode} | {contracts}c | ${args.account:,.0f}")
    log.info(f"Risk: daily limit ${DAILY_LOSS_LIMIT:,.0f}, max DD ${MAX_TRAILING_DD:,.0f}, eval target ${EVAL_TARGET:,.0f}")

    if args.mode == "live":
        run_live(trader)
    else:
        run_polling(trader)


if __name__ == "__main__":
    main()
