#!/usr/bin/env python3
"""
BerjQuant Forward Test — HTF Swing v3 on LucidFlex 150K.

Processes 15m MNQ bars through RSI + IB + MOM strategies, tracks paper
trades in SQLite, enforces LucidFlex prop firm rules.

Usage:
    python ingest.py feed data/mnq_15m_2026-03-25.csv   # Ingest bars
    python ingest.py feed data/mnq_15m_backfill.csv      # Bulk ingest
    python ingest.py status                               # Current state
    python ingest.py trades                               # Recent trades
    python ingest.py trades --days 30 --strategy RSI      # Filtered
    python ingest.py performance                          # Monthly breakdown
    python ingest.py compare                              # Live vs backtest
    python ingest.py reset                                # Wipe everything
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# ═══════════════════════════════════════════════════════════════════
# LUCIDFLEX 150K CONFIG
# ═══════════════════════════════════════════════════════════════════

DB_PATH = Path(__file__).parent / "forward_test.db"

CONTRACTS = 3
TICK_SIZE = 0.25
POINT_VALUE = 2.0
SLIP_TICKS = 2
COMM_PER_SIDE = 0.62
EXCH_PER_SIDE = 0.27
SLIP_PTS = SLIP_TICKS * TICK_SIZE  # 0.50
RT_COST = (COMM_PER_SIDE + EXCH_PER_SIDE) * 2 * CONTRACTS + SLIP_TICKS * TICK_SIZE * POINT_VALUE * 2 * CONTRACTS  # $11.34

STRATS = {
    "RSI": {"sl_ticks": 60, "tp_ticks": 400, "max_hold": 5},
    "IB":  {"sl_ticks": 80, "tp_ticks": 480, "max_hold": 10},
    "MOM": {"sl_ticks": 60, "tp_ticks": 400, "max_hold": 5},
}

# LucidFlex 150K rules
FLATTEN_HHMM = 1645       # 4:45 PM ET — LucidFlex session end
INITIAL_BALANCE = 150_000.0
DAILY_LOSS_LIMIT = None   # LucidFlex has NO daily loss limit
MLL = -4500.0             # EOD trailing drawdown
EVAL_TARGET = 9000.0      # Profit target to pass eval
EVAL_CONSISTENCY = 0.50   # Best day ≤ 50% of total profit (eval only)


# ═══════════════════════════════════════════════════════════════════
# INDICATORS (identical to run_htf_swing.py)
# ═══════════════════════════════════════════════════════════════════

def calc_ema(data, period):
    n = len(data)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    ema[period - 1] = np.mean(data[:period])
    k = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = data[i] * k + ema[i - 1] * (1 - k)
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
    rsi[period] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    for i in range(period, n - 1):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rsi[i + 1] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return rsi


def calc_atr(highs, lows, closes, period=14):
    n = len(highs)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    tr[0] = highs[0] - lows[0]
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATORS (match run_htf_swing.py exactly)
# ═══════════════════════════════════════════════════════════════════

def sig_rsi_extreme(closes, period=7, ob=70, os_=30):
    rsi = calc_rsi(closes, period)
    n = len(closes)
    sigs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if not np.isnan(rsi[i]):
            if rsi[i] < os_:
                sigs[i] = 1
            elif rsi[i] > ob:
                sigs[i] = -1
    return sigs, rsi


def sig_ib_breakout(opens, highs, lows, closes, hhmm, dates):
    n = len(opens)
    ib_data = {}
    for i in range(n):
        if hhmm[i] < 1000:
            d = dates[i]
            if d not in ib_data:
                ib_data[d] = {"high": highs[i], "low": lows[i]}
            else:
                ib_data[d]["high"] = max(ib_data[d]["high"], highs[i])
                ib_data[d]["low"] = min(ib_data[d]["low"], lows[i])
    sigs = np.zeros(n, dtype=np.int8)
    traded = {}
    for i in range(n):
        if hhmm[i] < 1000 or hhmm[i] >= 1530:
            continue
        d = dates[i]
        if d not in ib_data or d in traded:
            continue
        ib_h, ib_l = ib_data[d]["high"], ib_data[d]["low"]
        if ib_h - ib_l < 2.0:
            continue
        if highs[i] > ib_h:
            sigs[i] = 1; traded[d] = True
        elif lows[i] < ib_l:
            sigs[i] = -1; traded[d] = True
    return sigs


def sig_momentum_bar(opens, highs, lows, closes, volumes, atr_mult=1.0, vol_mult=1.0):
    atr = calc_atr(highs, lows, closes, 14)
    ema21 = calc_ema(closes, 21)
    n = len(closes)
    avg_vol = np.full(n, np.nan)
    for i in range(20, n):
        avg_vol[i] = np.mean(volumes[i - 20:i])
    sigs = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if np.isnan(atr[i]) or np.isnan(ema21[i]) or np.isnan(avg_vol[i]):
            continue
        if (highs[i] - lows[i]) < atr[i] * atr_mult:
            continue
        if volumes[i] < avg_vol[i] * vol_mult:
            continue
        bar_dir = 1 if closes[i] > opens[i] else -1
        trend_dir = 1 if closes[i] > ema21[i] else -1
        if bar_dir == trend_dir:
            sigs[i] = bar_dir
    return sigs


# ═══════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS bars (
            timestamp TEXT PRIMARY KEY,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            date_et TEXT, hhmm INTEGER
        );
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, direction TEXT, contracts INTEGER,
            signal_ts TEXT, entry_ts TEXT, entry_price REAL,
            exit_ts TEXT, exit_price REAL, exit_reason TEXT,
            gross_pnl REAL, cost REAL, net_pnl REAL,
            bars_held INTEGER, rsi_at_signal REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS positions (
            strategy TEXT PRIMARY KEY,
            direction INTEGER DEFAULT 0,
            entry_price REAL DEFAULT 0,
            entry_ts TEXT,
            stop_price REAL DEFAULT 0,
            target_price REAL DEFAULT 0,
            bars_held INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS pending_signals (
            strategy TEXT PRIMARY KEY,
            direction INTEGER DEFAULT 0,
            signal_ts TEXT
        );
        CREATE TABLE IF NOT EXISTS account (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades(entry_ts);
        CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
        CREATE INDEX IF NOT EXISTS idx_bars_date ON bars(date_et);
    """)
    cur = conn.execute("SELECT COUNT(*) FROM account")
    if cur.fetchone()[0] == 0:
        conn.execute("INSERT INTO account VALUES ('balance', ?)", (str(INITIAL_BALANCE),))
        conn.execute("INSERT INTO account VALUES ('peak_balance', ?)", (str(INITIAL_BALANCE),))
        conn.execute("INSERT INTO account VALUES ('start_date', ?)", (datetime.now().isoformat(),))
        conn.execute("INSERT INTO account VALUES ('bars_processed', '0')")
        conn.execute("INSERT INTO account VALUES ('last_bar_ts', '')")
        conn.commit()
    for strat in STRATS:
        conn.execute("INSERT OR IGNORE INTO positions (strategy) VALUES (?)", (strat,))
        conn.execute("INSERT OR IGNORE INTO pending_signals (strategy) VALUES (?)", (strat,))
    conn.commit()
    return conn


def get_account(conn) -> dict:
    return {r["key"]: r["value"] for r in conn.execute("SELECT * FROM account").fetchall()}


def set_account(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO account VALUES (?, ?)", (key, str(value)))


# ═══════════════════════════════════════════════════════════════════
# STRATEGY ENGINE
# ═══════════════════════════════════════════════════════════════════

def process_bar(conn: sqlite3.Connection, bar: dict) -> list[dict]:
    ts = bar["timestamp"]
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    vol = bar.get("volume", 0)
    hhmm = bar["hhmm"]

    conn.execute(
        "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)",
        (ts, o, h, l, c, vol, bar["date_et"], hhmm),
    )

    rows = list(reversed(conn.execute(
        "SELECT * FROM bars ORDER BY timestamp DESC LIMIT 500"
    ).fetchall()))
    n = len(rows)
    if n < 25:
        acct = get_account(conn)
        set_account(conn, "bars_processed", int(acct.get("bars_processed", 0)) + 1)
        set_account(conn, "last_bar_ts", ts)
        conn.commit()
        return []

    opens = np.array([r["open"] for r in rows])
    highs = np.array([r["high"] for r in rows])
    lows = np.array([r["low"] for r in rows])
    closes = np.array([r["close"] for r in rows])
    volumes = np.array([r["volume"] for r in rows], dtype=float)
    hhmms = np.array([r["hhmm"] for r in rows])
    dates_et = [r["date_et"] for r in rows]

    rsi_sigs, rsi_vals = sig_rsi_extreme(closes, 7, 70, 30)
    ib_sigs = sig_ib_breakout(opens, highs, lows, closes, hhmms, dates_et)
    mom_sigs = sig_momentum_bar(opens, highs, lows, closes, volumes, 1.0, 1.0)
    signals_map = {"RSI": rsi_sigs, "IB": ib_sigs, "MOM": mom_sigs}

    completed = []
    last_idx = n - 1

    for strat_name, params in STRATS.items():
        pos = dict(conn.execute("SELECT * FROM positions WHERE strategy=?", (strat_name,)).fetchone())
        pending = dict(conn.execute("SELECT * FROM pending_signals WHERE strategy=?", (strat_name,)).fetchone())
        in_pos = pos["direction"] != 0
        sl_pts = params["sl_ticks"] * TICK_SIZE
        tp_pts = params["tp_ticks"] * TICK_SIZE

        # Flatten at 16:45 ET
        if in_pos and hhmm >= FLATTEN_HHMM:
            ex = c - pos["direction"] * SLIP_PTS
            gross = (ex - pos["entry_price"]) * pos["direction"] * POINT_VALUE * CONTRACTS
            net = gross - RT_COST
            trade = {
                "strategy": strat_name, "direction": "LONG" if pos["direction"] == 1 else "SHORT",
                "contracts": CONTRACTS, "signal_ts": pos["entry_ts"], "entry_ts": pos["entry_ts"],
                "entry_price": pos["entry_price"], "exit_ts": ts, "exit_price": ex,
                "exit_reason": "EOD_flatten", "gross_pnl": gross, "cost": RT_COST, "net_pnl": net,
                "bars_held": pos["bars_held"] + 1, "rsi_at_signal": None,
            }
            conn.execute(
                "INSERT INTO trades (strategy,direction,contracts,signal_ts,entry_ts,entry_price,"
                "exit_ts,exit_price,exit_reason,gross_pnl,cost,net_pnl,bars_held,rsi_at_signal) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(trade.values()))
            conn.execute("UPDATE positions SET direction=0, bars_held=0 WHERE strategy=?", (strat_name,))
            conn.execute("UPDATE pending_signals SET direction=0 WHERE strategy=?", (strat_name,))
            completed.append(trade)
            in_pos = False
            continue

        # Execute pending
        if pending["direction"] != 0 and not in_pos:
            if hhmm >= FLATTEN_HHMM - 15:
                conn.execute("UPDATE pending_signals SET direction=0 WHERE strategy=?", (strat_name,))
            else:
                d = pending["direction"]
                entry_px = o + d * SLIP_PTS
                conn.execute(
                    "UPDATE positions SET direction=?, entry_price=?, entry_ts=?, "
                    "stop_price=?, target_price=?, bars_held=0 WHERE strategy=?",
                    (d, entry_px, ts, entry_px - d * sl_pts, entry_px + d * tp_pts, strat_name))
                conn.execute("UPDATE pending_signals SET direction=0 WHERE strategy=?", (strat_name,))
                in_pos = True
                pos = {"direction": d, "entry_price": entry_px, "entry_ts": ts,
                       "stop_price": entry_px - d * sl_pts, "target_price": entry_px + d * tp_pts, "bars_held": 0}

        # Manage position
        if in_pos and pos["bars_held"] > 0:
            d = pos["direction"]
            ex = None; reason = ""
            if d == 1 and l <= pos["stop_price"]:
                ex = pos["stop_price"] - SLIP_PTS; reason = "SL"
            elif d == -1 and h >= pos["stop_price"]:
                ex = pos["stop_price"] + SLIP_PTS; reason = "SL"
            if ex is None:
                if d == 1 and h >= pos["target_price"]:
                    ex = pos["target_price"] - SLIP_PTS; reason = "TP"
                elif d == -1 and l <= pos["target_price"]:
                    ex = pos["target_price"] + SLIP_PTS; reason = "TP"
            if ex is None and pos["bars_held"] >= params["max_hold"]:
                ex = c - d * SLIP_PTS; reason = "max_hold"
            if ex is not None:
                gross = (ex - pos["entry_price"]) * d * POINT_VALUE * CONTRACTS
                net = gross - RT_COST
                trade = {
                    "strategy": strat_name, "direction": "LONG" if d == 1 else "SHORT",
                    "contracts": CONTRACTS, "signal_ts": pos["entry_ts"],
                    "entry_ts": pos["entry_ts"], "entry_price": pos["entry_price"],
                    "exit_ts": ts, "exit_price": ex, "exit_reason": reason,
                    "gross_pnl": gross, "cost": RT_COST, "net_pnl": net,
                    "bars_held": pos["bars_held"],
                    "rsi_at_signal": float(rsi_vals[last_idx]) if not np.isnan(rsi_vals[last_idx]) else None,
                }
                conn.execute(
                    "INSERT INTO trades (strategy,direction,contracts,signal_ts,entry_ts,entry_price,"
                    "exit_ts,exit_price,exit_reason,gross_pnl,cost,net_pnl,bars_held,rsi_at_signal) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(trade.values()))
                conn.execute("UPDATE positions SET direction=0, bars_held=0 WHERE strategy=?", (strat_name,))
                completed.append(trade)
                in_pos = False

        if in_pos:
            conn.execute("UPDATE positions SET bars_held = bars_held + 1 WHERE strategy=?", (strat_name,))

        # New signal
        if not in_pos:
            sig = signals_map[strat_name][last_idx]
            if sig != 0 and 930 <= hhmm < (FLATTEN_HHMM - 15):
                conn.execute("UPDATE pending_signals SET direction=?, signal_ts=? WHERE strategy=?",
                             (int(sig), ts, strat_name))

    # Update account
    acct = get_account(conn)
    for t in completed:
        bal = float(acct.get("balance", INITIAL_BALANCE)) + t["net_pnl"]
        peak = max(float(acct.get("peak_balance", INITIAL_BALANCE)), bal)
        set_account(conn, "balance", bal)
        set_account(conn, "peak_balance", peak)
        acct["balance"] = str(bal)
        acct["peak_balance"] = str(peak)

    set_account(conn, "bars_processed", int(acct.get("bars_processed", 0)) + 1)
    set_account(conn, "last_bar_ts", ts)
    conn.commit()
    return completed


# ═══════════════════════════════════════════════════════════════════
# CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════

def cmd_feed(args):
    conn = init_db()
    path = Path(args.file).expanduser()
    if not path.exists():
        print(f"File not found: {path}")
        return

    df = pl.read_csv(str(path), try_parse_dates=True) if path.suffix == ".csv" else pl.read_parquet(str(path))
    df = df.sort("timestamp")
    last_ts = get_account(conn).get("last_bar_ts", "")

    new_bars = 0
    new_trades = []
    for row in df.to_dicts():
        ts = str(row["timestamp"])
        if ts <= last_ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if "T" in ts else datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        hhmm = row.get("hhmm", dt.hour * 100 + dt.minute)
        date_et = row.get("date_et", dt.strftime("%Y-%m-%d"))
        if isinstance(hhmm, str):
            hhmm = int(hhmm)
        if hhmm < 930 or hhmm >= 1600:
            continue
        bar = {"timestamp": ts, "open": float(row["open"]), "high": float(row["high"]),
               "low": float(row["low"]), "close": float(row["close"]),
               "volume": float(row.get("volume", 0)), "hhmm": hhmm, "date_et": str(date_et)}
        trades = process_bar(conn, bar)
        new_trades.extend(trades)
        new_bars += 1

    acct = get_account(conn)
    bal = float(acct["balance"])
    pnl = bal - INITIAL_BALANCE
    print(f"\n  Ingested: {new_bars} bars, {len(new_trades)} trades")
    print(f"  Balance:  ${bal:,.2f} (PnL: ${pnl:+,.2f})")
    if new_trades:
        print(f"\n  New trades:")
        for t in new_trades:
            print(f"    {t['strategy']:3s} {t['direction']:5s} → {t['exit_reason']:11s} ${t['net_pnl']:>+8.2f}")
    conn.close()


def cmd_status(args):
    conn = init_db()
    acct = get_account(conn)
    bal = float(acct.get("balance", INITIAL_BALANCE))
    peak = float(acct.get("peak_balance", INITIAL_BALANCE))
    dd = bal - peak
    bars = int(acct.get("bars_processed", 0))
    last_bar = acct.get("last_bar_ts", "none")

    today = date.today().isoformat()
    today_trades = conn.execute("SELECT * FROM trades WHERE entry_ts LIKE ?", (f"{today}%",)).fetchall()
    today_pnl = sum(r["net_pnl"] for r in today_trades)

    total = conn.execute("SELECT COUNT(*), COALESCE(SUM(net_pnl), 0) FROM trades").fetchone()
    positions = conn.execute("SELECT * FROM positions WHERE direction != 0").fetchall()

    # EOD MLL check (uses closing balance, not intraday)
    daily_closes = conn.execute("""
        SELECT date_et, SUM(net_pnl) as day_pnl FROM trades
        GROUP BY substr(entry_ts, 1, 10) ORDER BY entry_ts
    """).fetchall()
    eod_cum = 0; eod_peak = 0; eod_dd = 0
    for row in daily_closes:
        eod_cum += row["day_pnl"]
        eod_peak = max(eod_peak, eod_cum)
        eod_dd = min(eod_dd, eod_cum - eod_peak)

    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  HTF SWING v3 — LucidFlex 150K Forward Test     ║")
    print(f"  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Balance:      ${bal:>12,.2f}                    ║")
    print(f"  ║  Total PnL:    ${bal - INITIAL_BALANCE:>+12,.2f}                    ║")
    print(f"  ║  EOD Drawdown: ${eod_dd:>+12,.2f}  (MLL: -$4,500)     ║")
    print(f"  ║  Daily loss:   No limit (LucidFlex)              ║")
    print(f"  ║  Bars:         {bars:>8,}                         ║")
    print(f"  ║  Trades:       {total[0]:>8,}                         ║")
    print(f"  ║  Last bar:     {last_bar[:19] if last_bar else 'none':>19}        ║")
    print(f"  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Today ({today}):                          ║")
    print(f"  ║    Trades: {len(today_trades):>3}  PnL: ${today_pnl:>+10,.2f}              ║")
    print(f"  ╠══════════════════════════════════════════════════╣")

    if eod_dd > MLL:
        print(f"  ║  MLL Status:   ✅ SAFE ({eod_dd:+,.0f} vs {MLL:,.0f})        ║")
    else:
        print(f"  ║  MLL Status:   ❌ BREACHED ({eod_dd:+,.0f})              ║")

    eval_pnl = bal - INITIAL_BALANCE
    if eval_pnl >= EVAL_TARGET:
        print(f"  ║  Eval Target:  ✅ PASSED (${eval_pnl:,.0f} ≥ ${EVAL_TARGET:,.0f})     ║")
    else:
        print(f"  ║  Eval Target:  ${eval_pnl:>+10,.0f} / ${EVAL_TARGET:,.0f}            ║")

    print(f"  ╚══════════════════════════════════════════════════╝")

    if positions:
        print(f"\n  Open positions:")
        for p in positions:
            d = "LONG" if p["direction"] == 1 else "SHORT"
            print(f"    {p['strategy']:3s} {d:5s} @ {p['entry_price']:.2f} (bars: {p['bars_held']})")
    conn.close()


def cmd_trades(args):
    conn = init_db()
    cutoff = (datetime.now() - timedelta(days=args.days)).isoformat()
    query = "SELECT * FROM trades WHERE entry_ts > ?"
    params = [cutoff]
    if args.strategy:
        query += " AND strategy = ?"
        params.append(args.strategy.upper())
    query += " ORDER BY entry_ts DESC LIMIT ?"
    params.append(args.limit)

    rows = conn.execute(query, params).fetchall()
    if not rows:
        print("\n  No trades found.")
        conn.close()
        return

    pnls = [r["net_pnl"] for r in rows]
    wins = sum(1 for p in pnls if p > 0)
    print(f"\n  Last {args.days} days: {len(rows)} trades, ${sum(pnls):+,.2f}, "
          f"WR: {wins}/{len(rows)} ({wins/len(rows)*100:.0f}%)")
    print(f"\n  {'Time':<20} {'Strat':>5} {'Dir':>5} {'Exit':>10} {'PnL':>10} {'Bars':>5}")
    print(f"  {'─'*60}")
    for r in rows:
        print(f"  {r['entry_ts'][:19]:<20} {r['strategy']:>5} {r['direction']:>5} "
              f"{r['exit_reason']:>10} ${r['net_pnl']:>+9.2f} {r['bars_held']:>5}")
    conn.close()


def cmd_performance(args):
    conn = init_db()
    rows = conn.execute("SELECT * FROM trades ORDER BY entry_ts").fetchall()
    if not rows:
        print("\n  No trades yet.")
        conn.close()
        return

    monthly = defaultdict(lambda: {"pnl": 0, "n": 0, "wins": 0})
    for r in rows:
        key = r["entry_ts"][:7]
        monthly[key]["pnl"] += r["net_pnl"]
        monthly[key]["n"] += 1
        monthly[key]["wins"] += 1 if r["net_pnl"] > 0 else 0

    cum = 0
    print(f"\n  {'Month':<10} {'Trades':>7} {'WR':>6} {'PnL':>12} {'Cumulative':>12}")
    print(f"  {'─'*52}")
    for k in sorted(monthly):
        g = monthly[k]
        cum += g["pnl"]
        wr = g["wins"] / g["n"] * 100 if g["n"] else 0
        flag = "✅" if g["pnl"] > 0 else "❌"
        print(f"  {k:<10} {g['n']:>7} {wr:>5.0f}% ${g['pnl']:>+11,.2f} ${cum:>+11,.2f} {flag}")

    vals = [g["pnl"] for g in monthly.values()]
    n = len(vals)
    print(f"\n  Average:  ${np.mean(vals):>+10,.0f}/mo")
    print(f"  Median:   ${np.median(vals):>+10,.0f}/mo")
    print(f"  Best:     ${max(vals):>+10,.0f}")
    print(f"  Worst:    ${min(vals):>+10,.0f}")
    print(f"  Positive: {sum(1 for v in vals if v > 0)}/{n}")
    conn.close()


def cmd_compare(args):
    conn = init_db()
    rows = conn.execute("SELECT * FROM trades ORDER BY entry_ts").fetchall()
    if not rows:
        print("\n  No trades yet. Feed some bars first.")
        conn.close()
        return

    pnls = [r["net_pnl"] for r in rows]
    daily = defaultdict(float)
    by_strat = defaultdict(list)
    for r in rows:
        daily[r["entry_ts"][:10]] += r["net_pnl"]
        by_strat[r["strategy"]].append(r["net_pnl"])

    n_months = len(set(r["entry_ts"][:7] for r in rows))
    cum = 0; peak = 0; max_dd = 0
    for d in sorted(daily):
        cum += daily[d]; peak = max(peak, cum); max_dd = min(max_dd, cum - peak)

    # Backtest benchmarks (ANTHONY CSV Y2 OOS, LucidFlex 16:45 session)
    bt = {"monthly": 7186, "wr": 25.7, "worst_day": -1252, "max_dd": -1927,
          "rr": {"RSI": 3.36, "IB": 3.14, "MOM": 3.78}}

    daily_vals = list(daily.values())
    live_monthly = sum(pnls) / max(n_months, 1)
    live_wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100

    print(f"\n  ┌{'─'*22}┬{'─'*14}┬{'─'*14}┐")
    print(f"  │{'Metric':<22}│{'Live':>14}│{'Backtest Y2':>14}│")
    print(f"  ├{'─'*22}┼{'─'*14}┼{'─'*14}┤")
    print(f"  │{'Monthly avg':<22}│${live_monthly:>+12,.0f}│${bt['monthly']:>+12,}│")
    print(f"  │{'Win rate':<22}│{live_wr:>13.1f}%│{bt['wr']:>13.1f}%│")
    print(f"  │{'Worst day':<22}│${min(daily_vals):>+12,.0f}│${bt['worst_day']:>+12,}│")
    print(f"  │{'Max drawdown':<22}│${max_dd:>+12,.0f}│${bt['max_dd']:>+12,}│")
    print(f"  │{'Trades/day':<22}│{len(pnls)/max(len(daily),1):>14.1f}│{'~7.3':>14}│")
    print(f"  └{'─'*22}┴{'─'*14}┴{'─'*14}┘")

    print(f"\n  Strategy R:R (live vs backtest):")
    for s in ["RSI", "IB", "MOM"]:
        pl_list = by_strat.get(s, [])
        if not pl_list:
            continue
        wins = [p for p in pl_list if p > 0]
        losses = [p for p in pl_list if p <= 0]
        rr = abs(np.mean(wins) / np.mean(losses)) if losses and wins else 0
        print(f"    {s}: {rr:.2f} (backtest: {bt['rr'].get(s, 0):.2f})")
    conn.close()


def cmd_reset(args):
    if args.confirm != "RESET":
        print("  Type: python ingest.py reset --confirm RESET")
        return
    conn = init_db()
    conn.executescript("DELETE FROM trades; DELETE FROM bars; DELETE FROM positions; DELETE FROM pending_signals; DELETE FROM account;")
    conn.execute("INSERT INTO account VALUES ('balance', ?)", (str(INITIAL_BALANCE),))
    conn.execute("INSERT INTO account VALUES ('peak_balance', ?)", (str(INITIAL_BALANCE),))
    conn.execute("INSERT INTO account VALUES ('start_date', ?)", (datetime.now().isoformat(),))
    conn.execute("INSERT INTO account VALUES ('bars_processed', '0')")
    conn.execute("INSERT INTO account VALUES ('last_bar_ts', '')")
    for strat in STRATS:
        conn.execute("INSERT INTO positions (strategy) VALUES (?)", (strat,))
        conn.execute("INSERT INTO pending_signals (strategy) VALUES (?)", (strat,))
    conn.commit()
    print("  ✅ Reset complete. Balance: $150,000.")
    conn.close()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BerjQuant Forward Test — HTF Swing v3 on LucidFlex 150K")
    sub = parser.add_subparsers(dest="command")

    p_feed = sub.add_parser("feed", help="Ingest 15m bars from CSV/Parquet")
    p_feed.add_argument("file", help="Path to bar data file")

    p_status = sub.add_parser("status", help="Current forward test status")

    p_trades = sub.add_parser("trades", help="Recent trade history")
    p_trades.add_argument("--days", type=int, default=7)
    p_trades.add_argument("--strategy", type=str, default=None)
    p_trades.add_argument("--limit", type=int, default=50)

    p_perf = sub.add_parser("performance", help="Monthly performance breakdown")

    p_compare = sub.add_parser("compare", help="Live vs backtest comparison")

    p_reset = sub.add_parser("reset", help="Wipe all data")
    p_reset.add_argument("--confirm", type=str, default="")

    args = parser.parse_args()
    if args.command == "feed": cmd_feed(args)
    elif args.command == "status": cmd_status(args)
    elif args.command == "trades": cmd_trades(args)
    elif args.command == "performance": cmd_performance(args)
    elif args.command == "compare": cmd_compare(args)
    elif args.command == "reset": cmd_reset(args)
    else: parser.print_help()


if __name__ == "__main__":
    main()
