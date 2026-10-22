#!/usr/bin/env python3
"""
HTF Swing v3 — Paper Trader Dashboard (Textual TUI)

Leave this open on your PC. It auto-refreshes every 5 seconds.

Usage:
    python3 pt_dashboard.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("US/Eastern")
except Exception:
    from datetime import timezone, timedelta
    ET = timezone(timedelta(hours=-4))

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Static, Header, Footer, DataTable, Label
ROOT = Path(__file__).parent
STATE_FILE = ROOT / "paper_trader_state.json"
HEALTH_FILE = ROOT / "paper_trader_health.json"
TRADES_FILE = ROOT / "reports" / "paper_trader" / "trades.csv"
LOG_FILE = ROOT / "reports" / "paper_trader" / "paper_trader.log"


def is_running(name: str) -> bool:
    r = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True)
    return bool(r.stdout.strip())


def load_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_trades() -> list[dict]:
    if not TRADES_FILE.exists():
        return []
    import csv
    with open(TRADES_FILE) as f:
        return list(csv.DictReader(f))


def tail_log(n: int = 8) -> list[str]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


class StatusBar(Static):
    """Top status bar showing connection and account info."""

    def update_status(self):
        state = load_json(STATE_FILE)
        health = load_json(HEALTH_FILE)

        pt_up = is_running("paper_trader_live.py")
        sv_up = is_running("pt_supervisor.sh")

        now = datetime.now(ET)
        time_str = now.strftime("%H:%M:%S ET")
        date_str = now.strftime("%a %b %d")

        # Market status
        h, m = now.hour, now.minute
        hhmm = h * 100 + m
        weekday = now.weekday()
        if weekday >= 5:
            mkt = "[yellow]WEEKEND[/]"
        elif 930 <= hhmm < 1600:
            mkt = "[green bold]MARKET OPEN[/]"
        elif hhmm < 930:
            mkt = "[dim]PRE-MARKET[/]"
        else:
            mkt = "[dim]CLOSED[/]"

        # Process status
        if pt_up and sv_up:
            proc = "[green bold]LIVE[/]"
        elif sv_up and not pt_up:
            proc = "[yellow]RESTARTING[/]"
        else:
            proc = "[red bold]DOWN[/]"

        crashes = health.get("crash_count", 0)
        crash_str = f"  [red]crashes: {crashes}[/]" if crashes > 0 else ""

        # Account
        balance = state.get("balance", 0)
        initial = state.get("initial_balance", 150000)
        pnl = balance - initial
        daily = state.get("daily_pnl", 0)
        peak = state.get("peak_balance", initial)
        dd = balance - peak
        contracts = state.get("contracts", 3)

        pnl_color = "green" if pnl >= 0 else "red"
        daily_color = "green" if daily >= 0 else "red"
        dd_color = "green" if dd >= -2000 else "yellow" if dd >= -3500 else "red"

        # Positions
        positions = state.get("positions", {})
        if positions:
            pos_parts = []
            for strat, pos in positions.items():
                d = "L" if pos["direction"] == 1 else "S"
                pos_parts.append(f"{strat}:{d}@{pos['entry_px']:.0f}")
            pos_str = " | ".join(pos_parts)
        else:
            pos_str = "FLAT"

        self.update(
            f"  {date_str}  {time_str}  {mkt}    "
            f"Status: {proc}{crash_str}    "
            f"Contracts: {contracts}\n"
            f"  Balance: [bold]${balance:,.0f}[/]  "
            f"P&L: [{pnl_color}]${pnl:+,.0f}[/]  "
            f"Today: [{daily_color}]${daily:+,.0f}[/]  "
            f"DD: [{dd_color}]${dd:+,.0f}[/]  "
            f"Positions: {pos_str}"
        )


class PropFirmBar(Static):
    """Prop firm eval progress."""

    def update_status(self):
        state = load_json(STATE_FILE)
        balance = state.get("balance", 0)
        initial = state.get("initial_balance", 150000)
        peak = state.get("peak_balance", initial)
        pnl = balance - initial
        dd = balance - peak
        daily = state.get("daily_pnl", 0)
        daily_summary = state.get("daily_summary", {})
        worst_day = min((d["pnl"] for d in daily_summary.values()), default=0)
        worst_day = min(worst_day, daily)

        target = 9000
        pct = min(pnl / target * 100, 100) if target > 0 else 0
        bar_len = 30
        filled = max(int(pct / 100 * bar_len), 0)
        bar = "[green]" + "█" * filled + "[/]" + "[dim]░[/]" * (bar_len - filled)

        daily_ok = worst_day > -3000
        dd_ok = dd > -4500
        daily_color = "green" if daily_ok else "red"
        dd_color = "green" if dd_ok else "red"

        self.update(
            f"  Eval: ${pnl:+,.0f} / $9,000  {bar}  {pct:.0f}%    "
            f"Daily: [{daily_color}]${worst_day:+,.0f}[/] / -$3K [{daily_color}]{'OK' if daily_ok else 'FAIL'}[/]  "
            f"DD: [{dd_color}]${dd:+,.0f}[/] / -$4.5K [{dd_color}]{'OK' if dd_ok else 'FAIL'}[/]"
        )


class TradesTable(Static):
    """Recent trades table."""

    def update_trades(self):
        trades = load_trades()
        if not trades:
            self.update("  [dim]No trades yet. Waiting for market open...[/]")
            return

        # Last 15 trades
        recent = trades[-15:]
        lines = []
        lines.append(
            f"  [bold]{'Time':<17} {'Strat':<5} {'Dir':<6} {'Entry':>9} {'Exit':>9} "
            f"{'P&L':>9} {'Reason':<11}[/]"
        )
        lines.append(f"  {'─'*17} {'─'*5} {'─'*6} {'─'*9} {'─'*9} {'─'*9} {'─'*11}")

        for t in recent:
            pnl = float(t["net_pnl"])
            color = "green" if pnl > 0 else "red" if pnl < 0 else "dim"
            time_str = t["entry_time"][5:16]  # MM-DD HH:MM
            lines.append(
                f"  {time_str:<17} {t['strategy']:<5} {t['direction']:<6} "
                f"{float(t['entry_px']):>9.2f} {float(t['exit_px']):>9.2f} "
                f"[{color}]${pnl:>+8,.0f}[/] {t['reason']:<11}"
            )

        # Summary line
        total = sum(float(t["net_pnl"]) for t in trades)
        wins = sum(1 for t in trades if float(t["net_pnl"]) > 0)
        wr = wins / len(trades) * 100 if trades else 0
        lines.append(f"  {'─'*17} {'─'*5} {'─'*6} {'─'*9} {'─'*9} {'─'*9} {'─'*11}")
        tc = "green" if total >= 0 else "red"
        lines.append(
            f"  [bold]{len(trades)} trades  {wr:.0f}% win rate"
            f"{'':>28}[{tc}]${total:>+8,.0f}[/][/]"
        )

        self.update("\n".join(lines))


class DailySummary(Static):
    """Daily P&L bars."""

    def update_daily(self):
        state = load_json(STATE_FILE)
        daily = state.get("daily_summary", {})

        if not daily:
            self.update("  [dim]No completed trading days yet.[/]")
            return

        lines = []
        # Last 10 days
        days = sorted(daily.keys())[-10:]
        for d in days:
            pnl = daily[d]["pnl"]
            color = "green" if pnl > 0 else "red"
            bar_size = min(int(abs(pnl) / 50), 40)
            if pnl >= 0:
                bar = "[green]" + "█" * bar_size + "[/]"
            else:
                bar = "[red]" + "█" * bar_size + "[/]"
            lines.append(f"  {d[5:]}  [{color}]${pnl:>+8,.0f}[/]  {bar}")

        self.update("\n".join(lines))


class LogView(Static):
    """Recent log entries."""

    def update_log(self):
        lines = tail_log(6)
        if not lines:
            self.update("  [dim]No log entries yet.[/]")
            return

        formatted = []
        for l in lines:
            # Trim timestamp prefix for compactness
            if "] " in l:
                parts = l.split("] ", 1)
                ts = parts[0].split(" ")[-1] if parts[0] else ""
                msg = parts[1] if len(parts) > 1 else l
                # Color by level
                if "WARNING" in l or "ERROR" in l:
                    formatted.append(f"  [yellow]{ts}[/] {msg[:90]}")
                elif "OPEN" in msg or "CLOSE" in msg:
                    formatted.append(f"  [bold]{ts}[/] {msg[:90]}")
                else:
                    formatted.append(f"  [dim]{ts}[/] {msg[:90]}")
            else:
                formatted.append(f"  {l[:95]}")

        self.update("\n".join(formatted))


class PaperTraderDashboard(App):
    CSS = """
    Screen {
        background: $surface;
    }
    #status-bar {
        height: 3;
        background: $panel;
        border-bottom: solid $primary;
        padding: 0;
    }
    #prop-bar {
        height: 2;
        background: $panel;
        border-bottom: solid grey;
        padding: 0;
    }
    #main-area {
        height: 1fr;
    }
    #left-panel {
        width: 2fr;
        border-right: solid grey;
    }
    #right-panel {
        width: 1fr;
    }
    .section-title {
        height: 1;
        background: $primary-background;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    #trades-table {
        height: 1fr;
        overflow-y: auto;
    }
    #daily-summary {
        height: 1fr;
        border-bottom: solid grey;
    }
    #log-view {
        height: auto;
        max-height: 10;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatusBar(id="status-bar")
        yield PropFirmBar(id="prop-bar")
        with Horizontal(id="main-area"):
            with Vertical(id="left-panel"):
                yield Label(" TRADES", classes="section-title")
                yield TradesTable(id="trades-table")
            with Vertical(id="right-panel"):
                yield Label(" DAILY P&L", classes="section-title")
                yield DailySummary(id="daily-summary")
                yield Label(" LOG", classes="section-title")
                yield LogView(id="log-view")
        yield Footer()

    def on_mount(self):
        self.title = "HTF Swing v3 — Paper Trader"
        self.sub_title = "MNQ | 3c | Topstep 150K"
        self.refresh_data()
        self.set_interval(5, self.refresh_data)

    def refresh_data(self):
        self.query_one("#status-bar", StatusBar).update_status()
        self.query_one("#prop-bar", PropFirmBar).update_status()
        self.query_one("#trades-table", TradesTable).update_trades()
        self.query_one("#daily-summary", DailySummary).update_daily()
        self.query_one("#log-view", LogView).update_log()

    def action_refresh(self):
        self.refresh_data()


if __name__ == "__main__":
    app = PaperTraderDashboard()
    app.run()
