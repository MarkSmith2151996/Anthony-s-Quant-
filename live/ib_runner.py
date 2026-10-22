"""IB Paper Trading Runner — connects IB streaming to the MCQ Engine.

Entry point for running a 30-day prop firm paper trading simulation
with live IB data and a TradingView-style web dashboard.

Usage:
    python live/ib_runner.py --symbol MNQ --interval 15m --firm topstep_50k
    python live/ib_runner.py --symbol MNQ --interval 15m --firm topstep_50k --strategy strategies/saved/my_strat.json
    python live/ib_runner.py --symbol MNQ --interval 15m --firm topstep_50k --port 8080

Architecture:
    IB Gateway → IBProvider.stream_bars() → SignalEngine.on_bar() → PaperTrader → WebDashboard
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.data.providers.interactive_brokers import IBProvider
from engine.risk_manager import RiskManager
from engine.utils import (
    CONTRACT_SPECS,
    load_prop_firm_rules,
    load_session_config,
    load_events_calendar,
)
from live.signal_engine import SignalEngine
from live.paper_trader import PaperTrader
from live.web_dashboard import WebDashboard, _bar_to_tv, _trade_to_ws

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


# ── Strategy loading ─────────────────────────────────────────────

def load_strategy(strategy_path: str | None, symbol: str):
    """Load a strategy from a JSON file, or build a default EMA crossover."""
    if strategy_path:
        from strategies.serializer import StrategySerializer
        serializer = StrategySerializer()
        strategy = serializer.load(strategy_path)
        logger.info("Loaded strategy: %s from %s", strategy.name, strategy_path)
        return [strategy]

    # Default: EMA crossover
    from strategies.ema_crossover import EMACrossoverStrategy
    strategy = EMACrossoverStrategy(
        fast_period=9,
        slow_period=21,
        atr_period=14,
        atr_sl_multiplier=2.0,
        atr_tp_multiplier=4.0,
    )
    logger.info("Using default strategy: %s", strategy.name)
    return [strategy]


# ── IB Runner ────────────────────────────────────────────────────

class IBRunner:
    """Orchestrates IB data → signal engine → paper trader → dashboard."""

    def __init__(
        self,
        symbol: str,
        interval: str,
        firm_profile: str,
        strategy_path: str | None = None,
        port: int = 8080,
        ib_host: str | None = None,
        ib_port: int | None = None,
        ib_client_id: int | None = None,
    ):
        self.symbol = symbol.upper()
        self.interval = interval
        self.firm_profile = firm_profile
        self.strategy_path = strategy_path
        self.dashboard_port = port

        # Contract spec
        if self.symbol not in CONTRACT_SPECS:
            raise ValueError(
                f"Unknown symbol: {self.symbol}. "
                f"Available: {list(CONTRACT_SPECS.keys())}"
            )
        self.contract_spec = CONTRACT_SPECS[self.symbol]

        # Prop firm rules
        self.prop_rules = load_prop_firm_rules(CONFIG_DIR, firm_profile)
        logger.info(
            "Prop firm: %s | account: $%,.0f | daily limit: $%,.0f | max DD: $%,.0f",
            self.prop_rules.firm_name,
            self.prop_rules.account_size,
            self.prop_rules.daily_loss_limit,
            self.prop_rules.max_drawdown,
        )

        # Session & events config
        self.session_config = load_session_config(CONFIG_DIR)
        self.events_calendar = load_events_calendar(CONFIG_DIR)

        # Risk manager
        self.risk_manager = RiskManager(
            prop_rules=self.prop_rules,
            session_config=self.session_config,
            events_calendar=self.events_calendar,
            contract_spec=self.contract_spec,
        )

        # Strategies
        self.strategies = load_strategy(strategy_path, self.symbol)

        # Signal engine
        account_state = self.risk_manager.init_account(self.prop_rules.account_size)
        self.signal_engine = SignalEngine(
            strategies=self.strategies,
            risk_manager=self.risk_manager,
            contract_spec=self.contract_spec,
            account_state=account_state,
            prop_rules=self.prop_rules,
            primary_timeframe=interval,
        )

        # Paper trader
        self.paper_trader = PaperTrader(
            risk_manager=self.risk_manager,
            contract_spec=self.contract_spec,
            prop_rules=self.prop_rules,
            initial_balance=self.prop_rules.account_size,
        )

        # Wire signal engine → paper trader
        self.signal_engine.on_signal(self.paper_trader.on_signal)

        # Web dashboard
        self.dashboard = WebDashboard(
            symbol=self.symbol,
            interval=interval,
            firm_name=self.prop_rules.firm_name,
        )

        # Wire paper trader → dashboard
        self.paper_trader.on_trade(self._on_trade)
        self.paper_trader.on_alert(self._on_alert)

        # IB provider
        self.provider = IBProvider(
            host=ib_host,
            port=ib_port,
            client_id=ib_client_id,
        )

        self._running = False
        self._reconnect_delays = [5, 10, 30, 60]

    # ── Callbacks ────────────────────────────────────────────────

    def _on_trade(self, trade) -> None:
        """Push completed trade to dashboard."""
        self.dashboard.push_trade(_trade_to_ws(trade))
        self._push_account()

    def _on_alert(self, alert_type: str, message: str) -> None:
        """Log alerts."""
        logger.warning("[ALERT] %s: %s", alert_type, message)

    def _push_account(self) -> None:
        """Push current account state to dashboard."""
        self.dashboard.push_account(self.paper_trader.get_status())

    # ── Main loop ────────────────────────────────────────────────

    def run(self) -> None:
        """Start the runner — blocking until Ctrl+C."""
        self._running = True

        # Start dashboard
        self.dashboard.start(port=self.dashboard_port)
        logger.info("Dashboard at http://localhost:%d", self.dashboard_port)

        # Start signal engine
        self.signal_engine.start()

        # Backfill historical bars for chart
        self._backfill_history()

        # Push initial account state
        self._push_account()

        # Stream loop with reconnection
        reconnect_idx = 0
        while self._running:
            try:
                self._stream_loop()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                delay = self._reconnect_delays[
                    min(reconnect_idx, len(self._reconnect_delays) - 1)
                ]
                logger.error(
                    "Stream error: %s — reconnecting in %ds...", exc, delay,
                )
                reconnect_idx += 1
                time.sleep(delay)

                # Reconnect IB
                try:
                    self.provider.disconnect()
                except Exception:
                    pass

        self._shutdown()

    def _stream_loop(self) -> None:
        """Stream bars from IB and feed through the engine."""
        logger.info("Starting bar stream for %s (%s)...", self.symbol, self.interval)

        for bar_df in self.provider.stream_bars(self.symbol, self.interval):
            if not self._running:
                break

            bar_row = bar_df.row(0, named=True)

            # Feed to signal engine → paper trader
            self.signal_engine.on_bar(bar_row)

            # Update paper trader with latest price
            ts = bar_row.get("timestamp", datetime.now())
            if not isinstance(ts, datetime):
                ts = datetime.fromisoformat(str(ts))
            self.paper_trader.on_price_update(ts, bar_row["close"])

            # Push bar to dashboard
            tv_bar = _bar_to_tv(bar_row)
            self.dashboard.push_bar(tv_bar)

            # Push account state
            self._push_account()

    def _backfill_history(self) -> None:
        """Fetch recent historical bars for chart display."""
        try:
            end = datetime.now().strftime("%Y-%m-%d")
            # Backfill last 5 days for chart context
            from datetime import timedelta
            start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

            logger.info("Backfilling history: %s %s from %s to %s", self.symbol, self.interval, start, end)
            df = self.provider.get_historical_bars(
                self.symbol, self.interval, start, end,
            )

            # Convert to TV format and send to dashboard
            bars = []
            for row in df.iter_rows(named=True):
                bars.append(_bar_to_tv(row))

            self.dashboard.set_history(bars)

            # Feed historical bars through signal engine for warm-up
            for row in df.iter_rows(named=True):
                self.signal_engine.on_bar(row)

            logger.info("Backfilled %d bars", len(bars))

        except Exception as exc:
            logger.warning("Backfill failed (non-fatal): %s", exc)

    def _shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        logger.info("Shutting down...")

        # Stop engine
        self.signal_engine.stop()

        # Export results
        try:
            filepath = self.paper_trader.export_results()
            logger.info("Results exported to %s", filepath)
        except Exception as exc:
            logger.error("Failed to export results: %s", exc)

        # Print summary
        self.paper_trader.print_status()
        self.paper_trader.print_daily_summary()
        self.paper_trader.print_trade_log()

        # Disconnect IB
        try:
            self.provider.disconnect()
        except Exception:
            pass

        # Stop dashboard
        self.dashboard.stop()

        logger.info("Shutdown complete")


# ── CLI ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MCQ Engine — IB Paper Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python live/ib_runner.py --symbol MNQ --interval 15m --firm topstep_50k
    python live/ib_runner.py --symbol NQ --interval 5m --firm apex_50k --port 9090
    python live/ib_runner.py --symbol MNQ --interval 15m --firm topstep_50k --strategy strategies/saved/my_strat.json
""",
    )
    parser.add_argument(
        "--symbol", default="MNQ",
        help="Symbol to trade (default: MNQ)",
    )
    parser.add_argument(
        "--interval", default="15m",
        help="Bar interval (default: 15m)",
    )
    parser.add_argument(
        "--firm", default="topstep_50k",
        help="Prop firm profile from config/prop_firms.yaml (default: topstep_50k)",
    )
    parser.add_argument(
        "--strategy", default=None,
        help="Path to a saved strategy JSON file (default: EMA crossover)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Web dashboard port (default: 8080)",
    )
    parser.add_argument(
        "--ib-host", default=None,
        help="IB Gateway host (default: $IB_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--ib-port", type=int, default=None,
        help="IB Gateway port (default: $IB_PORT or 4002)",
    )
    parser.add_argument(
        "--ib-client-id", type=int, default=None,
        help="IB client ID (default: $IB_CLIENT_ID or 1)",
    )
    return parser.parse_args()


def main():
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    print()
    print("=" * 60)
    print("  MCQ Engine — IB Paper Trading Runner")
    print("=" * 60)
    print(f"  Symbol:     {args.symbol}")
    print(f"  Interval:   {args.interval}")
    print(f"  Firm:       {args.firm}")
    print(f"  Strategy:   {args.strategy or 'EMA Crossover (default)'}")
    print(f"  Dashboard:  http://localhost:{args.port}")
    print("=" * 60)
    print()

    runner = IBRunner(
        symbol=args.symbol,
        interval=args.interval,
        firm_profile=args.firm,
        strategy_path=args.strategy,
        port=args.port,
        ib_host=args.ib_host,
        ib_port=args.ib_port,
        ib_client_id=args.ib_client_id,
    )

    # Handle Ctrl+C gracefully
    def _signal_handler(signum, frame):
        logger.info("Received signal %d — initiating shutdown...", signum)
        runner._running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    runner.run()


if __name__ == "__main__":
    main()
