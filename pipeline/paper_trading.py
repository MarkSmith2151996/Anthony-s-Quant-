"""Stage 4 — Paper Trading Replay."""

import logging
import time
from datetime import datetime
from pathlib import Path

from pipeline.config import (
    PROJECT_ROOT,
    _get_contract_spec,
    _build_risk_manager,
    stage_header,
    stage_footer,
)

logger = logging.getLogger("mcq_pipeline")


def stage_paper_trade(args, validated_strategies, data, prop_rules, session_config, events_calendar):
    """
    Replay the best strategy through the signal engine and paper trader.

    Args:
        validated_strategies: list of (strategy_dict, results_dict) from validation.
    """
    stage_header("PAPER TRADING REPLAY")
    t0 = time.time()

    if not validated_strategies:
        logger.warning("No validated strategies available for paper trading.")
        stage_footer("PAPER TRADING", time.time() - t0, "skipped (no strategies)")
        return

    from strategies.generator import GeneratedStrategy
    from live.signal_engine import SignalEngine
    from live.paper_trader import PaperTrader
    from live.dashboard import TradingDashboard
    from live.alerts import AlertManager, ConsoleChannel, FileChannel

    contract_spec = _get_contract_spec(args.symbol)
    rm = _build_risk_manager(prop_rules, session_config, events_calendar, contract_spec)

    # Pick the best strategy (first in the validated list, already ranked)
    best_sd, best_res = validated_strategies[0]
    best_name = best_res["strategy_name"]
    logger.info("Paper trading with best strategy: '%s'", best_name)
    logger.info(
        "  MC Grade: %s  |  MC Score: %.1f  |  WF Efficiency: %.2f",
        best_res["mc_score"].grade,
        best_res["mc_score"].composite_score,
        best_res.get("wf_efficiency", 0),
    )

    # Reconstruct strategy
    try:
        strategy = GeneratedStrategy.from_dict(best_sd)
    except Exception:
        logger.error("Could not reconstruct best strategy '%s'", best_name)
        stage_footer("PAPER TRADING", time.time() - t0, "failed")
        return

    # ── Setup ──────────────────────────────────────────────────────
    account_state = rm.init_account(prop_rules.account_size)

    # Alert manager with console and file channels
    alert_manager = AlertManager()
    alert_manager.add_channel(ConsoleChannel())
    alert_manager.add_channel(
        FileChannel(filepath=str(PROJECT_ROOT / "logs" / "paper_alerts.log")),
    )

    # Signal engine
    engine = SignalEngine(
        strategies=[strategy],
        risk_manager=rm,
        contract_spec=contract_spec,
        account_state=account_state,
        prop_rules=prop_rules,
    )

    # Paper trader
    trader = PaperTrader(
        risk_manager=_build_risk_manager(
            prop_rules, session_config, events_calendar, contract_spec,
        ),
        contract_spec=contract_spec,
        prop_rules=prop_rules,
        initial_balance=prop_rules.account_size,
        slippage_ticks=2,
        log_dir=str(PROJECT_ROOT / "logs" / "paper_trading"),
    )

    # Wire signal engine to paper trader
    engine.on_signal(trader.on_signal)

    # Dashboard
    dashboard = TradingDashboard(
        prop_rules=prop_rules,
        output_dir=str(Path(args.output_dir)),
    )

    # ── Replay ─────────────────────────────────────────────────────
    primary_tf = "1m" if "1m" in data else sorted(data.keys())[0]
    df_replay = data[primary_tf]

    total_bars = len(df_replay)
    logger.info("Replaying %d bars (speed=%.2fs per bar)...", total_bars, args.paper_speed)

    progress_interval = max(1, total_bars // 20)

    def replay_callback(bar_idx, total, bar_signals):
        """Progress callback during replay."""
        row = df_replay.row(bar_idx, named=True)
        ts = row.get("timestamp")
        price = row.get("close", 0.0)

        if isinstance(ts, datetime):
            trader.on_price_update(ts, price)
        elif ts is not None:
            try:
                ts_dt = datetime.fromisoformat(str(ts))
                trader.on_price_update(ts_dt, price)
            except (ValueError, TypeError):
                pass

        if bar_idx % progress_interval == 0 or bar_idx == total - 1:
            pct = (bar_idx + 1) / total * 100
            n_trades = len(trader.account.trade_history)
            balance = trader.account.account_state.current_balance
            logger.info(
                "  Replay progress: %d/%d (%.0f%%) | %d trades | balance=$%.2f",
                bar_idx + 1, total, pct, n_trades, balance,
            )

    signals = engine.replay(
        df_replay,
        speed=args.paper_speed,
        callback=replay_callback,
    )

    # ── Results ────────────────────────────────────────────────────
    print()
    print("  Paper Trading Complete")
    print("  " + "-" * 50)
    print(f"  Signals generated:  {len(signals)}")
    print(f"  Signals executed:   {trader.account.signals_executed}")
    print(f"  Signals rejected:   {trader.account.signals_rejected}")
    print(f"  Trades completed:   {len(trader.account.trade_history)}")

    # Print status and daily summary
    trader.print_status()
    trader.print_daily_summary()
    trader.print_trade_log(last_n=20)

    # Dashboard update and report
    dashboard.update(trader.account)
    dashboard.print_status()

    try:
        report_path = dashboard.generate_report(output_dir=str(Path(args.output_dir)))
        logger.info("Performance report saved to %s", report_path)
    except Exception:
        logger.warning("Failed to generate HTML dashboard report")

    # Export paper trading results
    try:
        results_path = trader.export_results(
            filepath=Path(args.output_dir) / "paper_trading_results.json",
        )
        logger.info("Paper trading results exported to %s", results_path)
    except Exception:
        logger.warning("Failed to export paper trading results")

    elapsed = time.time() - t0
    n_trades = len(trader.account.trade_history)
    final_balance = trader.account.account_state.current_balance
    net_pnl = final_balance - prop_rules.account_size
    stage_footer(
        "PAPER TRADING", elapsed,
        f"{n_trades} trades, net P&L ${net_pnl:+,.2f}, balance ${final_balance:,.2f}",
    )
