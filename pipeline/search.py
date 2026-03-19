"""Stage 1 — Strategy Search."""

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

import polars as pl

from pipeline.config import (
    REPORTS_DIR,
    STRATEGIES_DIR,
    _get_contract_spec,
    _build_backtest_config,
    _save_json,
    stage_header,
    stage_footer,
)

logger = logging.getLogger("mcq_pipeline")


def stage_search(args, data, prop_rules, session_config, events_calendar) -> list:
    """
    Generate strategy combinations, run parallel backtests, filter and rank.

    Returns list of (strategy_dict, BacktestResult) tuples sorted by composite score.
    """
    stage_header("STRATEGY SEARCH")
    t0 = time.time()

    from signals.registry import SignalRegistry
    from strategies.generator import StrategyGenerator, ExitRules, SizingRules
    from strategies.serializer import StrategySerializer
    from engine.parallel_runner import ParallelRunner, RunConfig
    from engine.utils import BacktestConfig

    contract_spec = _get_contract_spec(args.symbol)
    bt_config = _build_backtest_config(args, data)

    # ── 1. Generate strategies ─────────────────────────────────────
    logger.info("Building strategy combinations...")

    registry = SignalRegistry()
    generator = StrategyGenerator(registry)

    # Define exit rule variations
    exit_variations = [
        ExitRules(stop_loss_type="fixed_points", stop_loss_value=4.0,
                  take_profit_type="fixed_points", take_profit_value=8.0),
        ExitRules(stop_loss_type="fixed_points", stop_loss_value=6.0,
                  take_profit_type="fixed_points", take_profit_value=12.0),
        ExitRules(stop_loss_type="atr_multiple", stop_loss_value=1.5,
                  take_profit_type="atr_multiple", take_profit_value=3.0),
        ExitRules(stop_loss_type="fixed_points", stop_loss_value=4.0,
                  take_profit_type="rr_ratio", take_profit_value=2.0),
        ExitRules(stop_loss_type="fixed_points", stop_loss_value=4.0,
                  take_profit_type="rr_ratio", take_profit_value=3.0),
    ]

    sizing_variations = [
        SizingRules(method="fixed", fixed_contracts=1),
        SizingRules(method="risk_pct", risk_pct=0.01),
    ]

    # Count possible combos first
    total_possible = generator.count_combinations(
        max_entry_signals=args.max_entry_signals,
        max_filters=args.max_filters,
        exit_variations=len(exit_variations),
        sizing_variations=len(sizing_variations),
    )
    logger.info(
        "Combinatorial space: %d possible strategies (capped at %d)",
        total_possible,
        args.max_strategies,
    )

    strategies = generator.generate(
        max_entry_signals=args.max_entry_signals,
        max_filters=args.max_filters,
        exit_variations=exit_variations,
        sizing_variations=sizing_variations,
        max_strategies=args.max_strategies,
    )
    logger.info("Generated %d strategies", len(strategies))

    if not strategies:
        logger.warning("No strategies generated. Check signal registry and parameters.")
        stage_footer("STRATEGY SEARCH", time.time() - t0, "0 strategies")
        return []

    # ── 2. Parallel backtesting ────────────────────────────────────
    run_config = RunConfig(
        data=data,
        config=bt_config,
        prop_rules=prop_rules,
        session_config=session_config,
        events_calendar=events_calendar,
        contract_spec=contract_spec,
        min_trades=10,
        min_profit_factor=0.0,
        min_sharpe=-999.0,
    )

    runner = ParallelRunner(run_config, n_workers=args.workers)

    if args.sequential:
        logger.info("Running backtests sequentially (debug mode)...")
        batch_result = runner.run_sequential(strategies)
    else:
        batch_result = runner.run(strategies)

    logger.info(
        "Backtest batch: %d completed, %d failed, %d passed filters (%.1f strats/sec)",
        batch_result.completed,
        batch_result.failed,
        batch_result.filtered,
        batch_result.total_strategies / batch_result.elapsed_seconds
        if batch_result.elapsed_seconds > 0 else 0,
    )

    if batch_result.errors:
        logger.warning("%d strategies failed:", len(batch_result.errors))
        for name, err in batch_result.errors[:5]:
            logger.warning("  %s: %s", name, err.split("\n")[0])
        if len(batch_result.errors) > 5:
            logger.warning("  ... and %d more", len(batch_result.errors) - 5)

    # ── 3. Rank by composite score ─────────────────────────────────
    top_results = ParallelRunner.rank_results(
        batch_result, metric="composite", top_n=args.top,
    )

    # Print leaderboard
    ParallelRunner.print_leaderboard(top_results, top_n=args.top)

    # ── 4. Save results ────────────────────────────────────────────
    # Save top strategy definitions for downstream stages
    serializer = StrategySerializer(strategies_dir=str(STRATEGIES_DIR))
    top_strategy_objects = []
    for strategy_dict, bt_result in top_results:
        from strategies.generator import GeneratedStrategy
        try:
            strat = GeneratedStrategy.from_dict(strategy_dict)
            top_strategy_objects.append(strat)
        except Exception:
            logger.warning("Could not reconstruct strategy from dict: %s", strategy_dict.get("name"))

    if top_strategy_objects:
        batch_path = serializer.save_batch(top_strategy_objects, batch_name="search_top")
        logger.info("Saved top %d strategies to %s", len(top_strategy_objects), batch_path)

    # Save search results summary
    search_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_generated": len(strategies),
        "total_completed": batch_result.completed,
        "total_failed": batch_result.failed,
        "total_passed_filter": batch_result.filtered,
        "elapsed_seconds": round(batch_result.elapsed_seconds, 2),
        "top_strategies": [],
    }
    for rank, (sd, br) in enumerate(top_results, 1):
        m = br.metrics
        search_summary["top_strategies"].append({
            "rank": rank,
            "name": br.strategy_name,
            "trades": m.total_trades if m else 0,
            "win_rate": round(m.win_rate, 1) if m else 0,
            "profit_factor": round(m.profit_factor, 2) if m else 0,
            "sharpe_ratio": round(m.sharpe_ratio, 2) if m else 0,
            "total_pnl": round(m.total_pnl, 2) if m else 0,
            "max_drawdown": round(m.max_drawdown, 2) if m else 0,
        })
    _save_json(search_summary, REPORTS_DIR / "search_results.json")

    elapsed = time.time() - t0
    stage_footer(
        "STRATEGY SEARCH", elapsed,
        f"{batch_result.filtered} passed -> top {len(top_results)} selected",
    )
    return top_results


def _load_saved_search_results() -> list:
    """
    Try to load search results from the last saved batch.

    Returns list of (strategy_dict, BacktestResult) tuples,
    or empty list if nothing is found.
    """
    from strategies.serializer import StrategySerializer
    from strategies.generator import GeneratedStrategy
    from engine.utils import BacktestResult, BacktestConfig

    serializer = StrategySerializer(strategies_dir=str(STRATEGIES_DIR))
    saved = serializer.list_saved()

    # Find the most recent batch file
    batch_files = [
        s for s in saved
        if s.get("type") == "batch" and "search_top" in s.get("name", "")
    ]

    if not batch_files:
        logger.info("No saved search results found in %s", STRATEGIES_DIR)
        return []

    # Sort by saved_at descending
    batch_files.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
    latest = batch_files[0]

    logger.info("Loading saved strategies from %s", latest["path"])
    try:
        strategies = serializer.load_batch(latest["path"])

        # We need BacktestResult objects — create minimal ones
        # (the MC stage only uses .trades, so we need at least a dummy config)
        dummy_config = BacktestConfig(
            symbol="MNQ",
            prop_firm_profile="topstep_50k",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        results = []
        for strat in strategies:
            # Use the strategy dict as the strategy_dict, and create a
            # placeholder BacktestResult. The MC stage will re-run if needed.
            sd = strat.to_dict()
            # We don't have trades — the user needs to re-run search or
            # run MC separately after producing backtests.
            logger.warning(
                "Strategy '%s' loaded but has no backtest trades. "
                "Monte Carlo will require re-running search.",
                strat.name,
            )
            # Return empty results — caller should handle gracefully
            br = BacktestResult(
                strategy_name=strat.name,
                config=dummy_config,
                trades=[],
                equity_curve=[],
            )
            results.append((sd, br))

        return results
    except Exception:
        logger.error("Failed to load saved strategies: %s", traceback.format_exc())
        return []
