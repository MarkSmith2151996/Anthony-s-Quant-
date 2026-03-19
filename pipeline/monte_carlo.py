"""Stage 2 — Monte Carlo Stress Testing."""

import logging
import time
import traceback
from pathlib import Path

from pipeline.config import (
    _get_contract_spec,
    stage_header,
    stage_footer,
)

logger = logging.getLogger("mcq_pipeline")


def stage_monte_carlo(args, top_strategies, prop_rules) -> list:
    """
    Run Monte Carlo simulation on each top strategy, score and rank.

    Args:
        top_strategies: list of (strategy_dict, BacktestResult) from search stage.

    Returns list of (strategy_dict, MCResult, StrategyScore) tuples.
    """
    stage_header("MONTE CARLO STRESS TESTING")
    t0 = time.time()

    from monte_carlo.simulator import MonteCarloSimulator, MCConfig
    from monte_carlo.scoring import StrategyScorer
    from monte_carlo.visualization import MCVisualizer

    contract_spec = _get_contract_spec(args.symbol)

    mc_config = MCConfig(
        n_simulations=args.mc_sims,
        initial_capital=prop_rules.account_size,
        prop_firm_rules=prop_rules,
        tick_value=contract_spec.tick_value,
        n_workers=args.workers,
    )

    simulator = MonteCarloSimulator(mc_config)
    scorer = StrategyScorer(prop_rules=prop_rules)
    viz = MCVisualizer(output_dir=str(Path(args.output_dir) / "charts"))

    mc_results = []  # (strategy_dict, mc_result)
    total = len(top_strategies)

    for i, (strategy_dict, bt_result) in enumerate(top_strategies, 1):
        name = bt_result.strategy_name
        trades = bt_result.trades
        n_trades = len(trades)

        if n_trades < 5:
            logger.warning(
                "[%d/%d] Skipping '%s' — only %d trades (need >= 5)",
                i, total, name, n_trades,
            )
            continue

        logger.info(
            "[%d/%d] Running MC simulation for '%s' (%d trades, %d sims)...",
            i, total, name, n_trades, args.mc_sims,
        )

        try:
            mc_result = simulator.run(trades, strategy_name=name)
            mc_results.append((strategy_dict, mc_result))

            logger.info(
                "  -> median=$%.0f  P(profit)=%.1f%%  P(ruin)=%.1f%%  "
                "pass_rate=%.1f%%  composite=%.1f",
                mc_result.median_return,
                mc_result.probability_of_profit * 100,
                mc_result.probability_of_ruin * 100,
                mc_result.prop_firm_pass_rate * 100,
                mc_result.composite_score,
            )
        except Exception:
            logger.error("MC simulation failed for '%s':", name)
            logger.error(traceback.format_exc())
            continue

    if not mc_results:
        logger.warning("No strategies survived Monte Carlo simulation.")
        stage_footer("MONTE CARLO", time.time() - t0, "0 survivors")
        return []

    # ── Score all MC results ───────────────────────────────────────
    mc_result_objects = [mcr for _, mcr in mc_results]
    scores = scorer.score_batch(mc_result_objects)

    # Rank by composite score (viable strategies first)
    ranked_scores = scorer.rank(scores, top_n=args.top, viable_only=False)

    # Print the MC leaderboard
    scorer.print_leaderboard(ranked_scores, top_n=args.top)

    # ── Generate visualization for top strategies ──────────────────
    for score in ranked_scores[:min(5, len(ranked_scores))]:
        try:
            viz.full_report(score.mc_result, strategy_score=score)
        except Exception:
            logger.warning("Failed to generate charts for '%s'", score.strategy_name)

    # Generate leaderboard comparison chart
    if len(ranked_scores) >= 2:
        try:
            viz.leaderboard_comparison(ranked_scores, top_n=min(10, len(ranked_scores)))
        except Exception:
            logger.warning("Failed to generate leaderboard comparison chart")

    # ── Export MC report ───────────────────────────────────────────
    try:
        report_path = scorer.export_report(
            ranked_scores,
            filepath=Path(args.output_dir) / "mc_analysis_report.json",
        )
        logger.info("MC analysis report saved to %s", report_path)
    except Exception:
        logger.warning("Failed to export MC report")

    # ── Build combined result list ─────────────────────────────────
    # Map strategy_name -> strategy_dict for lookup
    name_to_dict = {}
    for sd, mcr in mc_results:
        name_to_dict[mcr.strategy_name] = sd

    scored_results = []
    for score in ranked_scores:
        sd = name_to_dict.get(score.strategy_name, {})
        scored_results.append((sd, score.mc_result, score))

    elapsed = time.time() - t0
    viable_count = sum(1 for s in ranked_scores if s.is_viable)
    stage_footer(
        "MONTE CARLO", elapsed,
        f"{len(ranked_scores)} scored, {viable_count} viable",
    )
    return scored_results
