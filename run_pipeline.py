#!/usr/bin/env python3
"""
MCQ Engine — Full Pipeline Runner

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --stage search     # Strategy search only
    python run_pipeline.py --stage mc         # Monte Carlo on saved strategies
    python run_pipeline.py --stage validate   # Walk-forward + regime analysis
    python run_pipeline.py --stage paper      # Paper trading replay
    python run_pipeline.py --fetch            # Fetch fresh data first
    python run_pipeline.py --max-strategies 100  # Limit strategy count
    python run_pipeline.py --mc-sims 5000     # MC simulation count
    python run_pipeline.py --top 10           # Keep top N strategies
"""

import logging
import sys
import time

from pipeline.config import CONFIG_DIR, setup_logging
from pipeline.cli import parse_args, banner
from pipeline.data_loader import load_data
from pipeline.search import stage_search, _load_saved_search_results
from pipeline.monte_carlo import stage_monte_carlo
from pipeline.validation import stage_validate
from pipeline.paper_trading import stage_paper_trade
from pipeline.reporting import save_final_leaderboard, print_final_summary

setup_logging()
logger = logging.getLogger("mcq_pipeline")


def main():
    args = parse_args()
    banner()

    pipeline_start = time.time()
    stage_timings: dict[str, float] = {}

    # ── Load configuration ─────────────────────────────────────────
    from engine.utils import (
        load_prop_firm_rules,
        load_session_config,
        load_events_calendar,
    )

    try:
        prop_rules = load_prop_firm_rules(CONFIG_DIR, args.firm)
        session_config = load_session_config(CONFIG_DIR)
        events_calendar = load_events_calendar(CONFIG_DIR)
    except Exception as e:
        logger.error("Failed to load configuration from %s: %s", CONFIG_DIR, e)
        sys.exit(1)

    logger.info("Loaded config: firm=%s, symbol=%s", prop_rules.firm_name, args.symbol)
    logger.info(
        "  Account: $%s | Daily limit: $%s | Max DD: $%s (%s)",
        f"{prop_rules.account_size:,.0f}",
        f"{prop_rules.daily_loss_limit:,.0f}",
        f"{prop_rules.max_drawdown:,.0f}",
        prop_rules.drawdown_type,
    )

    # ── Load data ──────────────────────────────────────────────────
    t0 = time.time()
    data = load_data(args)
    stage_timings["Data Loading"] = time.time() - t0

    if not data or all(df.is_empty() for df in data.values()):
        logger.error("No data available. Cannot proceed.")
        sys.exit(1)

    # ── Determine which stages to run ──────────────────────────────
    run_search = args.stage in ("search", "all")
    run_mc = args.stage in ("mc", "all")
    run_validate = args.stage in ("validate", "all")
    run_paper = args.stage in ("paper", "all")

    # Intermediate results that carry between stages
    top_strategies = []
    mc_survivors = []
    validated_strategies = []

    # ── Stage 1: Strategy Search ───────────────────────────────────
    if run_search:
        t0 = time.time()
        top_strategies = stage_search(
            args, data, prop_rules, session_config, events_calendar,
        )
        stage_timings["Strategy Search"] = time.time() - t0

    # ── Stage 2: Monte Carlo ───────────────────────────────────────
    if run_mc:
        if not top_strategies and not run_search:
            top_strategies = _load_saved_search_results()

        if top_strategies:
            t0 = time.time()
            mc_survivors = stage_monte_carlo(args, top_strategies, prop_rules)
            stage_timings["Monte Carlo"] = time.time() - t0
        else:
            logger.warning("No strategies available for Monte Carlo. Run --stage search first.")

    # ── Stage 3: Validation ────────────────────────────────────────
    if run_validate:
        if not mc_survivors and not run_mc:
            top_strategies = _load_saved_search_results()
            if top_strategies:
                t0 = time.time()
                mc_survivors = stage_monte_carlo(args, top_strategies, prop_rules)
                stage_timings["Monte Carlo"] = time.time() - t0

        if mc_survivors:
            t0 = time.time()
            validated_strategies = stage_validate(
                args, mc_survivors, data, prop_rules, session_config, events_calendar,
            )
            stage_timings["Validation"] = time.time() - t0
        else:
            logger.warning("No MC survivors for validation. Run earlier stages first.")

    # ── Stage 4: Paper Trading ─────────────────────────────────────
    if run_paper:
        if not validated_strategies and mc_survivors:
            validated_strategies = [
                (sd, {"strategy_name": mcr.strategy_name, "mc_score": score, "mc_result": mcr})
                for sd, mcr, score in mc_survivors
            ]

        if validated_strategies:
            t0 = time.time()
            stage_paper_trade(
                args, validated_strategies, data, prop_rules, session_config, events_calendar,
            )
            stage_timings["Paper Trading"] = time.time() - t0
        else:
            logger.warning("No validated strategies for paper trading. Run earlier stages first.")

    # ── Save final leaderboard ─────────────────────────────────────
    if validated_strategies:
        save_final_leaderboard(validated_strategies, args.output_dir)

    # ── Final summary ──────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    print_final_summary(stage_timings, total_time)


if __name__ == "__main__":
    main()
