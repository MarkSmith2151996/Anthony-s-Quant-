"""Stage 3 — Walk-Forward Validation + Regime Analysis."""

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

from pipeline.config import (
    REPORTS_DIR,
    _get_contract_spec,
    _build_backtest_config,
    _build_risk_manager,
    _save_json,
    stage_header,
    stage_footer,
)

logger = logging.getLogger("mcq_pipeline")


def stage_validate(args, mc_survivors, data, prop_rules, session_config, events_calendar) -> list:
    """
    Walk-forward validation, regime analysis, and overfitting detection.

    Args:
        mc_survivors: list of (strategy_dict, MCResult, StrategyScore) from MC stage.

    Returns list of validated (strategy_dict, results_dict) tuples.
    """
    stage_header("WALK-FORWARD VALIDATION + REGIME ANALYSIS")
    t0 = time.time()

    from strategies.generator import GeneratedStrategy
    from validation.walk_forward import WalkForwardValidator
    from validation.regime import RegimeDetector
    from validation.correlation import CorrelationAnalyzer, OverfitDetector
    from engine.utils import BacktestConfig

    contract_spec = _get_contract_spec(args.symbol)
    bt_config = _build_backtest_config(args, data)
    rm = _build_risk_manager(prop_rules, session_config, events_calendar, contract_spec)

    wf_validator = WalkForwardValidator(
        data=data,
        risk_manager=rm,
        contract_spec=contract_spec,
        config=bt_config,
        account_size=prop_rules.account_size,
    )

    regime_detector = RegimeDetector()
    overfit_detector = OverfitDetector(account_size=prop_rules.account_size)

    validated = []
    total = len(mc_survivors)

    for i, (strategy_dict, mc_result, mc_score) in enumerate(mc_survivors, 1):
        name = mc_result.strategy_name
        logger.info("[%d/%d] Validating '%s' ...", i, total, name)

        results = {
            "strategy_name": name,
            "mc_score": mc_score,
            "mc_result": mc_result,
        }

        # Reconstruct strategy object
        try:
            strategy = GeneratedStrategy.from_dict(strategy_dict)
        except Exception:
            logger.warning("  Could not reconstruct strategy '%s'; skipping.", name)
            continue

        # ── Walk-Forward ───────────────────────────────────────────
        try:
            wf_result = wf_validator.validate(
                strategy,
                train_days=40,
                test_days=15,
                step_days=15,
            )
            wf_validator.print_report(wf_result)
            results["wf_result"] = wf_result
            results["wf_efficiency"] = wf_result.wf_efficiency
            results["is_overfit_wf"] = wf_result.is_overfit
            results["pct_profitable_windows"] = wf_result.pct_profitable_windows
        except Exception:
            logger.error("  Walk-forward failed for '%s':", name)
            logger.error("  %s", traceback.format_exc().split("\n")[-2])
            results["wf_result"] = None
            results["wf_efficiency"] = 0.0
            results["is_overfit_wf"] = True

        # ── Regime Analysis ────────────────────────────────────────
        primary_tf = "1m" if "1m" in data else sorted(data.keys())[0]
        df_primary = data[primary_tf]

        try:
            # We need trades from the full backtest for regime tagging.
            # Re-run a quick backtest to get trades (the WF result has
            # fragmented trades across windows).
            from engine.backtester import VectorizedBacktester
            bt = VectorizedBacktester(
                data=data,
                risk_manager=_build_risk_manager(
                    prop_rules, session_config, events_calendar, contract_spec
                ),
                contract_spec=contract_spec,
                config=bt_config,
            )
            full_bt = bt.run(strategy)
            full_trades = full_bt.trades

            regime_analysis = regime_detector.analyze_strategy(
                full_trades, df_primary, strategy_name=name,
            )
            regime_detector.print_analysis(regime_analysis)
            results["regime_analysis"] = regime_analysis
            results["regime_sensitivity"] = regime_analysis.regime_sensitivity
        except Exception:
            logger.error("  Regime analysis failed for '%s':", name)
            logger.error("  %s", traceback.format_exc().split("\n")[-2])
            results["regime_analysis"] = None
            results["regime_sensitivity"] = 1.0
            full_trades = []

        # ── Overfitting Detection ──────────────────────────────────
        try:
            wf_res = results.get("wf_result")
            overfit_report = overfit_detector.analyze(
                strategy_name=name,
                trades=full_trades,
                wf_result=wf_res,
            )
            overfit_detector.print_report(overfit_report)
            results["overfit_report"] = overfit_report
            results["overfit_verdict"] = overfit_report.verdict
        except Exception:
            logger.error("  Overfit detection failed for '%s':", name)
            logger.error("  %s", traceback.format_exc().split("\n")[-2])
            results["overfit_report"] = None
            results["overfit_verdict"] = "unknown"

        validated.append((strategy_dict, results))

    # ── Correlation Analysis (portfolio-level) ─────────────────────
    if len(validated) >= 2:
        print()
        print("  Running portfolio correlation analysis...")
        try:
            correlation_analyzer = CorrelationAnalyzer(account_size=prop_rules.account_size)
            strategy_trade_pairs = []
            for sd, res in validated:
                # Use WF OOS trades if available, else empty
                wf_res = res.get("wf_result")
                if wf_res and wf_res.oos_trades:
                    strategy_trade_pairs.append((res["strategy_name"], wf_res.oos_trades))
            if len(strategy_trade_pairs) >= 2:
                portfolio = correlation_analyzer.analyze_portfolio(strategy_trade_pairs)
                correlation_analyzer.print_portfolio_report(portfolio)
        except Exception:
            logger.warning("Portfolio correlation analysis failed")

    # ── Filter overfit strategies ──────────────────────────────────
    surviving = []
    rejected_count = 0
    for sd, res in validated:
        verdict = res.get("overfit_verdict", "unknown")
        is_overfit_wf = res.get("is_overfit_wf", False)

        # Reject strategies that are "likely_overfit" AND failed WF
        if verdict == "likely_overfit" and is_overfit_wf:
            logger.info(
                "  REJECTED '%s': likely overfit (verdict=%s, WF overfit=%s)",
                res["strategy_name"], verdict, is_overfit_wf,
            )
            rejected_count += 1
            continue

        surviving.append((sd, res))

    # ── Save validation summary ────────────────────────────────────
    val_summary = {
        "timestamp": datetime.now().isoformat(),
        "strategies_validated": len(validated),
        "strategies_rejected": rejected_count,
        "strategies_surviving": len(surviving),
        "strategies": [],
    }
    for sd, res in surviving:
        entry = {
            "name": res["strategy_name"],
            "wf_efficiency": round(res.get("wf_efficiency", 0), 3),
            "pct_profitable_windows": round(res.get("pct_profitable_windows", 0), 1),
            "regime_sensitivity": round(res.get("regime_sensitivity", 1.0), 3),
            "overfit_verdict": res.get("overfit_verdict", "unknown"),
            "mc_composite_score": round(res["mc_score"].composite_score, 2),
            "mc_grade": res["mc_score"].grade,
        }
        val_summary["strategies"].append(entry)
    _save_json(val_summary, REPORTS_DIR / "validation_results.json")

    elapsed = time.time() - t0
    stage_footer(
        "VALIDATION", elapsed,
        f"{len(validated)} validated, {rejected_count} rejected, {len(surviving)} surviving",
    )
    return surviving
