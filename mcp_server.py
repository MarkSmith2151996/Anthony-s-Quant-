"""MCQ Engine MCP Server — exposes the strategy research engine as MCP tools.

Start: python3 mcp_server.py
Claude Code discovers this via .claude/settings.json.
"""

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging to file (stdout/stderr reserved for MCP JSON-RPC) ────────
_log_file = PROJECT_ROOT / "mcp_server.log"
logging.basicConfig(
    filename=str(_log_file),
    level=logging.INFO,
    format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mcp_server")
logger.info("MCQ Engine MCP Server starting up")

# ── Engine imports ────────────────────────────────────────────────────
from engine.utils import (
    CONTRACT_SPECS,
    BacktestConfig,
    BacktestResult,
    ContractSpec,
    PerformanceMetrics,
    PropFirmRules,
    Trade,
    load_events_calendar,
    load_prop_firm_rules,
    load_session_config,
    load_yaml,
)
from engine.data import DataCatalog, get_provider, read_bars, write_bars
from engine.backtester import VectorizedBacktester
from engine.risk_manager import RiskManager
from engine.metrics import calculate_metrics
from signals.registry import SignalRegistry
from strategies.generator import (
    ExitRules,
    GeneratedStrategy,
    SizingRules,
    StrategyGenerator,
)
from monte_carlo.simulator import MCConfig, MCResult, MonteCarloSimulator
from validation.walk_forward import WalkForwardValidator, WFResult

# ── MCP framework ─────────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcq-engine")

CONFIG_DIR = PROJECT_ROOT / "config"


# ── Engine State (persists between tool calls) ────────────────────────


@dataclass
class EngineState:
    """In-memory state shared across all tool calls."""

    data: dict[str, dict[str, Any]] = field(default_factory=dict)
    # {symbol: {interval: pl.DataFrame}}
    strategies: list[GeneratedStrategy] = field(default_factory=list)
    backtest_results: dict[int, BacktestResult] = field(default_factory=dict)
    mc_results: dict[int, MCResult] = field(default_factory=dict)
    validation_results: dict[int, Any] = field(default_factory=dict)


state = EngineState()


# ── Serialization helpers ─────────────────────────────────────────────


def _safe_float(v: Any) -> float:
    """Convert to float, capping inf/nan."""
    if v is None:
        return 0.0
    f = float(v)
    if math.isinf(f):
        return 999.99 if f > 0 else -999.99
    if math.isnan(f):
        return 0.0
    return round(f, 4)


def _serialize_metrics(m: PerformanceMetrics) -> dict:
    """Convert PerformanceMetrics to a JSON-safe dict."""
    return {
        "total_trades": m.total_trades,
        "winning_trades": m.winning_trades,
        "losing_trades": m.losing_trades,
        "win_rate": round(m.win_rate, 2),
        "total_pnl": round(m.total_pnl, 2),
        "gross_profit": round(m.gross_profit, 2),
        "gross_loss": round(m.gross_loss, 2),
        "profit_factor": _safe_float(m.profit_factor),
        "sharpe_ratio": _safe_float(m.sharpe_ratio),
        "max_drawdown": round(m.max_drawdown, 2),
        "max_drawdown_pct": round(m.max_drawdown_pct, 2),
        "avg_trade_pnl": round(m.avg_trade_pnl, 2),
        "avg_winner": round(m.avg_winner, 2),
        "avg_loser": round(m.avg_loser, 2),
        "largest_winner": round(m.largest_winner, 2),
        "largest_loser": round(m.largest_loser, 2),
        "avg_hold_time_minutes": round(m.avg_hold_time_seconds / 60, 1),
        "profit_by_session": {
            k: round(v, 2) for k, v in m.profit_by_session.items()
        },
        "consistency_score": round(m.consistency_score, 2),
    }


def _serialize_trade(t: Trade) -> dict:
    """Convert Trade to a JSON-safe dict."""
    return {
        "trade_id": t.trade_id,
        "symbol": t.symbol,
        "direction": t.direction,
        "entry_time": t.entry_time.isoformat() if isinstance(t.entry_time, datetime) else str(t.entry_time),
        "entry_price": round(t.entry_price, 4),
        "exit_time": t.exit_time.isoformat() if isinstance(t.exit_time, datetime) else str(t.exit_time),
        "exit_price": round(t.exit_price, 4),
        "contracts": t.contracts,
        "gross_pnl": round(t.gross_pnl, 2),
        "commission": round(t.commission, 2),
        "net_pnl": round(t.net_pnl, 2),
        "duration_minutes": round(t.duration_seconds / 60, 1),
        "session_segment": t.session_segment,
        "exit_reason": t.exit_reason,
    }


def _serialize_mc_result(r: MCResult) -> dict:
    """Convert MCResult to a JSON-safe dict (no numpy arrays)."""
    return {
        "strategy_name": r.strategy_name,
        "n_simulations": r.n_simulations,
        "median_return": round(r.median_return, 2),
        "mean_return": round(r.mean_return, 2),
        "pct_5th_return": round(r.pct_5th_return, 2),
        "pct_95th_return": round(r.pct_95th_return, 2),
        "probability_of_profit": round(r.probability_of_profit * 100, 1),
        "probability_of_ruin": round(r.probability_of_ruin * 100, 1),
        "prop_firm_pass_rate": round(r.prop_firm_pass_rate * 100, 1),
        "composite_score": round(r.composite_score, 2),
        "median_sharpe": round(float(np.median(r.sharpe_ratios)), 2),
        "median_profit_factor": _safe_float(np.median(r.profit_factors)),
        "median_win_rate": round(float(np.median(r.win_rates)), 2),
        "median_max_drawdown": round(float(np.median(r.max_drawdowns)), 2),
    }


def _serialize_wf_result(r: WFResult) -> dict:
    """Convert WFResult to a JSON-safe dict."""
    windows = []
    for w in r.windows:
        win = {
            "window_id": w.window_id,
            "train_period": f"{w.train_start.strftime('%Y-%m-%d')} to {w.train_end.strftime('%Y-%m-%d')}",
            "test_period": f"{w.test_start.strftime('%Y-%m-%d')} to {w.test_end.strftime('%Y-%m-%d')}",
        }
        if w.train_metrics:
            win["is_sharpe"] = _safe_float(w.train_metrics.sharpe_ratio)
            win["is_trades"] = w.train_metrics.total_trades
            win["is_pnl"] = round(w.train_metrics.total_pnl, 2)
        if w.test_metrics:
            win["oos_sharpe"] = _safe_float(w.test_metrics.sharpe_ratio)
            win["oos_trades"] = w.test_metrics.total_trades
            win["oos_pnl"] = round(w.test_metrics.total_pnl, 2)
        windows.append(win)

    result: dict[str, Any] = {
        "strategy_name": r.strategy_name,
        "n_windows": len(r.windows),
        "windows": windows,
        "wf_efficiency": _safe_float(r.wf_efficiency),
        "wf_profit_ratio": _safe_float(r.wf_profit_ratio),
        "pct_profitable_windows": round(r.pct_profitable_windows, 1),
        "sharpe_stability": _safe_float(r.sharpe_stability),
        "is_overfit": r.is_overfit,
        "overfit_reasons": r.overfit_reasons,
    }
    if r.oos_metrics:
        result["oos_aggregate"] = _serialize_metrics(r.oos_metrics)
    return result


# ── Helper: build risk manager from config ────────────────────────────


def _build_risk_manager(
    symbol: str, firm_profile: str = "topstep_50k"
) -> tuple[RiskManager, PropFirmRules, ContractSpec]:
    """Build a RiskManager from config files."""
    prop_rules = load_prop_firm_rules(CONFIG_DIR, firm_profile)
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    contract_spec = CONTRACT_SPECS.get(symbol)
    if contract_spec is None:
        raise ValueError(
            f"Unknown symbol {symbol!r}. Available: {list(CONTRACT_SPECS.keys())}"
        )
    rm = RiskManager(prop_rules, session_config, events_calendar, contract_spec)
    return rm, prop_rules, contract_spec


# ═══════════════════════════════════════════════════════════════════════
# DATA TOOLS (4)
# ═══════════════════════════════════════════════════════════════════════


@mcp.tool()
def list_instruments() -> dict:
    """List all configured contract specs (MNQ, NQ, ES, etc.) with their tick sizes, values, and point values."""
    instruments = {}
    for sym, spec in CONTRACT_SPECS.items():
        instruments[sym] = {
            "symbol": spec.symbol,
            "tick_size": spec.tick_size,
            "tick_value": spec.tick_value,
            "point_value": spec.point_value,
        }
    return {"instruments": instruments, "count": len(instruments)}


@mcp.tool()
def list_available_data() -> dict:
    """Show all cached Parquet data — symbols, intervals, bar counts, and date ranges."""
    catalog = DataCatalog()
    try:
        summary = catalog.summary()
    except Exception as e:
        logger.exception("list_available_data failed")
        return {"error": str(e), "data": []}
    return {"data": summary, "count": len(summary)}


@mcp.tool()
def load_data(
    symbol: str,
    interval: str = "1m",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Load Parquet data from disk into memory for backtesting.

    Args:
        symbol: Instrument symbol (e.g. "MNQ", "NQ", "ES").
        interval: Bar interval (e.g. "1m", "5m").
        start: Optional start date "YYYY-MM-DD".
        end: Optional end date "YYYY-MM-DD".
    """
    try:
        df = read_bars(symbol, interval, start=start, end=end)
    except FileNotFoundError as e:
        return {"error": str(e)}

    if symbol not in state.data:
        state.data[symbol] = {}
    state.data[symbol][interval] = df

    ts_col = df["timestamp"]
    return {
        "symbol": symbol,
        "interval": interval,
        "rows": len(df),
        "columns": df.columns,
        "date_range": {
            "start": str(ts_col[0]) if len(df) > 0 else None,
            "end": str(ts_col[-1]) if len(df) > 0 else None,
        },
    }


@mcp.tool()
def fetch_data(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    provider: str = "yahoo",
) -> dict:
    """Download data from a provider, cache to Parquet, and load into memory.

    Args:
        symbol: Instrument symbol.
        interval: Bar interval (e.g. "1m", "5m").
        start: Start date "YYYY-MM-DD".
        end: End date "YYYY-MM-DD".
        provider: Data provider name ("yahoo", "databento", "motivewave").
    """
    try:
        prov = get_provider(provider)
        df = prov.get_historical_bars(symbol, interval, start, end)
    except Exception as e:
        logger.exception("fetch_data failed for %s/%s", symbol, interval)
        return {"error": str(e)}

    # Cache to Parquet
    try:
        write_bars(df, symbol, interval)
    except Exception as e:
        logger.warning("Could not write Parquet: %s", e)

    # Load into memory
    if symbol not in state.data:
        state.data[symbol] = {}
    state.data[symbol][interval] = df

    ts_col = df["timestamp"]
    return {
        "symbol": symbol,
        "interval": interval,
        "provider": provider,
        "rows": len(df),
        "date_range": {
            "start": str(ts_col[0]) if len(df) > 0 else None,
            "end": str(ts_col[-1]) if len(df) > 0 else None,
        },
        "cached": True,
    }


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY TOOLS (4)
# ═══════════════════════════════════════════════════════════════════════


@mcp.tool()
def list_signals(
    category: str | None = None,
    signal_type: str | None = None,
) -> dict:
    """Show available signals with descriptions and parameter ranges.

    Args:
        category: Filter by category (trend, momentum, volatility, volume, orderflow, price_action, time_filter).
        signal_type: Filter by type (entry, filter, indicator).
    """
    registry = SignalRegistry()
    signals = registry.get_all()

    result = []
    for name, sig in signals.items():
        if category and sig.category != category:
            continue
        if signal_type and sig.signal_type != signal_type:
            continue
        result.append({
            "name": sig.name,
            "category": sig.category,
            "signal_type": sig.signal_type,
            "description": sig.description,
            "parameters": sig.parameters,
            "entry_columns": sig.entry_columns,
            "filter_columns": sig.filter_columns,
        })

    return {"signals": result, "count": len(result)}


@mcp.tool()
def count_strategy_combinations(
    max_entry_signals: int = 2,
    max_filters: int = 1,
) -> dict:
    """Count the strategy search space size without generating.

    Args:
        max_entry_signals: Max entry signals per strategy (1-3).
        max_filters: Max filter signals per strategy (0-3).
    """
    registry = SignalRegistry()
    generator = StrategyGenerator(registry)
    count = generator.count_combinations(
        max_entry_signals=max_entry_signals,
        max_filters=max_filters,
    )
    return {
        "total_combinations": count,
        "max_entry_signals": max_entry_signals,
        "max_filters": max_filters,
    }


@mcp.tool()
def generate_strategies(
    max_strategies: int = 50,
    max_entry_signals: int = 2,
    max_filters: int = 1,
) -> dict:
    """Generate strategy combinations from the signal library and store them in memory.

    Args:
        max_strategies: Maximum strategies to generate.
        max_entry_signals: Max entry signals per strategy.
        max_filters: Max filter signals per strategy.
    """
    registry = SignalRegistry()
    generator = StrategyGenerator(registry)

    strategies = generator.generate(
        max_entry_signals=max_entry_signals,
        max_filters=max_filters,
        max_strategies=max_strategies,
    )

    state.strategies = strategies
    logger.info("Generated %d strategies", len(strategies))

    summaries = []
    for i, s in enumerate(strategies):
        summaries.append({
            "index": i,
            "name": s.name,
            "entries": [e["signal_name"] for e in s.entry_signals],
            "filters": [f["signal_name"] for f in s.entry_filters],
            "stop_loss": f"{s.exit_rules.stop_loss_type}:{s.exit_rules.stop_loss_value}",
            "take_profit": f"{s.exit_rules.take_profit_type}:{s.exit_rules.take_profit_value}",
        })

    return {"strategies": summaries, "count": len(strategies)}


@mcp.tool()
def get_strategy(index: int) -> dict:
    """View a strategy's full definition by index.

    Args:
        index: Strategy index from generate_strategies output.
    """
    if not state.strategies:
        return {"error": "No strategies generated yet. Run generate_strategies first."}
    if index < 0 or index >= len(state.strategies):
        return {"error": f"Index {index} out of range (0-{len(state.strategies) - 1})."}

    return {"strategy": state.strategies[index].to_dict(), "index": index}


# ═══════════════════════════════════════════════════════════════════════
# EXECUTION TOOLS (3)
# ═══════════════════════════════════════════════════════════════════════


@mcp.tool()
def run_backtest(
    strategy_index: int,
    symbol: str = "MNQ",
    interval: str = "1m",
    firm: str = "topstep_50k",
) -> dict:
    """Backtest a strategy and return performance metrics.

    Args:
        strategy_index: Index of strategy to backtest.
        symbol: Instrument symbol.
        interval: Bar interval for primary timeframe.
        firm: Prop firm profile name.
    """
    # Validate strategy
    if not state.strategies:
        return {"error": "No strategies generated. Run generate_strategies first."}
    if strategy_index < 0 or strategy_index >= len(state.strategies):
        return {"error": f"Index {strategy_index} out of range (0-{len(state.strategies) - 1})."}

    # Validate data
    if symbol not in state.data or interval not in state.data[symbol]:
        return {
            "error": f"No data loaded for {symbol}/{interval}. Run load_data first.",
            "loaded": {s: list(intervals.keys()) for s, intervals in state.data.items()},
        }

    strategy = state.strategies[strategy_index]
    df = state.data[symbol][interval]

    try:
        rm, prop_rules, contract_spec = _build_risk_manager(symbol, firm)
    except Exception as e:
        return {"error": f"Config error: {e}"}

    # Determine date range from data
    ts = df["timestamp"]
    bt_config = BacktestConfig(
        symbol=symbol,
        prop_firm_profile=firm,
        start_date=str(ts[0])[:10],
        end_date=str(ts[-1])[:10],
    )

    try:
        bt = VectorizedBacktester(
            data={interval: df},
            risk_manager=rm,
            contract_spec=contract_spec,
            config=bt_config,
        )
        result = bt.run(strategy)
    except Exception as e:
        logger.exception("Backtest failed for strategy %d", strategy_index)
        return {"error": f"Backtest failed: {e}"}

    # Calculate metrics
    metrics = calculate_metrics(result.trades, prop_rules.account_size)
    result.metrics = metrics

    # Store result
    state.backtest_results[strategy_index] = result

    return {
        "strategy_index": strategy_index,
        "strategy_name": strategy.name,
        "symbol": symbol,
        "firm": firm,
        "metrics": _serialize_metrics(metrics),
        "trade_count": len(result.trades),
        "sample_trades": [_serialize_trade(t) for t in result.trades[:5]],
    }


@mcp.tool()
def run_monte_carlo(
    strategy_index: int,
    n_simulations: int = 1000,
    firm: str = "topstep_50k",
) -> dict:
    """Run Monte Carlo stress test on a backtested strategy's trades.

    Args:
        strategy_index: Index of strategy (must have been backtested first).
        n_simulations: Number of MC simulations to run.
        firm: Prop firm profile for compliance checks.
    """
    if strategy_index not in state.backtest_results:
        return {
            "error": f"No backtest result for strategy {strategy_index}. Run run_backtest first.",
            "backtested": list(state.backtest_results.keys()),
        }

    bt_result = state.backtest_results[strategy_index]
    trades = bt_result.trades
    if not trades:
        return {"error": "No trades from backtest — nothing to simulate."}

    # Build MC config
    try:
        prop_rules = load_prop_firm_rules(CONFIG_DIR, firm)
    except Exception as e:
        return {"error": f"Config error: {e}"}

    symbol = bt_result.config.symbol
    spec = CONTRACT_SPECS.get(symbol)
    tick_value = spec.tick_value if spec else 0.50

    mc_config = MCConfig(
        n_simulations=n_simulations,
        initial_capital=prop_rules.account_size,
        tick_value=tick_value,
        prop_firm_rules=prop_rules,
        n_workers=1,  # sequential to avoid multiprocessing deadlocks in stdio
    )

    try:
        simulator = MonteCarloSimulator(mc_config)
        strategy_name = state.strategies[strategy_index].name if strategy_index < len(state.strategies) else "unknown"
        mc_result = simulator.run(trades, strategy_name=strategy_name)
    except Exception as e:
        logger.exception("MC simulation failed for strategy %d", strategy_index)
        return {"error": f"MC simulation failed: {e}"}

    state.mc_results[strategy_index] = mc_result

    return {
        "strategy_index": strategy_index,
        **_serialize_mc_result(mc_result),
    }


@mcp.tool()
def run_validation(
    strategy_index: int,
    symbol: str = "MNQ",
    interval: str = "1m",
    firm: str = "topstep_50k",
    train_days: int = 60,
    test_days: int = 20,
    step_days: int = 20,
) -> dict:
    """Run walk-forward validation to check for overfitting.

    Args:
        strategy_index: Index of strategy to validate.
        symbol: Instrument symbol.
        interval: Bar interval.
        firm: Prop firm profile.
        train_days: Training window size in trading days.
        test_days: Testing window size in trading days.
        step_days: Step between windows in trading days.
    """
    if not state.strategies:
        return {"error": "No strategies generated. Run generate_strategies first."}
    if strategy_index < 0 or strategy_index >= len(state.strategies):
        return {"error": f"Index {strategy_index} out of range (0-{len(state.strategies) - 1})."}

    if symbol not in state.data or interval not in state.data[symbol]:
        return {"error": f"No data loaded for {symbol}/{interval}. Run load_data first."}

    strategy = state.strategies[strategy_index]
    df = state.data[symbol][interval]

    try:
        rm, prop_rules, contract_spec = _build_risk_manager(symbol, firm)
    except Exception as e:
        return {"error": f"Config error: {e}"}

    ts = df["timestamp"]
    bt_config = BacktestConfig(
        symbol=symbol,
        prop_firm_profile=firm,
        start_date=str(ts[0])[:10],
        end_date=str(ts[-1])[:10],
    )

    try:
        validator = WalkForwardValidator(
            data={interval: df},
            risk_manager=rm,
            contract_spec=contract_spec,
            config=bt_config,
            account_size=prop_rules.account_size,
        )
        wf_result = validator.validate(
            strategy=strategy,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
        )
    except Exception as e:
        logger.exception("Walk-forward validation failed for strategy %d", strategy_index)
        return {"error": f"Validation failed: {e}"}

    state.validation_results[strategy_index] = wf_result

    return {
        "strategy_index": strategy_index,
        **_serialize_wf_result(wf_result),
    }


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE + UTILITY TOOLS (4)
# ═══════════════════════════════════════════════════════════════════════


@mcp.tool()
def run_search(
    symbol: str = "MNQ",
    interval: str = "1m",
    firm: str = "topstep_50k",
    max_strategies: int = 50,
    max_entry_signals: int = 2,
    max_filters: int = 1,
) -> dict:
    """Full search: generate strategies, backtest all, return ranked leaderboard.

    Args:
        symbol: Instrument symbol.
        interval: Bar interval.
        firm: Prop firm profile.
        max_strategies: Max strategies to generate and test.
        max_entry_signals: Max entry signals per strategy.
        max_filters: Max filters per strategy.
    """
    # Check data
    if symbol not in state.data or interval not in state.data[symbol]:
        return {"error": f"No data loaded for {symbol}/{interval}. Run load_data first."}

    df = state.data[symbol][interval]

    # Generate strategies
    registry = SignalRegistry()
    generator = StrategyGenerator(registry)
    strategies = generator.generate(
        max_entry_signals=max_entry_signals,
        max_filters=max_filters,
        max_strategies=max_strategies,
    )
    state.strategies = strategies
    logger.info("Search: generated %d strategies", len(strategies))

    # Build backtest infrastructure
    try:
        rm, prop_rules, contract_spec = _build_risk_manager(symbol, firm)
    except Exception as e:
        return {"error": f"Config error: {e}"}

    ts = df["timestamp"]
    bt_config = BacktestConfig(
        symbol=symbol,
        prop_firm_profile=firm,
        start_date=str(ts[0])[:10],
        end_date=str(ts[-1])[:10],
    )

    # Backtest each strategy sequentially
    results: list[dict] = []
    for i, strategy in enumerate(strategies):
        try:
            bt = VectorizedBacktester(
                data={interval: df},
                risk_manager=rm,
                contract_spec=contract_spec,
                config=bt_config,
            )
            bt_result = bt.run(strategy)
            metrics = calculate_metrics(bt_result.trades, prop_rules.account_size)
            bt_result.metrics = metrics
            state.backtest_results[i] = bt_result

            results.append({
                "index": i,
                "name": strategy.name,
                "total_trades": metrics.total_trades,
                "total_pnl": round(metrics.total_pnl, 2),
                "sharpe_ratio": _safe_float(metrics.sharpe_ratio),
                "profit_factor": _safe_float(metrics.profit_factor),
                "win_rate": round(metrics.win_rate, 2),
                "max_drawdown": round(metrics.max_drawdown, 2),
            })
        except Exception as e:
            logger.warning("Strategy %d failed: %s", i, e)
            results.append({
                "index": i,
                "name": strategy.name,
                "error": str(e),
            })

    # Sort by Sharpe ratio (descending), filter out errors
    valid = [r for r in results if "error" not in r and r["total_trades"] > 0]
    valid.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    return {
        "symbol": symbol,
        "firm": firm,
        "strategies_tested": len(strategies),
        "strategies_with_trades": len(valid),
        "leaderboard": valid[:20],
    }


@mcp.tool()
def get_results(top_n: int = 10) -> dict:
    """Get a leaderboard of backtested strategies ranked by Sharpe ratio.

    Args:
        top_n: Number of top strategies to return.
    """
    if not state.backtest_results:
        return {"error": "No backtest results yet. Run run_backtest or run_search first."}

    entries = []
    for idx, bt in state.backtest_results.items():
        m = bt.metrics
        if m is None:
            continue
        entry: dict[str, Any] = {
            "index": idx,
            "name": bt.strategy_name,
            "total_trades": m.total_trades,
            "total_pnl": round(m.total_pnl, 2),
            "sharpe_ratio": _safe_float(m.sharpe_ratio),
            "profit_factor": _safe_float(m.profit_factor),
            "win_rate": round(m.win_rate, 2),
            "max_drawdown": round(m.max_drawdown, 2),
        }

        # Add MC results if available
        if idx in state.mc_results:
            mc = state.mc_results[idx]
            entry["mc_composite_score"] = round(mc.composite_score, 2)
            entry["mc_pass_rate"] = round(mc.prop_firm_pass_rate * 100, 1)

        # Add validation results if available
        if idx in state.validation_results:
            wf = state.validation_results[idx]
            entry["wf_efficiency"] = _safe_float(wf.wf_efficiency)
            entry["wf_overfit"] = wf.is_overfit

        entries.append(entry)

    # Sort by Sharpe descending
    entries.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    return {"leaderboard": entries[:top_n], "total_backtested": len(entries)}


@mcp.tool()
def get_prop_firms() -> dict:
    """List all available prop firm profiles with their rules."""
    try:
        data = load_yaml(CONFIG_DIR / "prop_firms.yaml")
    except Exception as e:
        return {"error": str(e)}

    firms = {}
    for profile, firm in data.items():
        firms[profile] = {
            "firm_name": firm["firm_name"],
            "account_size": firm["account_size"],
            "daily_loss_limit": firm["daily_loss_limit"],
            "max_drawdown": firm["max_drawdown"],
            "drawdown_type": firm["drawdown_type"],
            "max_contracts": firm["max_contracts"],
            "consistency_rule": firm["consistency_rule"],
            "commission_rt": firm["commission_per_contract_rt"],
            "exchange_fees_rt": firm["exchange_fees_per_contract_rt"],
        }

    return {"profiles": firms, "count": len(firms)}


@mcp.tool()
def engine_status() -> dict:
    """Get the current server state: loaded data, strategies, and results."""
    loaded_data = {}
    for sym, intervals in state.data.items():
        loaded_data[sym] = {}
        for iv, df in intervals.items():
            ts = df["timestamp"]
            loaded_data[sym][iv] = {
                "rows": len(df),
                "start": str(ts[0]) if len(df) > 0 else None,
                "end": str(ts[-1]) if len(df) > 0 else None,
            }

    return {
        "loaded_data": loaded_data,
        "strategies_generated": len(state.strategies),
        "backtest_results": len(state.backtest_results),
        "mc_results": len(state.mc_results),
        "validation_results": len(state.validation_results),
        "instruments": list(CONTRACT_SPECS.keys()),
    }


# ═══════════════════════════════════════════════════════════════════════
# PAPER TRADER TOOLS
# ═══════════════════════════════════════════════════════════════════════

PAPER_TRADER_STATE = PROJECT_ROOT / "paper_trader_state.json"
PAPER_TRADER_TRADES = PROJECT_ROOT / "reports" / "paper_trader" / "trades.csv"
PAPER_TRADER_LOG = PROJECT_ROOT / "reports" / "paper_trader" / "paper_trader.log"
PAPER_TRADER_HEALTH = PROJECT_ROOT / "paper_trader_health.json"


@mcp.tool()
def paper_trader_status() -> dict:
    """Get the live paper trader status: balance, P&L, positions, risk metrics."""
    if not PAPER_TRADER_STATE.exists():
        return {"error": "Paper trader not started. Run: python3 paper_trader_live.py --reset"}

    with open(PAPER_TRADER_STATE) as f:
        state = json.load(f)

    initial = state.get("initial_balance", 150000)
    balance = state.get("balance", initial)
    peak = state.get("peak_balance", initial)
    dd = balance - peak
    total_pnl = balance - initial

    # Check if process is running
    import subprocess
    result = subprocess.run(["pgrep", "-f", "paper_trader_live.py"],
                          capture_output=True, text=True)
    running = bool(result.stdout.strip())

    return {
        "running": running,
        "killed": state.get("killed", False),
        "balance": round(balance, 2),
        "initial_balance": initial,
        "total_pnl": round(total_pnl, 2),
        "pnl_pct": round(total_pnl / initial * 100, 2),
        "peak_balance": round(peak, 2),
        "drawdown": round(dd, 2),
        "daily_pnl": round(state.get("daily_pnl", 0), 2),
        "current_date": state.get("current_date"),
        "bars_processed": state.get("bar_count", 0),
        "contracts": state.get("contracts", 3),
        "open_positions": {
            k: {
                "direction": "LONG" if v["direction"] == 1 else "SHORT",
                "entry_px": v["entry_px"],
                "entry_time": v["entry_time"],
                "stop": v["stop_px"],
                "target": v["target_px"],
            }
            for k, v in state.get("positions", {}).items()
        },
        "daily_summary": state.get("daily_summary", {}),
        "last_save": state.get("last_save"),
    }


@mcp.tool()
def paper_trader_trades(last_n: int = 20) -> dict:
    """Get recent paper trading trades."""
    if not PAPER_TRADER_TRADES.exists():
        return {"trades": [], "message": "No trades recorded yet."}

    import csv
    with open(PAPER_TRADER_TRADES) as f:
        reader = csv.DictReader(f)
        trades = list(reader)

    recent = trades[-last_n:]
    total_pnl = sum(float(t["net_pnl"]) for t in trades)
    wins = sum(1 for t in trades if float(t["net_pnl"]) > 0)

    # Per-strategy summary
    by_strat = {}
    for t in trades:
        s = t["strategy"]
        if s not in by_strat:
            by_strat[s] = {"trades": 0, "pnl": 0, "wins": 0}
        by_strat[s]["trades"] += 1
        by_strat[s]["pnl"] += float(t["net_pnl"])
        if float(t["net_pnl"]) > 0:
            by_strat[s]["wins"] += 1

    for s in by_strat:
        by_strat[s]["pnl"] = round(by_strat[s]["pnl"], 2)
        by_strat[s]["win_rate"] = round(by_strat[s]["wins"] / by_strat[s]["trades"] * 100, 1)

    return {
        "total_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(wins / len(trades) * 100, 1) if trades else 0,
        "by_strategy": by_strat,
        "recent_trades": recent,
    }


@mcp.tool()
def paper_trader_log(lines: int = 30) -> dict:
    """Get recent paper trader log entries."""
    if not PAPER_TRADER_LOG.exists():
        return {"log": [], "message": "No log file yet."}

    with open(PAPER_TRADER_LOG) as f:
        all_lines = f.readlines()

    return {
        "total_lines": len(all_lines),
        "recent": [l.rstrip() for l in all_lines[-lines:]],
    }


@mcp.tool()
def paper_trader_performance() -> dict:
    """Get paper trader performance analytics: daily P&L, win streaks, prop firm check."""
    if not PAPER_TRADER_STATE.exists():
        return {"error": "Paper trader not started."}

    with open(PAPER_TRADER_STATE) as f:
        state = json.load(f)

    daily = state.get("daily_summary", {})
    if not daily:
        return {"message": "No completed trading days yet.", "trading_days": 0}

    pnls = [d["pnl"] for d in daily.values()]
    balances = [d["balance"] for d in daily.values()]
    initial = state.get("initial_balance", 150000)
    balance = state.get("balance", initial)

    # Streaks
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    streak = 0
    for p in pnls:
        if p > 0:
            streak = streak + 1 if streak > 0 else 1
            max_win_streak = max(max_win_streak, streak)
        elif p < 0:
            streak = streak - 1 if streak < 0 else -1
            max_loss_streak = min(max_loss_streak, streak)
        else:
            streak = 0
    current_streak = streak

    # Prop firm eval
    total_pnl = balance - initial
    worst_day = min(pnls) if pnls else 0
    dd = balance - state.get("peak_balance", initial)

    return {
        "trading_days": len(daily),
        "total_pnl": round(total_pnl, 2),
        "avg_daily_pnl": round(np.mean(pnls), 2) if pnls else 0,
        "best_day": round(max(pnls), 2) if pnls else 0,
        "worst_day": round(worst_day, 2),
        "profitable_days": sum(1 for p in pnls if p > 0),
        "losing_days": sum(1 for p in pnls if p < 0),
        "max_win_streak": max_win_streak,
        "max_loss_streak": abs(max_loss_streak),
        "current_streak": current_streak,
        "prop_firm": {
            "eval_target": 9000,
            "progress": round(total_pnl, 2),
            "progress_pct": round(total_pnl / 9000 * 100, 1),
            "daily_limit_ok": worst_day > -3000,
            "max_dd_ok": dd > -4500,
            "passed": total_pnl >= 9000 and worst_day > -3000 and dd > -4500,
        },
        "daily_detail": {d: round(v["pnl"], 2) for d, v in sorted(daily.items())},
    }


@mcp.tool()
def paper_trader_health() -> dict:
    """Check if the paper trader is alive. Shows crash count, restarts, uptime.
    If status is 'crashed' or 'dead', it needs to be restarted."""
    import subprocess

    health = {}
    if PAPER_TRADER_HEALTH.exists():
        with open(PAPER_TRADER_HEALTH) as f:
            health = json.load(f)

    # Check if process is actually running
    result = subprocess.run(["pgrep", "-f", "paper_trader_live.py"],
                          capture_output=True, text=True)
    process_running = bool(result.stdout.strip())

    # Check if supervisor is running
    result2 = subprocess.run(["pgrep", "-f", "pt_supervisor.sh"],
                           capture_output=True, text=True)
    supervisor_running = bool(result2.stdout.strip())

    # Determine actual status
    if process_running:
        actual_status = "running"
    elif supervisor_running:
        actual_status = "restarting"
    else:
        actual_status = "down"

    return {
        "process_running": process_running,
        "supervisor_running": supervisor_running,
        "actual_status": actual_status,
        "reported_status": health.get("status", "unknown"),
        "crash_count": health.get("crash_count", 0),
        "restarts": health.get("restarts", 0),
        "last_check": health.get("last_check"),
        "last_restart": health.get("last_restart"),
        "started_at": health.get("started_at"),
        "recent_crashes": health.get("crashes", [])[-5:],
        "action_needed": actual_status == "down",
        "restart_command": "cd /home/dev/projects/Anthony-s-Quant- && nohup ./pt_supervisor.sh > /dev/null 2>&1 &",
    }


@mcp.tool()
def paper_trader_restart() -> dict:
    """Restart the paper trader if it's down. Kills existing processes and starts fresh."""
    import subprocess

    # Kill any existing processes
    subprocess.run(["pkill", "-f", "paper_trader_live.py"], capture_output=True)
    subprocess.run(["pkill", "-f", "pt_supervisor.sh"], capture_output=True)

    import time
    time.sleep(2)

    # Start supervisor
    env = dict(os.environ)
    env["DATABENTO_API_KEY"] = env.get("DATABENTO_API_KEY", "")

    # Read API key from health file or bashrc if not in env
    if not env.get("DATABENTO_API_KEY"):
        bashrc = Path.home() / ".bashrc"
        if bashrc.exists():
            for line in bashrc.read_text().splitlines():
                if "DATABENTO_API_KEY" in line and "export" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        env["DATABENTO_API_KEY"] = parts[1].strip().strip('"').strip("'")

    proc = subprocess.Popen(
        ["bash", str(PROJECT_ROOT / "pt_supervisor.sh")],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    time.sleep(3)

    # Verify it started
    result = subprocess.run(["pgrep", "-f", "paper_trader_live.py"],
                          capture_output=True, text=True)
    running = bool(result.stdout.strip())

    return {
        "success": running,
        "supervisor_pid": proc.pid,
        "message": "Paper trader restarted successfully" if running else "Failed to start — check logs",
    }


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting MCP server on stdio (with paper trader tools)")
    mcp.run(transport="stdio")
