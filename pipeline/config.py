"""Path constants, helpers, and shared utilities for the pipeline."""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger("mcq_pipeline")

# ── Project paths ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
STRATEGIES_DIR = PROJECT_ROOT / "strategies" / "saved"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure key directories exist
for _d in (RESULTS_DIR, STRATEGIES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Stage formatting ─────────────────────────────────────────────────

def stage_header(name: str):
    """Print a prominent stage header."""
    width = 62
    print()
    print("=" * width)
    print(f"  STAGE: {name}")
    print("=" * width)
    print()


def stage_footer(name: str, elapsed: float, summary: str = ""):
    """Print stage completion footer."""
    print()
    print(f"  [{name}] completed in {elapsed:.1f}s{' -- ' + summary if summary else ''}")
    print("-" * 62)


# ── Helpers ──────────────────────────────────────────────────────────

def _get_contract_spec(symbol: str):
    """Return the correct ContractSpec for a symbol."""
    from engine.utils import MNQ_SPEC, NQ_SPEC
    return MNQ_SPEC if symbol.upper() == "MNQ" else NQ_SPEC


def _build_risk_manager(prop_rules, session_config, events_calendar, contract_spec):
    """Build a RiskManager from configs."""
    from engine.risk_manager import RiskManager
    return RiskManager(
        prop_rules=prop_rules,
        session_config=session_config,
        events_calendar=events_calendar,
        contract_spec=contract_spec,
    )


def _build_backtest_config(args, data: dict[str, pl.DataFrame]):
    """Build a BacktestConfig with date range from the data."""
    from engine.utils import BacktestConfig
    primary_tf = "1m" if "1m" in data else sorted(data.keys())[0]
    df = data[primary_tf]
    start = str(df["timestamp"][0])[:10]
    end = str(df["timestamp"][-1])[:10]
    return BacktestConfig(
        symbol=args.symbol,
        prop_firm_profile=args.firm,
        start_date=start,
        end_date=end,
        slippage_ticks=2,
        initial_capital=50000.0,
    )


def _save_json(data: dict, filepath: Path):
    """Save a dict as JSON with numpy-safe serializer."""

    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return str(obj)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2, default=_default))
    logger.info("Saved %s", filepath)


def setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
