"""CLI argument parsing and banner for the MCQ Engine pipeline."""

import argparse

from pipeline.config import REPORTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="MCQ Engine Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
""",
    )
    parser.add_argument(
        "--stage",
        choices=["search", "mc", "validate", "paper", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch fresh data from Yahoo Finance before running",
    )
    parser.add_argument("--symbol", default="MNQ", help="Symbol to trade (MNQ or NQ)")
    parser.add_argument("--firm", default="topstep_50k", help="Prop firm profile name")
    parser.add_argument(
        "--max-strategies", type=int, default=200,
        help="Max strategies to generate in search stage",
    )
    parser.add_argument(
        "--max-entry-signals", type=int, default=2,
        help="Max entry signals per strategy combination",
    )
    parser.add_argument(
        "--max-filters", type=int, default=1,
        help="Max filters per strategy combination",
    )
    parser.add_argument(
        "--mc-sims", type=int, default=5000,
        help="Monte Carlo simulations per strategy",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Keep top N strategies after each filtering stage",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--output-dir", default=str(REPORTS_DIR),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--paper-speed", type=float, default=0.0,
        help="Paper trading replay speed in seconds (0 = as fast as possible)",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run backtests sequentially instead of in parallel (useful for debugging)",
    )
    return parser.parse_args()


def banner():
    print(r"""
 _____ _____ _____    _____         _
|     |     |     |  |   __|___ ___|_|___ ___
| | | |   --|  |  |  |   __|   | . | |   | -_|
|_|_|_|_____|__  _|  |_____|_|_|_  |_|_|_|___|
               |__|            |___|

+==============================================================+
|                    MCQ ENGINE PIPELINE                        |
|         Monte Carlo Quantitative Trading Engine              |
+--------------------------------------------------------------+
|  Phase 1: Data Pipeline       [==========]  Complete         |
|  Phase 2: Strategy Generation [==========]  Complete         |
|  Phase 3: Monte Carlo         [==========]  Complete         |
|  Phase 4: Validation          [==========]  Complete         |
|  Phase 5: Paper Trading       [==========]  Complete         |
+==============================================================+
""")
