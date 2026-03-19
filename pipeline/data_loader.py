"""Stage 0 — Data Loading."""

import logging
import time

import polars as pl

from pipeline.config import (
    CONFIG_DIR,
    DATA_DIR,
    stage_header,
    stage_footer,
)

logger = logging.getLogger("mcq_pipeline")


def load_data(args) -> dict[str, pl.DataFrame]:
    """Load or fetch market data, returning {"1m": df, "5m": df}."""
    stage_header("DATA LOADING")
    t0 = time.time()

    if args.fetch:
        logger.info("Fetching fresh data from Yahoo Finance...")
        from engine.data_fetcher import YFinanceFetcher
        fetcher = YFinanceFetcher(output_dir=str(DATA_DIR))
        data = fetcher.fetch_multi_timeframe(symbol=args.symbol, days_back_1m=59)
        # Persist for reuse
        for tf, df in data.items():
            out_dir = DATA_DIR / args.symbol / tf
            out_dir.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_dir / "all.parquet")
        logger.info("Fresh data saved to %s", DATA_DIR / args.symbol)
    else:
        logger.info("Loading data from %s ...", DATA_DIR)
        from engine.data_pipeline import BarBuilder

        try:
            from engine.data import read_bars
            df_1m = read_bars(args.symbol, "1m", data_dir=DATA_DIR)
        except Exception:
            from engine.data_pipeline import load_parquet
            try:
                df_1m = load_parquet(DATA_DIR, args.symbol, "1m")
            except FileNotFoundError:
                # Fallback: try fetching from Yahoo Finance
                logger.warning(
                    "No local data found for %s/1m. Attempting Yahoo Finance fetch...",
                    args.symbol,
                )
                from engine.data_fetcher import YFinanceFetcher
                fetcher = YFinanceFetcher(output_dir=str(DATA_DIR))
                data = fetcher.fetch_multi_timeframe(symbol=args.symbol, days_back_1m=59)
                for tf, df in data.items():
                    out_dir = DATA_DIR / args.symbol / tf
                    out_dir.mkdir(parents=True, exist_ok=True)
                    df.write_parquet(out_dir / "all.parquet")
                elapsed = time.time() - t0
                _print_data_summary(data)
                stage_footer("DATA LOADING", elapsed, f"fetched {sum(len(d) for d in data.values())} bars")
                return data

        # Build 5m from 1m
        try:
            from engine.data import read_bars
            df_5m = read_bars(args.symbol, "5m", data_dir=DATA_DIR)
        except Exception:
            from engine.data_pipeline import load_parquet
            try:
                df_5m = load_parquet(DATA_DIR, args.symbol, "5m")
            except FileNotFoundError:
                logger.info("No 5m data found; resampling from 1m bars...")
                df_5m = BarBuilder.resample(df_1m, freq="5m")

        data = {"1m": df_1m, "5m": df_5m}

    # Clean the data
    from engine.data_pipeline import DataCleaner
    try:
        from engine.utils import load_session_config
        session_config = load_session_config(CONFIG_DIR)
    except Exception:
        session_config = None

    cleaner = DataCleaner(session_config=session_config)
    for tf in list(data.keys()):
        data[tf] = cleaner.clean(data[tf])

    elapsed = time.time() - t0
    _print_data_summary(data)
    stage_footer("DATA LOADING", elapsed, f"{sum(len(d) for d in data.values())} total bars")
    return data


def _print_data_summary(data: dict[str, pl.DataFrame]):
    """Print a summary of loaded data."""
    print()
    print("  Data Summary")
    print("  " + "-" * 50)
    for tf in sorted(data.keys()):
        df = data[tf]
        if df.is_empty():
            print(f"  {tf:>4s}:  (empty)")
            continue
        n = len(df)
        start = df["timestamp"][0]
        end = df["timestamp"][-1]
        print(f"  {tf:>4s}:  {n:>8,} bars  |  {start}  ->  {end}")
    print()
