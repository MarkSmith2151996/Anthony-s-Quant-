#!/usr/bin/env python3
"""
Fetch MNQ 1-minute bars from Databento, resample to 15m RTH, save as CSV.

Usage:
    python fetch_bars.py                    # Fetch most recent trading day
    python fetch_bars.py --date 2026-03-25  # Fetch a specific date
    python fetch_bars.py --backfill 30      # Fetch last 30 trading days

Requires:
    pip install databento polars python-dotenv

Config:
    Create ~/berjquant/.env with:
    DATABENTO_API_KEY=db-your-key-here
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_api_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("ERROR: DATABENTO_API_KEY not found.")
        print("Create ~/berjquant/.env with: DATABENTO_API_KEY=db-your-key-here")
        sys.exit(1)
    return key


def fetch_1m(api_key: str, start: str, end: str) -> pl.DataFrame:
    import databento as db
    client = db.Historical(api_key)
    print(f"  Fetching MNQ 1m: {start} → {end}")
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["MNQ.c.0"],
        schema="ohlcv-1m",
        start=start,
        end=end,
    )
    df = pl.from_pandas(data.to_df().reset_index())
    print(f"  Downloaded {len(df):,} raw bars")
    return df


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    ts_col = "ts_event" if "ts_event" in df.columns else "timestamp"
    result = df.select([
        pl.col(ts_col).alias("timestamp"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    ])
    # Convert to ET
    if result["timestamp"].dtype.time_zone:
        result = result.with_columns(
            pl.col("timestamp").dt.convert_time_zone("US/Eastern")
                .dt.replace_time_zone(None)
        )
    else:
        result = result.with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC")
                .dt.convert_time_zone("US/Eastern")
                .dt.replace_time_zone(None)
        )
    result = result.with_columns([
        pl.col("timestamp").dt.date().cast(pl.Utf8).alias("date_et"),
        (pl.col("timestamp").dt.hour() * 100 + pl.col("timestamp").dt.minute()).alias("hhmm"),
    ])
    # RTH only: 9:30–16:00 ET
    return result.filter((pl.col("hhmm") >= 930) & (pl.col("hhmm") < 1600)).sort("timestamp")


def resample_15m(df_1m: pl.DataFrame) -> pl.DataFrame:
    df = df_1m.with_columns(pl.col("timestamp").dt.truncate("15m").alias("bar_ts"))
    counts = df.group_by("bar_ts").len()
    valid = counts.filter(pl.col("len") >= 5)
    result = df.group_by("bar_ts").agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
        pl.col("date_et").last(),
        pl.col("hhmm").last(),
    ]).rename({"bar_ts": "timestamp"}).sort("timestamp")
    return result.join(valid.select("bar_ts"), left_on="timestamp", right_on="bar_ts")


def fetch_date(api_key: str, target: date) -> Path | None:
    start = target.isoformat()
    end = (target + timedelta(days=1)).isoformat()
    try:
        raw = fetch_1m(api_key, start, end)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    if len(raw) == 0:
        print(f"  No data for {target} (weekend/holiday?)")
        return None
    df_1m = normalize(raw)
    if len(df_1m) == 0:
        print(f"  No RTH bars for {target}")
        return None
    df_15m = resample_15m(df_1m)
    path = DATA_DIR / f"mnq_15m_{target.isoformat()}.csv"
    df_15m.write_csv(str(path))
    print(f"  {target}: {len(df_1m)} 1m → {len(df_15m)} 15m bars → {path.name}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Fetch MNQ 15m bars from Databento")
    parser.add_argument("--date", type=str, help="Specific date YYYY-MM-DD")
    parser.add_argument("--backfill", type=int, help="Fetch last N calendar days")
    args = parser.parse_args()

    api_key = load_api_key()

    if args.backfill:
        print(f"\nBackfilling last {args.backfill} days...")
        paths = []
        for i in range(args.backfill, 0, -1):
            target = date.today() - timedelta(days=i)
            if target.weekday() >= 5:
                continue
            p = fetch_date(api_key, target)
            if p:
                paths.append(p)
        if paths:
            combined = pl.concat([pl.read_csv(str(p)) for p in paths]).sort("timestamp")
            combo = DATA_DIR / "mnq_15m_backfill.csv"
            combined.write_csv(str(combo))
            print(f"\n  Combined: {len(combined)} bars → {combo.name}")

    elif args.date:
        fetch_date(api_key, date.fromisoformat(args.date))

    else:
        yesterday = date.today() - timedelta(days=1)
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)
        fetch_date(api_key, yesterday)

    print("Done.")


if __name__ == "__main__":
    main()
