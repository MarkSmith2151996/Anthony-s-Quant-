"""Hive-partitioned Parquet storage for market data."""

import logging
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Default data directory
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def write_bars(
    df: pl.DataFrame,
    symbol: str,
    interval: str,
    data_dir: Path | str | None = None,
) -> Path:
    """Write OHLCV bars to Hive-partitioned Parquet.

    Layout: data_dir/symbol=X/interval=Y/year=Z/YYYY-MM.parquet

    Args:
        df: DataFrame with standard OHLCV columns.
        symbol: Instrument symbol (e.g., "NQ", "MNQ").
        interval: Bar interval (e.g., "1m", "5m").
        data_dir: Base data directory. Defaults to data/processed.

    Returns:
        Path to the symbol/interval directory.
    """
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    base_dir = data_dir / f"symbol={symbol}" / f"interval={interval}"

    # Extract year and year-month for partitioning
    df = df.with_columns([
        pl.col("timestamp").dt.year().alias("_year"),
        pl.col("timestamp").dt.strftime("%Y-%m").alias("_month"),
    ])

    for (year, month), group in df.group_by(["_year", "_month"]):
        year_dir = base_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        path = year_dir / f"{month}.parquet"
        table = group.drop(["_year", "_month"]).to_arrow()
        pq.write_table(table, path, compression="zstd", compression_level=3)

    logger.info("Wrote %d bars to %s", len(df), base_dir)
    return base_dir


def read_bars(
    symbol: str,
    interval: str,
    start: str | None = None,
    end: str | None = None,
    data_dir: Path | str | None = None,
) -> pl.DataFrame:
    """Read OHLCV bars from Hive-partitioned or legacy Parquet layout.

    Tries Hive layout first: data_dir/symbol=X/interval=Y/year=Z/*.parquet
    Falls back to legacy flat layout: data_dir/SYMBOL/INTERVAL/*.parquet

    Args:
        symbol: Instrument symbol.
        interval: Bar interval.
        start: Optional start date filter "YYYY-MM-DD".
        end: Optional end date filter "YYYY-MM-DD".
        data_dir: Base data directory.

    Returns:
        Polars DataFrame sorted by timestamp.
    """
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    # Try Hive layout first
    hive_path = data_dir / f"symbol={symbol}" / f"interval={interval}"
    if hive_path.exists():
        return _read_parquet_dir(hive_path, start, end, recursive=True)

    # Fall back to legacy flat layout
    legacy_path = data_dir / symbol / interval
    if legacy_path.exists():
        return _read_parquet_dir(legacy_path, start, end, recursive=False)

    raise FileNotFoundError(
        f"No data found for {symbol}/{interval} in {data_dir}. "
        f"Checked: {hive_path}, {legacy_path}"
    )


def _filter_files_by_date(
    files: list[Path],
    start: str | None,
    end: str | None,
) -> list[Path]:
    """Filter Hive-partitioned files by year directory to skip irrelevant partitions."""
    if not start and not end:
        return files

    start_year = int(start[:4]) if start else 0
    end_year = int(end[:4]) if end else 9999

    filtered = []
    for f in files:
        # Check if any parent directory is a year= partition
        year = None
        for part in f.parts:
            if part.startswith("year="):
                try:
                    year = int(part.split("=")[1])
                except ValueError:
                    pass
        if year is not None and (year < start_year or year > end_year):
            continue
        filtered.append(f)

    return filtered if filtered else files  # fall back to all if filtering removes everything


def _read_parquet_dir(
    path: Path,
    start: str | None,
    end: str | None,
    recursive: bool = False,
) -> pl.DataFrame:
    """Read all parquet files from a directory."""
    if recursive:
        files = sorted(path.rglob("*.parquet"))
    else:
        files = sorted(path.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files in {path}")

    # Skip irrelevant year partitions when date filters are provided
    files = _filter_files_by_date(files, start, end)

    # Read schema from first file to get actual data columns
    # (avoids PyArrow auto-adding Hive partition columns)
    data_columns = pq.read_schema(str(files[0])).names
    tables = [pq.read_table(str(f), columns=data_columns) for f in files]
    combined = pa.concat_tables(tables)
    df = pl.from_arrow(combined).sort("timestamp")

    return _apply_date_filter(df, start, end)


def _apply_date_filter(
    df: pl.DataFrame,
    start: str | None,
    end: str | None,
) -> pl.DataFrame:
    """Apply optional date range filter."""
    if start:
        df = df.filter(pl.col("timestamp") >= pl.lit(start).str.to_datetime())
    if end:
        df = df.filter(pl.col("timestamp") <= pl.lit(end).str.to_datetime())
    return df


def write_continuous(
    df: pl.DataFrame,
    symbol: str,
    method: str = "back_adjusted",
    data_dir: Path | str | None = None,
) -> Path:
    """Write a continuous contract series.

    Layout: data_dir/../continuous/SYMBOL_METHOD.parquet
    """
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    cont_dir = data_dir.parent / "continuous"
    cont_dir.mkdir(parents=True, exist_ok=True)

    path = cont_dir / f"{symbol}_{method}.parquet"
    table = df.to_arrow()
    pq.write_table(table, path, compression="zstd", compression_level=3)
    logger.info("Wrote continuous %s (%s) to %s", symbol, method, path)
    return path


def read_continuous(
    symbol: str,
    method: str = "back_adjusted",
    data_dir: Path | str | None = None,
) -> pl.DataFrame:
    """Read a continuous contract series."""
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    path = data_dir.parent / "continuous" / f"{symbol}_{method}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"No continuous contract file at {path}")

    table = pq.read_table(str(path))
    return pl.from_arrow(table).sort("timestamp")
