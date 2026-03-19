"""MotiveWave CSV data provider adapter."""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from engine.data.providers.base import DataProvider
from engine.utils import ContractSpec, CONTRACT_SPECS

logger = logging.getLogger(__name__)


class MotiveWaveProvider(DataProvider):
    """
    MotiveWave CSV import provider.

    Parses MotiveWave CSV exports into normalized Polars DataFrames.
    """

    name = "motivewave"

    # Common date/time formats MotiveWave may use
    DATE_FORMATS = [
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M",
        "%Y%m%d %H:%M:%S",
    ]

    def __init__(self, raw_dir: str | Path | None = None, output_dir: str | Path | None = None):
        self._raw_dir = Path(raw_dir) if raw_dir else None
        self._output_dir = Path(output_dir) if output_dir else None

    def get_historical_bars(
        self,
        symbol: str,
        interval: str = "1m",
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """Load bars from MotiveWave CSV files.

        Requires raw_dir to be set in constructor.
        """
        if not self._raw_dir:
            raise ValueError("MotiveWaveProvider requires raw_dir to be set")

        df = self.ingest_directory(symbol)
        df = self.validate_output(df)

        # Filter by date range if specified
        if start:
            df = df.filter(pl.col("timestamp") >= datetime.strptime(start, "%Y-%m-%d"))
        if end:
            df = df.filter(pl.col("timestamp") <= datetime.strptime(end, "%Y-%m-%d"))

        return df

    def get_contract_specs(self, symbol: str) -> ContractSpec:
        """Return ContractSpec for a symbol."""
        symbol = symbol.upper()
        if symbol in CONTRACT_SPECS:
            return CONTRACT_SPECS[symbol]
        raise ValueError(f"No contract spec for {symbol}")

    def list_available_symbols(self) -> list[str]:
        """List symbols available from local CSV files."""
        if not self._raw_dir or not self._raw_dir.exists():
            return []
        return [d.name for d in self._raw_dir.iterdir() if d.is_dir()]

    def detect_format(self, filepath: Path) -> dict:
        """Auto-detect CSV format by inspecting first lines."""
        with open(filepath) as f:
            lines = [f.readline() for _ in range(5)]

        # Detect delimiter
        header = lines[0]
        if "\t" in header:
            delimiter = "\t"
        elif ";" in header:
            delimiter = ";"
        else:
            delimiter = ","

        # Detect columns
        cols = [c.strip().strip('"').lower() for c in header.split(delimiter)]

        # Detect date format from first data line
        date_format = None
        if len(lines) > 1:
            first_val = lines[1].split(delimiter)[0].strip().strip('"')
            for fmt in self.DATE_FORMATS:
                try:
                    datetime.strptime(first_val, fmt)
                    date_format = fmt
                    break
                except ValueError:
                    continue

        return {
            "delimiter": delimiter,
            "columns": cols,
            "date_format": date_format,
            "has_header": True,
        }

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize column names to standard: timestamp, open, high, low, close, volume, tick_count."""
        col_map = {}
        lower_cols = {c.lower().strip(): c for c in df.columns}

        # Map common variants
        mappings = {
            "timestamp": ["timestamp", "date", "datetime", "date/time", "time"],
            "open": ["open", "o"],
            "high": ["high", "h"],
            "low": ["low", "l"],
            "close": ["close", "c", "last"],
            "volume": ["volume", "vol", "v"],
            "tick_count": ["tick_count", "ticks", "tick count", "tickcount", "count"],
        }

        for target, variants in mappings.items():
            for v in variants:
                if v in lower_cols:
                    col_map[lower_cols[v]] = target
                    break

        df = df.rename(col_map)

        # Ensure required columns exist
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {df.columns}")

        # Add tick_count if missing
        if "tick_count" not in df.columns:
            df = df.with_columns(pl.lit(1).alias("tick_count"))

        return df.select(["timestamp", "open", "high", "low", "close", "volume", "tick_count"])

    def ingest_file(
        self,
        filepath: Path,
        symbol: str,
        timezone: str = "US/Eastern",
    ) -> pl.DataFrame:
        """Parse a single MotiveWave CSV export."""
        fmt = self.detect_format(filepath)

        df = pl.read_csv(
            filepath,
            separator=fmt["delimiter"],
            has_header=fmt["has_header"],
            try_parse_dates=True,
            ignore_errors=True,
        )

        df = self._normalize_columns(df)

        # Parse timestamp if it's a string
        if df["timestamp"].dtype == pl.Utf8:
            if fmt["date_format"]:
                df = df.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, fmt["date_format"]).alias("timestamp")
                )
            else:
                df = df.with_columns(
                    pl.col("timestamp").str.to_datetime().alias("timestamp")
                )

        # Ensure numeric types
        for col in ["open", "high", "low", "close"]:
            if df[col].dtype != pl.Float64:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
        for col in ["volume", "tick_count"]:
            if df[col].dtype != pl.Int64:
                df = df.with_columns(pl.col(col).cast(pl.Int64))

        # Sort by timestamp
        df = df.sort("timestamp")

        return df

    def ingest_directory(self, symbol: str, pattern: str = "*.csv") -> pl.DataFrame:
        """Batch ingest all CSV files matching pattern."""
        if not self._raw_dir:
            raise ValueError("raw_dir not set")
        files = sorted(self._raw_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {self._raw_dir}")

        dfs = [self.ingest_file(f, symbol) for f in files]
        combined = pl.concat(dfs)
        return combined.sort("timestamp").unique(subset=["timestamp"], keep="first")

    def save_parquet(self, df: pl.DataFrame, symbol: str, timeframe: str) -> Path:
        """Save DataFrame to partitioned Parquet file."""
        if not self._output_dir:
            raise ValueError("output_dir not set")
        out_dir = self._output_dir / symbol / timeframe
        out_dir.mkdir(parents=True, exist_ok=True)

        # Partition by year-month
        df = df.with_columns(
            pl.col("timestamp").dt.strftime("%Y-%m").alias("_partition")
        )

        for partition, group in df.group_by("_partition"):
            month_str = partition[0]
            path = out_dir / f"{month_str}.parquet"
            group.drop("_partition").write_parquet(path)

        return out_dir
