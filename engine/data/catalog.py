"""Data catalog — discover what data is available locally."""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


class DataCatalog:
    """Catalog of locally available market data."""

    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    def list_symbols(self) -> list[str]:
        """List all symbols with data available.

        Detects both Hive layout (symbol=X/) and legacy layout (X/).
        """
        symbols = set()
        if not self.data_dir.exists():
            return []

        for entry in self.data_dir.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith("symbol="):
                symbols.add(name.split("=", 1)[1])
            elif not name.startswith("."):
                # Legacy layout — directory name is the symbol
                symbols.add(name)

        return sorted(symbols)

    def list_intervals(self, symbol: str) -> list[str]:
        """List available intervals for a symbol."""
        intervals = set()

        # Check Hive layout
        hive_path = self.data_dir / f"symbol={symbol}"
        if hive_path.exists():
            for entry in hive_path.iterdir():
                if entry.is_dir() and entry.name.startswith("interval="):
                    intervals.add(entry.name.split("=", 1)[1])

        # Check legacy layout
        legacy_path = self.data_dir / symbol
        if legacy_path.exists():
            for entry in legacy_path.iterdir():
                if entry.is_dir() and not entry.name.startswith("."):
                    intervals.add(entry.name)

        return sorted(intervals)

    def date_range(self, symbol: str, interval: str) -> tuple[datetime, datetime] | None:
        """Get the date range for a symbol/interval.

        Returns (start, end) datetime tuple, or None if no data.
        """
        from engine.data.storage import read_bars

        try:
            df = read_bars(symbol, interval, data_dir=self.data_dir)
            if df.is_empty():
                return None
            return (df["timestamp"][0], df["timestamp"][-1])
        except FileNotFoundError:
            return None

    def summary(self) -> list[dict]:
        """Get a summary of all available data.

        Returns list of dicts with symbol, interval, bar_count, start, end.
        """
        results = []
        for symbol in self.list_symbols():
            for interval in self.list_intervals(symbol):
                try:
                    from engine.data.storage import read_bars
                    df = read_bars(symbol, interval, data_dir=self.data_dir)
                    results.append({
                        "symbol": symbol,
                        "interval": interval,
                        "bar_count": len(df),
                        "start": str(df["timestamp"][0]) if not df.is_empty() else None,
                        "end": str(df["timestamp"][-1]) if not df.is_empty() else None,
                    })
                except (FileNotFoundError, Exception) as e:
                    logger.debug("Could not read %s/%s: %s", symbol, interval, e)

        return results
