"""Abstract base class for data providers."""

from abc import ABC, abstractmethod
from typing import Iterator

import polars as pl

from engine.utils import ContractSpec


# Standard OHLCV schema that all providers must return
STANDARD_SCHEMA = {
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "tick_count": pl.Int64,
}

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "tick_count"]


class DataProvider(ABC):
    """Abstract interface for market data providers."""

    name: str = "base"

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV bars.

        Args:
            symbol: Instrument symbol (e.g., "NQ", "ES", "MNQ").
            interval: Bar interval (e.g., "1m", "5m", "1h", "1d").
            start: Start date as "YYYY-MM-DD".
            end: End date as "YYYY-MM-DD".

        Returns:
            Polars DataFrame with standard OHLCV schema.
        """

    @abstractmethod
    def get_contract_specs(self, symbol: str) -> ContractSpec:
        """Return the ContractSpec for a symbol."""

    @abstractmethod
    def list_available_symbols(self) -> list[str]:
        """List symbols this provider can serve."""

    def stream_bars(
        self,
        symbol: str,
        interval: str,
    ) -> Iterator[pl.DataFrame]:
        """Stream live bars. Optional — not all providers support this.

        Yields single-row DataFrames as new bars arrive.
        """
        raise NotImplementedError(f"{self.name} does not support live streaming")

    def validate_output(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and enforce the standard OHLCV schema.

        Ensures all required columns exist with correct types.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Provider {self.name} returned DataFrame missing columns: {missing}")

        # Cast types to match standard schema
        casts = []
        for col, expected_type in STANDARD_SCHEMA.items():
            if col in df.columns and df[col].dtype != expected_type:
                if expected_type == pl.Datetime:
                    casts.append(pl.col(col).cast(pl.Datetime("us")))
                else:
                    casts.append(pl.col(col).cast(expected_type))
        if casts:
            df = df.with_columns(casts)

        # Ensure timestamp monotonicity
        df = df.sort("timestamp")

        return df
