"""Databento data provider adapter."""

import logging
import os
from typing import Iterator

import polars as pl

from engine.data.providers.base import DataProvider
from engine.utils import ContractSpec, CONTRACT_SPECS

logger = logging.getLogger(__name__)


class DatabentoProvider(DataProvider):
    """
    Databento data provider — primary source for historical and live data.

    Supports 15yr+ of 1m history, live streaming, continuous contract symbology.
    Requires the `databento` package and a valid API key.
    """

    name = "databento"

    SYMBOL_MAP = {
        "NQ": "NQ.FUT.CME",
        "MNQ": "MNQ.FUT.CME",
        "ES": "ES.FUT.CME",
        "MES": "MES.FUT.CME",
        "GC": "GC.FUT.COMEX",
        "CL": "CL.FUT.NYMEX",
    }

    # Databento interval string mapping
    INTERVAL_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "1d": "1day",
    }

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("DATABENTO_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Databento client."""
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise ValueError(
                "Databento API key required. Set DATABENTO_API_KEY environment variable "
                "or pass api_key to constructor."
            )

        try:
            import databento as db
        except ImportError:
            raise ImportError(
                "The 'databento' package is required for DatabentoProvider. "
                "Install it with: pip install databento"
            )

        self._client = db.Historical(key=self._api_key)
        return self._client

    def get_historical_bars(
        self,
        symbol: str,
        interval: str = "1m",
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV bars from Databento."""
        # Validate inputs before attempting client/SDK initialization
        db_symbol = self.SYMBOL_MAP.get(symbol.upper())
        if not db_symbol:
            raise ValueError(
                f"Unknown symbol: {symbol}. "
                f"Available: {list(self.SYMBOL_MAP.keys())}"
            )

        db_interval = self.INTERVAL_MAP.get(interval)
        if not db_interval:
            raise ValueError(
                f"Unknown interval: {interval}. "
                f"Available: {list(self.INTERVAL_MAP.keys())}"
            )

        if not start or not end:
            raise ValueError("Databento requires both start and end dates")

        client = self._get_client()

        logger.info(
            "Fetching %s %s bars from %s to %s via Databento...",
            symbol, interval, start, end,
        )

        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[db_symbol],
            schema=f"ohlcv-{db_interval}",
            start=start,
            end=end,
        )

        # Convert to pandas then polars
        pdf = data.to_df()
        if pdf.empty:
            raise ValueError(f"No data returned from Databento for {symbol} {interval}")

        df = pl.DataFrame({
            "timestamp": pdf.index.values if hasattr(pdf.index, 'values') else pdf["ts_event"].values,
            "open": pdf["open"].values / 1e9,  # Databento fixed-point prices
            "high": pdf["high"].values / 1e9,
            "low": pdf["low"].values / 1e9,
            "close": pdf["close"].values / 1e9,
            "volume": pdf["volume"].values,
        })

        # Ensure timestamp type
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
        )

        # Add tick_count (Databento doesn't provide this directly for OHLCV)
        if "tick_count" not in df.columns:
            df = df.with_columns(
                (pl.col("volume") / 5).cast(pl.Int64).clip(lower_bound=1).alias("tick_count")
            )

        df = self.validate_output(df)

        # Resample if needed (e.g., fetch 1m and user wants 5m)
        # This is handled upstream — we return exactly what was requested

        logger.info(
            "Fetched %d bars for %s (%s) from %s to %s",
            len(df), symbol, interval, df["timestamp"][0], df["timestamp"][-1],
        )
        return df

    def get_contract_specs(self, symbol: str) -> ContractSpec:
        """Return ContractSpec for a symbol."""
        symbol = symbol.upper()
        if symbol in CONTRACT_SPECS:
            return CONTRACT_SPECS[symbol]
        raise ValueError(f"No contract spec for {symbol}")

    def list_available_symbols(self) -> list[str]:
        """List symbols available from Databento."""
        return list(self.SYMBOL_MAP.keys())

    def stream_bars(
        self,
        symbol: str,
        interval: str = "1m",
    ) -> Iterator[pl.DataFrame]:
        """Stream live bars from Databento.

        Yields single-row DataFrames as new bars complete.
        """
        if not self._api_key:
            raise ValueError(
                "Databento API key required for streaming. "
                "Set DATABENTO_API_KEY environment variable."
            )

        try:
            import databento as db
        except ImportError:
            raise ImportError(
                "The 'databento' package is required for streaming. "
                "Install it with: pip install databento"
            )

        db_symbol = self.SYMBOL_MAP.get(symbol.upper())
        if not db_symbol:
            raise ValueError(f"Unknown symbol: {symbol}")

        db_interval = self.INTERVAL_MAP.get(interval)
        if not db_interval:
            raise ValueError(f"Unknown interval: {interval}")

        logger.info("Starting live stream for %s (%s)...", symbol, interval)

        live_client = db.Live(key=self._api_key)
        live_client.subscribe(
            dataset="GLBX.MDP3",
            symbols=[db_symbol],
            schema=f"ohlcv-{db_interval}",
        )
        live_client.start()

        for record in live_client:
            if hasattr(record, "open"):
                bar = pl.DataFrame({
                    "timestamp": [record.ts_event],
                    "open": [record.open / 1e9],
                    "high": [record.high / 1e9],
                    "low": [record.low / 1e9],
                    "close": [record.close / 1e9],
                    "volume": [int(record.volume)],
                    "tick_count": [max(1, int(record.volume / 5))],
                })
                bar = bar.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
                )
                yield bar
