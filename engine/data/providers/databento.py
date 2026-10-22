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

    Symbology: uses continuous front-month contracts (e.g. "MNQ.c.0")
    with stype_in="continuous".
    """

    name = "databento"

    # Continuous front-month symbology
    SYMBOL_MAP = {
        "NQ": "NQ.c.0",
        "MNQ": "MNQ.c.0",
        "ES": "ES.c.0",
        "MES": "MES.c.0",
        "GC": "GC.c.0",
        "CL": "CL.c.0",
    }

    # Databento schema mapping (SDK v0.70+)
    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "1d": "1d",
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
            stype_in="continuous",
            start=start,
            end=end,
        )

        # Convert to pandas then polars
        pdf = data.to_df()
        if pdf.empty:
            raise ValueError(f"No data returned from Databento for {symbol} {interval}")

        # SDK v0.70+ returns real prices (not fixed-point /1e9)
        df = pl.DataFrame({
            "timestamp": pdf.index.values,
            "open": pdf["open"].values,
            "high": pdf["high"].values,
            "low": pdf["low"].values,
            "close": pdf["close"].values,
            "volume": pdf["volume"].values,
        })

        # Ensure timestamp is timezone-naive UTC datetime
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
        )

        # Add tick_count estimate (Databento OHLCV doesn't provide this directly)
        if "tick_count" not in df.columns:
            df = df.with_columns(
                (pl.col("volume") / 5).cast(pl.Int64).clip(lower_bound=1).alias("tick_count")
            )

        df = self.validate_output(df)

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
            stype_in="continuous",
        )
        live_client.start()

        for record in live_client:
            if hasattr(record, "open"):
                bar = pl.DataFrame({
                    "timestamp": [record.ts_event],
                    "open": [record.open],
                    "high": [record.high],
                    "low": [record.low],
                    "close": [record.close],
                    "volume": [int(record.volume)],
                    "tick_count": [max(1, int(record.volume / 5))],
                })
                bar = bar.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
                )
                yield bar
