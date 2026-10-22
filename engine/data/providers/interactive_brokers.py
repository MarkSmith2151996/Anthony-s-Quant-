"""Interactive Brokers data provider via ib_async."""

import asyncio
import logging
import os
import time as _time
from datetime import datetime, timedelta
from typing import Iterator

import polars as pl

from engine.data.providers.base import DataProvider
from engine.utils import ContractSpec, CONTRACT_SPECS

logger = logging.getLogger(__name__)


class IBProvider(DataProvider):
    """
    Interactive Brokers data provider — live streaming and historical data.

    Connects to IB Gateway or TWS via the ib_async library. Supports
    historical bar fetches (up to 1 year of 1m data) and real-time
    streaming via keepUpToDate.

    Requires:
        - IB Gateway or TWS running with API enabled
        - ``ib_async`` package: ``pip install ib_async``

    Environment variables:
        - IB_HOST: Gateway host (default: 127.0.0.1)
        - IB_PORT: Gateway port (default: 4002 for paper, 4001 for live)
        - IB_CLIENT_ID: Client ID (default: 1)
    """

    name = "interactive_brokers"

    # Maps our symbols to (exchange, secType, includeExpired)
    SYMBOL_MAP: dict[str, dict] = {
        "NQ": {"exchange": "CME", "currency": "USD"},
        "MNQ": {"exchange": "CME", "currency": "USD"},
        "ES": {"exchange": "CME", "currency": "USD"},
        "MES": {"exchange": "CME", "currency": "USD"},
        "GC": {"exchange": "COMEX", "currency": "USD"},
        "CL": {"exchange": "NYMEX", "currency": "USD"},
    }

    # IB barSize strings for each interval
    INTERVAL_MAP: dict[str, str] = {
        "1m": "1 min",
        "5m": "5 mins",
        "15m": "15 mins",
        "30m": "30 mins",
        "1h": "1 hour",
        "1d": "1 day",
    }

    # IB duration strings — max per request for each bar size
    DURATION_MAP: dict[str, str] = {
        "1m": "1 D",
        "5m": "5 D",
        "15m": "10 D",
        "30m": "30 D",
        "1h": "30 D",
        "1d": "365 D",
    }

    # IB pacing: max 60 requests per 10 minutes → ~0.6s between requests
    _PACING_DELAY = 0.6

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
    ):
        self._host = host or os.environ.get("IB_HOST", "127.0.0.1")
        self._port = port or int(os.environ.get("IB_PORT", "4002"))
        self._client_id = client_id or int(os.environ.get("IB_CLIENT_ID", "1"))
        self._ib = None
        self._last_request_time: float = 0.0

    def _ensure_connected(self):
        """Lazy-connect to IB Gateway."""
        if self._ib is not None and self._ib.isConnected():
            return self._ib

        try:
            import ib_async
        except ImportError:
            raise ImportError(
                "The 'ib_async' package is required for IBProvider. "
                "Install it with: pip install ib_async"
            )

        self._ib = ib_async.IB()
        logger.info(
            "Connecting to IB Gateway at %s:%d (client_id=%d)...",
            self._host, self._port, self._client_id,
        )
        self._ib.connect(self._host, self._port, clientId=self._client_id)
        logger.info("Connected to IB Gateway")
        return self._ib

    def disconnect(self):
        """Disconnect from IB Gateway."""
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IB Gateway")
        self._ib = None

    def _pace(self):
        """Enforce IB pacing limits (60 requests / 10 min)."""
        elapsed = _time.monotonic() - self._last_request_time
        if elapsed < self._PACING_DELAY:
            _time.sleep(self._PACING_DELAY - elapsed)
        self._last_request_time = _time.monotonic()

    def _make_contract(self, symbol: str, continuous: bool = True):
        """Build an IB contract object for the given symbol."""
        import ib_async

        symbol = symbol.upper()
        if symbol not in self.SYMBOL_MAP:
            raise ValueError(
                f"Unknown symbol: {symbol}. "
                f"Available: {list(self.SYMBOL_MAP.keys())}"
            )

        spec = self.SYMBOL_MAP[symbol]

        if continuous:
            return ib_async.ContFuture(
                symbol=symbol,
                exchange=spec["exchange"],
                currency=spec["currency"],
            )
        else:
            return ib_async.Future(
                symbol=symbol,
                exchange=spec["exchange"],
                currency=spec["currency"],
            )

    # ── DataProvider interface ────────────────────────────────────

    def get_historical_bars(
        self,
        symbol: str,
        interval: str = "1m",
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV bars from IB.

        Handles chunked requests for large date ranges to respect IB's
        per-request duration limits.
        """
        symbol = symbol.upper()
        if symbol not in self.SYMBOL_MAP:
            raise ValueError(
                f"Unknown symbol: {symbol}. "
                f"Available: {list(self.SYMBOL_MAP.keys())}"
            )

        bar_size = self.INTERVAL_MAP.get(interval)
        if not bar_size:
            raise ValueError(
                f"Unknown interval: {interval}. "
                f"Available: {list(self.INTERVAL_MAP.keys())}"
            )

        if not start or not end:
            raise ValueError("IBProvider requires both start and end dates")

        ib = self._ensure_connected()
        contract = self._make_contract(symbol, continuous=True)
        ib.qualifyContracts(contract)

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)

        # Determine chunk size based on interval
        chunk_days = self._chunk_days(interval)
        all_bars = []
        current_end = end_dt

        logger.info(
            "Fetching %s %s bars from %s to %s via IB (chunks of %d days)...",
            symbol, interval, start, end, chunk_days,
        )

        while current_end > start_dt:
            self._pace()

            duration = self.DURATION_MAP[interval]
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=current_end.strftime("%Y%m%d-%H:%M:%S"),
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=2,
                )
            except Exception as exc:
                logger.warning("IB historical request failed: %s", exc)
                break

            if not bars:
                break

            all_bars.extend(bars)

            # Move window back
            earliest = min(b.date for b in bars)
            if isinstance(earliest, str):
                earliest = datetime.fromisoformat(earliest)
            current_end = earliest - timedelta(seconds=1)

        if not all_bars:
            raise ValueError(
                f"No data returned from IB for {symbol} {interval} "
                f"({start} to {end})"
            )

        df = self._bars_to_polars(all_bars)

        # Filter to requested range
        df = df.filter(
            (pl.col("timestamp") >= start_dt)
            & (pl.col("timestamp") < end_dt)
        )

        df = self.validate_output(df)

        logger.info(
            "Fetched %d bars for %s (%s) from %s to %s",
            len(df), symbol, interval,
            df["timestamp"][0], df["timestamp"][-1],
        )
        return df

    def get_contract_specs(self, symbol: str) -> ContractSpec:
        """Return ContractSpec for a symbol."""
        symbol = symbol.upper()
        if symbol in CONTRACT_SPECS:
            return CONTRACT_SPECS[symbol]
        raise ValueError(f"No contract spec for {symbol}")

    def list_available_symbols(self) -> list[str]:
        """List symbols available from IB."""
        return list(self.SYMBOL_MAP.keys())

    def stream_bars(
        self,
        symbol: str,
        interval: str = "1m",
    ) -> Iterator[pl.DataFrame]:
        """Stream live bars from IB using keepUpToDate.

        Yields single-row Polars DataFrames as new bars complete.
        Uses reqHistoricalData with keepUpToDate=True, which
        provides real-time bar updates.
        """
        symbol = symbol.upper()
        bar_size = self.INTERVAL_MAP.get(interval)
        if not bar_size:
            raise ValueError(
                f"Unknown interval: {interval}. "
                f"Available: {list(self.INTERVAL_MAP.keys())}"
            )

        ib = self._ensure_connected()
        contract = self._make_contract(symbol, continuous=False)
        ib.qualifyContracts(contract)

        logger.info("Starting live stream for %s (%s) via IB...", symbol, interval)

        # Use keepUpToDate for real-time bars
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=self.DURATION_MAP[interval],
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
            keepUpToDate=True,
        )

        # Track which bars we've already yielded
        seen_count = len(bars)

        while ib.isConnected():
            ib.sleep(1)

            if len(bars) > seen_count:
                # New bars have arrived — yield each one
                for i in range(seen_count, len(bars)):
                    bar = bars[i]
                    row = self._bar_to_dict(bar)
                    yield pl.DataFrame([row]).with_columns(
                        pl.col("timestamp").cast(pl.Datetime("us"))
                    )
                seen_count = len(bars)

    # ── Async streaming (for ib_runner) ──────────────────────────

    async def stream_bars_async(
        self,
        symbol: str,
        interval: str = "1m",
    ):
        """Async generator that yields bars as they arrive from IB.

        Used by ib_runner.py for the asyncio event loop.
        """
        import ib_async

        symbol = symbol.upper()
        bar_size = self.INTERVAL_MAP.get(interval)
        if not bar_size:
            raise ValueError(f"Unknown interval: {interval}")

        ib = self._ensure_connected()
        contract = self._make_contract(symbol, continuous=False)
        ib.qualifyContracts(contract)

        logger.info("Starting async stream for %s (%s)...", symbol, interval)

        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=self.DURATION_MAP[interval],
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
            keepUpToDate=True,
        )

        seen_count = len(bars)

        while ib.isConnected():
            await asyncio.sleep(1)
            ib.sleep(0)  # Process IB events

            if len(bars) > seen_count:
                for i in range(seen_count, len(bars)):
                    yield self._bar_to_dict(bars[i])
                seen_count = len(bars)

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _chunk_days(interval: str) -> int:
        """Return the number of days per chunk for a given interval."""
        return {
            "1m": 1,
            "5m": 5,
            "15m": 10,
            "30m": 30,
            "1h": 30,
            "1d": 365,
        }.get(interval, 1)

    @staticmethod
    def _bar_to_dict(bar) -> dict:
        """Convert a single IB BarData to a dict."""
        ts = bar.date
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        # Strip timezone for UTC-naive storage
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)

        return {
            "timestamp": ts,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
            "tick_count": int(getattr(bar, "barCount", 0)) or max(1, int(bar.volume / 5)),
        }

    def _bars_to_polars(self, bars: list) -> pl.DataFrame:
        """Convert a list of IB BarData objects to a Polars DataFrame."""
        rows = [self._bar_to_dict(b) for b in bars]
        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)

        # Ensure correct types
        df = df.with_columns([
            pl.col("timestamp").cast(pl.Datetime("us")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
            pl.col("tick_count").cast(pl.Int64),
        ])

        # Deduplicate and sort
        df = df.unique(subset=["timestamp"]).sort("timestamp")

        return df
