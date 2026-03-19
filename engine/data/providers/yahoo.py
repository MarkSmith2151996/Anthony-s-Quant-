"""Yahoo Finance data provider adapter."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from engine.data.providers.base import DataProvider
from engine.utils import ContractSpec, CONTRACT_SPECS

logger = logging.getLogger(__name__)


class YahooProvider(DataProvider):
    """
    Yahoo Finance data provider.

    Limitations:
    - 1m bars: max ~60 days of history (7 days per request)
    - 5m bars: max ~60 days
    - 1h bars: max ~730 days
    - 1d bars: max ~20 years
    - MNQ not available — uses NQ=F as proxy
    """

    name = "yahoo"

    SYMBOL_MAP = {
        "NQ": "NQ=F",
        "MNQ": "NQ=F",  # MNQ tracks NQ; same price, different tick value
        "ES": "ES=F",
        "MES": "ES=F",
        "GC": "GC=F",
        "CL": "CL=F",
    }

    def get_historical_bars(
        self,
        symbol: str,
        interval: str = "1m",
        start: str | None = None,
        end: str | None = None,
    ) -> pl.DataFrame:
        """Fetch OHLCV bars from Yahoo Finance."""
        import yfinance as yf

        yf_symbol = self.SYMBOL_MAP.get(symbol.upper(), f"{symbol}=F")
        ticker = yf.Ticker(yf_symbol)

        if interval == "1m":
            df = self._fetch_1m_chunked(ticker, start, end)
        else:
            if start and end:
                pdf = ticker.history(start=start, end=end, interval=interval)
            else:
                period_map = {
                    "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max",
                }
                pdf = ticker.history(
                    period=period_map.get(interval, "60d"),
                    interval=interval,
                )
            df = self._pandas_to_polars(pdf)

        if df.is_empty():
            raise ValueError(f"No data returned for {yf_symbol} interval={interval}")

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
        """List symbols available from Yahoo."""
        return list(self.SYMBOL_MAP.keys())

    def _fetch_1m_chunked(
        self,
        ticker,
        start: str | None,
        end: str | None,
    ) -> pl.DataFrame:
        """Fetch 1-minute data in 7-day chunks (Yahoo API limit)."""
        import yfinance as yf

        total_days = 59
        chunks = []
        end_dt = datetime.now() if not end else datetime.strptime(end, "%Y-%m-%d")
        days_fetched = 0

        while days_fetched < total_days:
            chunk_days = min(7, total_days - days_fetched)
            start_dt = end_dt - timedelta(days=chunk_days)

            try:
                pdf = ticker.history(
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    interval="1m",
                )
                if not pdf.empty:
                    chunk_df = self._pandas_to_polars(pdf)
                    chunks.append(chunk_df)
                    logger.info(
                        "  Chunk: %d bars (%s to %s)",
                        len(chunk_df), start_dt.date(), end_dt.date(),
                    )
            except Exception as e:
                logger.warning("  Failed chunk %s-%s: %s", start_dt.date(), end_dt.date(), e)

            end_dt = start_dt
            days_fetched += chunk_days

        if not chunks:
            return pl.DataFrame()

        combined = pl.concat(chunks)
        return combined.sort("timestamp").unique(subset=["timestamp"], keep="first")

    def _pandas_to_polars(self, pdf) -> pl.DataFrame:
        """Convert yfinance pandas DataFrame to standard Polars format."""
        if pdf.empty:
            return pl.DataFrame()

        pdf = pdf.reset_index()

        # yfinance returns "Datetime" for intraday, "Date" for daily
        ts_col = "Datetime" if "Datetime" in pdf.columns else "Date"

        df = pl.DataFrame({
            "timestamp": pdf[ts_col].values,
            "open": pdf["Open"].values,
            "high": pdf["High"].values,
            "low": pdf["Low"].values,
            "close": pdf["Close"].values,
            "volume": pdf["Volume"].values,
        })

        # Ensure timestamp is proper datetime, strip timezone for consistency
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None)
        )

        # Add tick_count estimate (not available from Yahoo)
        df = df.with_columns(
            (pl.col("volume") / 10).cast(pl.Int64).alias("tick_count")
        )

        # Filter to trading hours only (8:00-17:00 ET)
        df = df.with_columns(
            pl.col("timestamp").dt.hour().alias("_hour")
        )
        df = df.filter(
            (pl.col("_hour") >= 8) & (pl.col("_hour") < 17)
        ).drop("_hour")

        return df
