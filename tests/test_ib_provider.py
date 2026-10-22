"""Tests for the Interactive Brokers data provider — fully mocked, no real IB."""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import polars as pl
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════
#  TestIBProviderInterface
# ═══════════════════════════════════════════════════════════════════

class TestIBProviderInterface:
    """Test that IBProvider implements the DataProvider interface."""

    def test_implements_interface(self):
        """IBProvider is a proper DataProvider subclass."""
        from engine.data.providers.base import DataProvider
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        assert isinstance(provider, DataProvider)
        assert provider.name == "interactive_brokers"

    def test_list_available_symbols(self):
        """list_available_symbols returns the expected futures."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        symbols = provider.list_available_symbols()
        assert "NQ" in symbols
        assert "MNQ" in symbols
        assert "ES" in symbols
        assert "MES" in symbols
        assert "GC" in symbols
        assert "CL" in symbols

    def test_get_contract_specs(self):
        """get_contract_specs returns valid ContractSpec objects."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        spec = provider.get_contract_specs("MNQ")
        assert spec.symbol == "MNQ"
        assert spec.tick_size == 0.25
        assert spec.tick_value == 0.50
        assert spec.point_value == 2.0

    def test_get_contract_specs_unknown_raises(self):
        """get_contract_specs raises for unknown symbol."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        with pytest.raises(ValueError, match="No contract spec"):
            provider.get_contract_specs("INVALID")


# ═══════════════════════════════════════════════════════════════════
#  TestIBSymbolMapping
# ═══════════════════════════════════════════════════════════════════

class TestIBSymbolMapping:
    """Test symbol/interval mapping."""

    def test_symbol_map_has_exchange(self):
        """Each symbol maps to an exchange."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        for symbol, spec in provider.SYMBOL_MAP.items():
            assert "exchange" in spec
            assert "currency" in spec
            assert spec["currency"] == "USD"

    def test_nq_maps_to_cme(self):
        """NQ/MNQ map to CME."""
        from engine.data.providers.interactive_brokers import IBProvider

        assert IBProvider.SYMBOL_MAP["NQ"]["exchange"] == "CME"
        assert IBProvider.SYMBOL_MAP["MNQ"]["exchange"] == "CME"

    def test_gc_maps_to_comex(self):
        """GC maps to COMEX."""
        from engine.data.providers.interactive_brokers import IBProvider

        assert IBProvider.SYMBOL_MAP["GC"]["exchange"] == "COMEX"

    def test_cl_maps_to_nymex(self):
        """CL maps to NYMEX."""
        from engine.data.providers.interactive_brokers import IBProvider

        assert IBProvider.SYMBOL_MAP["CL"]["exchange"] == "NYMEX"

    def test_interval_map_completeness(self):
        """All common intervals are mapped."""
        from engine.data.providers.interactive_brokers import IBProvider

        for interval in ("1m", "5m", "15m", "30m", "1h", "1d"):
            assert interval in IBProvider.INTERVAL_MAP


# ═══════════════════════════════════════════════════════════════════
#  TestIBHistoricalBars
# ═══════════════════════════════════════════════════════════════════

class TestIBHistoricalBars:
    """Test historical bar fetching with mocked IB."""

    def test_unknown_symbol_raises(self):
        """get_historical_bars raises for unknown symbol."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        with pytest.raises(ValueError, match="Unknown symbol"):
            provider.get_historical_bars("INVALID", "1m", "2025-01-01", "2025-01-02")

    def test_unknown_interval_raises(self):
        """get_historical_bars raises for unknown interval."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        with pytest.raises(ValueError, match="Unknown interval"):
            provider.get_historical_bars("NQ", "3m", "2025-01-01", "2025-01-02")

    def test_missing_dates_raises(self):
        """get_historical_bars raises when start/end dates missing."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        with pytest.raises(ValueError, match="requires both start and end"):
            provider.get_historical_bars("NQ", "1m", start=None, end=None)

    def test_successful_fetch_with_mock(self):
        """get_historical_bars returns valid DataFrame with mocked IB."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()

        # Create mock IB bars
        mock_bars = []
        base_time = datetime(2025, 1, 6, 9, 30)
        for i in range(5):
            bar = MagicMock()
            bar.date = base_time + timedelta(minutes=i)
            bar.open = 20000.0 + i
            bar.high = 20005.0 + i
            bar.low = 19995.0 + i
            bar.close = 20002.0 + i
            bar.volume = 100 + i * 10
            bar.barCount = 50 + i
            mock_bars.append(bar)

        # Mock IB connection — return bars on first call, empty on second
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.reqHistoricalData.side_effect = [mock_bars, []]
        provider._ib = mock_ib

        # Skip pacing in tests
        provider._last_request_time = 0

        # Mock ib_async module for _make_contract
        mock_ib_async = MagicMock()
        with patch.dict(sys.modules, {"ib_async": mock_ib_async}):
            df = provider.get_historical_bars("NQ", "1m", "2025-01-06", "2025-01-06")

        assert len(df) == 5
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "tick_count" in df.columns

        # Check types
        assert df["open"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Int64
        assert df["tick_count"].dtype == pl.Int64

        # Check values
        assert df["open"][0] == pytest.approx(20000.0)
        assert df["close"][-1] == pytest.approx(20006.0)

    def test_empty_response_raises(self):
        """get_historical_bars raises on empty API response."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        provider._last_request_time = 0

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.reqHistoricalData.return_value = []
        provider._ib = mock_ib

        mock_ib_async = MagicMock()
        with patch.dict(sys.modules, {"ib_async": mock_ib_async}):
            with pytest.raises(ValueError, match="No data returned"):
                provider.get_historical_bars("NQ", "1m", "2025-01-06", "2025-01-07")


# ═══════════════════════════════════════════════════════════════════
#  TestIBPacing
# ═══════════════════════════════════════════════════════════════════

class TestIBPacing:
    """Test the pacing rate limiter."""

    def test_pacing_delay_constant(self):
        """Pacing delay should be at least 0.6 seconds."""
        from engine.data.providers.interactive_brokers import IBProvider

        assert IBProvider._PACING_DELAY >= 0.6

    def test_pace_enforces_delay(self):
        """_pace() sleeps if called too quickly."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        provider._last_request_time = time.monotonic()

        start = time.monotonic()
        provider._pace()
        elapsed = time.monotonic() - start

        # Should have slept for close to _PACING_DELAY
        assert elapsed >= IBProvider._PACING_DELAY - 0.05

    def test_pace_no_delay_if_enough_time_passed(self):
        """_pace() doesn't sleep if enough time has passed."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        provider._last_request_time = time.monotonic() - 10.0  # 10s ago

        start = time.monotonic()
        provider._pace()
        elapsed = time.monotonic() - start

        # Should not have slept
        assert elapsed < 0.1


# ═══════════════════════════════════════════════════════════════════
#  TestIBBarConversion
# ═══════════════════════════════════════════════════════════════════

class TestIBBarConversion:
    """Test bar data conversion."""

    def test_bar_to_dict(self):
        """_bar_to_dict converts IB BarData correctly."""
        from engine.data.providers.interactive_brokers import IBProvider

        bar = MagicMock()
        bar.date = datetime(2025, 1, 6, 9, 30)
        bar.open = 20000.0
        bar.high = 20010.0
        bar.low = 19990.0
        bar.close = 20005.0
        bar.volume = 150
        bar.barCount = 75

        result = IBProvider._bar_to_dict(bar)

        assert result["timestamp"] == datetime(2025, 1, 6, 9, 30)
        assert result["open"] == 20000.0
        assert result["high"] == 20010.0
        assert result["low"] == 19990.0
        assert result["close"] == 20005.0
        assert result["volume"] == 150
        assert result["tick_count"] == 75

    def test_bar_to_dict_strips_timezone(self):
        """_bar_to_dict strips timezone info."""
        from engine.data.providers.interactive_brokers import IBProvider
        from zoneinfo import ZoneInfo

        bar = MagicMock()
        bar.date = datetime(2025, 1, 6, 9, 30, tzinfo=ZoneInfo("US/Eastern"))
        bar.open = 20000.0
        bar.high = 20010.0
        bar.low = 19990.0
        bar.close = 20005.0
        bar.volume = 150
        bar.barCount = 75

        result = IBProvider._bar_to_dict(bar)
        assert result["timestamp"].tzinfo is None

    def test_bar_to_dict_string_date(self):
        """_bar_to_dict handles string date format."""
        from engine.data.providers.interactive_brokers import IBProvider

        bar = MagicMock()
        bar.date = "2025-01-06 09:30:00"
        bar.open = 20000.0
        bar.high = 20010.0
        bar.low = 19990.0
        bar.close = 20005.0
        bar.volume = 150
        bar.barCount = 75

        result = IBProvider._bar_to_dict(bar)
        assert isinstance(result["timestamp"], datetime)

    def test_bar_to_dict_fallback_tick_count(self):
        """_bar_to_dict falls back to volume/5 for tick_count."""
        from engine.data.providers.interactive_brokers import IBProvider

        bar = MagicMock(spec=["date", "open", "high", "low", "close", "volume"])
        bar.date = datetime(2025, 1, 6, 9, 30)
        bar.open = 20000.0
        bar.high = 20010.0
        bar.low = 19990.0
        bar.close = 20005.0
        bar.volume = 200
        # No barCount attribute

        result = IBProvider._bar_to_dict(bar)
        assert result["tick_count"] == 40  # 200 / 5

    def test_bars_to_polars(self):
        """_bars_to_polars produces correct Polars DataFrame."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        mock_bars = []
        for i in range(3):
            bar = MagicMock()
            bar.date = datetime(2025, 1, 6, 9, 30 + i)
            bar.open = 20000.0 + i
            bar.high = 20005.0 + i
            bar.low = 19995.0 + i
            bar.close = 20002.0 + i
            bar.volume = 100 + i
            bar.barCount = 50 + i
            mock_bars.append(bar)

        df = provider._bars_to_polars(mock_bars)

        assert len(df) == 3
        assert df["timestamp"].dtype == pl.Datetime("us")
        assert df["open"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Int64
        assert df["tick_count"].dtype == pl.Int64
        # Should be sorted by timestamp
        assert df["timestamp"][0] < df["timestamp"][-1]

    def test_bars_to_polars_deduplicates(self):
        """_bars_to_polars removes duplicate timestamps."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        ts = datetime(2025, 1, 6, 9, 30)
        mock_bars = []
        for _ in range(3):
            bar = MagicMock()
            bar.date = ts  # Same timestamp
            bar.open = 20000.0
            bar.high = 20005.0
            bar.low = 19995.0
            bar.close = 20002.0
            bar.volume = 100
            bar.barCount = 50
            mock_bars.append(bar)

        df = provider._bars_to_polars(mock_bars)
        assert len(df) == 1


# ═══════════════════════════════════════════════════════════════════
#  TestIBFactoryRegistration
# ═══════════════════════════════════════════════════════════════════

class TestIBFactoryRegistration:
    """Test that IB provider is registered in the factory."""

    def test_get_provider_ib(self):
        """get_provider('ib') returns IBProvider."""
        from engine.data import get_provider
        from engine.data.providers.interactive_brokers import IBProvider

        provider = get_provider("ib")
        assert isinstance(provider, IBProvider)

    def test_get_provider_interactive_brokers(self):
        """get_provider('interactive_brokers') also works."""
        from engine.data import get_provider
        from engine.data.providers.interactive_brokers import IBProvider

        provider = get_provider("interactive_brokers")
        assert isinstance(provider, IBProvider)

    def test_factory_error_message_includes_ib(self):
        """Factory error message lists 'ib' as an option."""
        from engine.data import get_provider

        with pytest.raises(ValueError, match="ib"):
            get_provider("nonexistent")


# ═══════════════════════════════════════════════════════════════════
#  TestIBConnection
# ═══════════════════════════════════════════════════════════════════

class TestIBConnection:
    """Test connection management."""

    def test_default_connection_params(self):
        """Default connection parameters from env or hardcoded."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        assert provider._host == "127.0.0.1"
        assert provider._port == 4002
        assert provider._client_id == 1

    def test_custom_connection_params(self):
        """Custom connection parameters override defaults."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider(host="10.0.0.1", port=7497, client_id=5)
        assert provider._host == "10.0.0.1"
        assert provider._port == 7497
        assert provider._client_id == 5

    def test_disconnect_when_not_connected(self):
        """disconnect() is safe to call when not connected."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        # Should not raise
        provider.disconnect()
        assert provider._ib is None

    def test_make_contract_unknown_symbol_raises(self):
        """_make_contract raises for unknown symbol."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        mock_ib_async = MagicMock()
        with patch.dict(sys.modules, {"ib_async": mock_ib_async}):
            with pytest.raises(ValueError, match="Unknown symbol"):
                provider._make_contract("INVALID")

    def test_make_contract_continuous(self):
        """_make_contract creates ContFuture for continuous=True."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        mock_ib_async = MagicMock()
        with patch.dict(sys.modules, {"ib_async": mock_ib_async}):
            provider._make_contract("NQ", continuous=True)
            mock_ib_async.ContFuture.assert_called_once_with(
                symbol="NQ", exchange="CME", currency="USD",
            )

    def test_make_contract_front_month(self):
        """_make_contract creates Future for continuous=False."""
        from engine.data.providers.interactive_brokers import IBProvider

        provider = IBProvider()
        mock_ib_async = MagicMock()
        with patch.dict(sys.modules, {"ib_async": mock_ib_async}):
            provider._make_contract("NQ", continuous=False)
            mock_ib_async.Future.assert_called_once_with(
                symbol="NQ", exchange="CME", currency="USD",
            )


# ═══════════════════════════════════════════════════════════════════
#  TestIBChunking
# ═══════════════════════════════════════════════════════════════════

class TestIBChunking:
    """Test historical data chunking logic."""

    def test_chunk_days_1m(self):
        """1m interval uses 1-day chunks."""
        from engine.data.providers.interactive_brokers import IBProvider

        assert IBProvider._chunk_days("1m") == 1

    def test_chunk_days_1d(self):
        """1d interval uses 365-day chunks."""
        from engine.data.providers.interactive_brokers import IBProvider

        assert IBProvider._chunk_days("1d") == 365

    def test_chunk_days_unknown(self):
        """Unknown interval defaults to 1-day chunks."""
        from engine.data.providers.interactive_brokers import IBProvider

        assert IBProvider._chunk_days("unknown") == 1
