"""Tests for the data layer: storage, quality, catalog, and providers."""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ──────────────────────────────────────────────────────────

def make_synthetic_bars(
    num_bars: int = 100,
    start_price: float = 18000.0,
    start_time: datetime | None = None,
) -> pl.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    if start_time is None:
        start_time = datetime(2025, 1, 6, 9, 30)

    np.random.seed(42)
    prices = [start_price]
    for _ in range(num_bars - 1):
        prices.append(prices[-1] + np.random.normal(0, 2.0))

    timestamps = [start_time + timedelta(minutes=i) for i in range(num_bars)]

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": [p + np.random.uniform(-0.5, 0.5) for p in prices],
        "high": [p + abs(np.random.normal(0, 1.0)) for p in prices],
        "low": [p - abs(np.random.normal(0, 1.0)) for p in prices],
        "close": prices,
        "volume": [int(abs(np.random.normal(500, 200))) for _ in range(num_bars)],
        "tick_count": [int(abs(np.random.normal(50, 20))) for _ in range(num_bars)],
    }).with_columns([
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
        pl.col("tick_count").cast(pl.Int64),
    ])


# ═══════════════════════════════════════════════════════════════════
#  TestStorage
# ═══════════════════════════════════════════════════════════════════

class TestStorage:
    """Test Hive-partitioned storage and legacy fallback."""

    def test_write_and_read_hive(self, tmp_path):
        """Write bars with Hive layout and read them back."""
        from engine.data.storage import write_bars, read_bars

        df = make_synthetic_bars(100)
        write_bars(df, "NQ", "1m", data_dir=tmp_path)

        result = read_bars("NQ", "1m", data_dir=tmp_path)
        assert len(result) == 100
        assert "timestamp" in result.columns
        assert result["timestamp"][0] == df["timestamp"][0]
        assert result["timestamp"][-1] == df["timestamp"][-1]

    def test_read_legacy_fallback(self, tmp_path):
        """Read from legacy flat layout (SYMBOL/INTERVAL/*.parquet)."""
        import pyarrow.parquet as pq
        from engine.data.storage import read_bars

        # Create legacy layout
        legacy_dir = tmp_path / "NQ" / "1m"
        legacy_dir.mkdir(parents=True)

        df = make_synthetic_bars(50)
        pq.write_table(df.to_arrow(), legacy_dir / "all.parquet")

        result = read_bars("NQ", "1m", data_dir=tmp_path)
        assert len(result) == 50

    def test_date_range_filtering(self, tmp_path):
        """Test date range filtering on read."""
        from engine.data.storage import write_bars, read_bars

        df = make_synthetic_bars(200)
        write_bars(df, "NQ", "1m", data_dir=tmp_path)

        # Filter to a subset
        start = "2025-01-06 10:00:00"
        end = "2025-01-06 11:00:00"
        result = read_bars("NQ", "1m", start=start, end=end, data_dir=tmp_path)
        assert len(result) < 200
        assert result["timestamp"][0] >= datetime(2025, 1, 6, 10, 0)

    def test_write_read_continuous(self, tmp_path):
        """Test continuous contract write/read."""
        from engine.data.storage import write_continuous, read_continuous

        df = make_synthetic_bars(100)
        write_continuous(df, "NQ", "back_adjusted", data_dir=tmp_path)

        result = read_continuous("NQ", "back_adjusted", data_dir=tmp_path)
        assert len(result) == 100

    def test_file_not_found_raises(self, tmp_path):
        """Test that missing data raises FileNotFoundError."""
        from engine.data.storage import read_bars

        with pytest.raises(FileNotFoundError):
            read_bars("MISSING", "1m", data_dir=tmp_path)


# ═══════════════════════════════════════════════════════════════════
#  TestQuality
# ═══════════════════════════════════════════════════════════════════

class TestQuality:
    """Test data quality validation."""

    def test_validate_good_ohlcv(self):
        """Good OHLCV data passes validation unchanged."""
        from engine.data.quality import DataCleaner

        df = make_synthetic_bars(50)
        cleaner = DataCleaner()
        result = cleaner.validate_ohlcv(df)
        assert len(result) == 50

    def test_validate_fixes_bad_high(self):
        """High < close should be fixed."""
        from engine.data.quality import DataCleaner

        df = pl.DataFrame({
            "timestamp": [datetime(2025, 1, 6, 9, 30)],
            "open": [100.0],
            "high": [99.0],  # Invalid: high < open
            "low": [98.0],
            "close": [101.0],
            "volume": [100],
            "tick_count": [10],
        })
        cleaner = DataCleaner()
        result = cleaner.validate_ohlcv(df)
        # High should be corrected to max(open, high, close) = 101.0
        assert result["high"][0] == 101.0

    def test_remove_bad_ticks(self):
        """Bars with > 2% price change should be removed."""
        from engine.data.quality import DataCleaner

        timestamps = [datetime(2025, 1, 6, 9, 30 + i) for i in range(5)]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [100.0, 100.5, 200.0, 100.5, 101.0],  # 200.0 is a bad tick
            "high": [101.0, 101.0, 201.0, 101.0, 102.0],
            "low": [99.0, 100.0, 199.0, 100.0, 100.0],
            "close": [100.5, 100.5, 200.0, 100.5, 101.0],
            "volume": [100, 100, 100, 100, 100],
            "tick_count": [10, 10, 10, 10, 10],
        })
        cleaner = DataCleaner()
        result = cleaner.remove_bad_ticks(df, max_pct_change=2.0)
        # The 200.0 bar should be removed
        assert len(result) < 5
        assert 200.0 not in result["close"].to_list()

    def test_detect_gaps(self):
        """Gap detection should find time gaps."""
        from engine.data.quality import DataCleaner

        timestamps = [
            datetime(2025, 1, 6, 9, 30),
            datetime(2025, 1, 6, 9, 31),
            datetime(2025, 1, 6, 9, 35),  # 4 minute gap
            datetime(2025, 1, 6, 9, 36),
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [100.0, 100.5, 101.0, 101.5],
            "high": [101.0, 101.0, 102.0, 102.0],
            "low": [99.0, 100.0, 100.0, 101.0],
            "close": [100.5, 100.5, 101.5, 101.5],
            "volume": [100, 100, 100, 100],
            "tick_count": [10, 10, 10, 10],
        })
        cleaner = DataCleaner()
        gaps = cleaner.detect_gaps(df, max_gap_seconds=120)
        assert len(gaps) == 1
        assert gaps["gap_seconds"][0] == 240.0  # 4 minutes

    def test_clean_pipeline(self):
        """Full clean pipeline should not crash."""
        from engine.data.quality import DataCleaner

        df = make_synthetic_bars(100)
        cleaner = DataCleaner()
        result = cleaner.clean(df)
        assert len(result) > 0
        assert "timestamp" in result.columns


# ═══════════════════════════════════════════════════════════════════
#  TestCatalog
# ═══════════════════════════════════════════════════════════════════

class TestCatalog:
    """Test data catalog."""

    def test_list_symbols_hive(self, tmp_path):
        """List symbols from Hive layout."""
        from engine.data.storage import write_bars
        from engine.data.catalog import DataCatalog

        df = make_synthetic_bars(50)
        write_bars(df, "NQ", "1m", data_dir=tmp_path)
        write_bars(df, "ES", "1m", data_dir=tmp_path)

        catalog = DataCatalog(data_dir=tmp_path)
        symbols = catalog.list_symbols()
        assert "NQ" in symbols
        assert "ES" in symbols

    def test_list_symbols_legacy(self, tmp_path):
        """List symbols from legacy layout."""
        import pyarrow.parquet as pq
        from engine.data.catalog import DataCatalog

        legacy_dir = tmp_path / "MNQ" / "1m"
        legacy_dir.mkdir(parents=True)
        pq.write_table(make_synthetic_bars(10).to_arrow(), legacy_dir / "all.parquet")

        catalog = DataCatalog(data_dir=tmp_path)
        symbols = catalog.list_symbols()
        assert "MNQ" in symbols

    def test_list_intervals(self, tmp_path):
        """List intervals for a symbol."""
        from engine.data.storage import write_bars
        from engine.data.catalog import DataCatalog

        df = make_synthetic_bars(50)
        write_bars(df, "NQ", "1m", data_dir=tmp_path)
        write_bars(df, "NQ", "5m", data_dir=tmp_path)

        catalog = DataCatalog(data_dir=tmp_path)
        intervals = catalog.list_intervals("NQ")
        assert "1m" in intervals
        assert "5m" in intervals

    def test_date_range(self, tmp_path):
        """Get date range for a symbol/interval."""
        from engine.data.storage import write_bars
        from engine.data.catalog import DataCatalog

        df = make_synthetic_bars(100)
        write_bars(df, "NQ", "1m", data_dir=tmp_path)

        catalog = DataCatalog(data_dir=tmp_path)
        dr = catalog.date_range("NQ", "1m")
        assert dr is not None
        assert dr[0] == df["timestamp"][0]
        assert dr[1] == df["timestamp"][-1]

    def test_summary(self, tmp_path):
        """Get a full catalog summary."""
        from engine.data.storage import write_bars
        from engine.data.catalog import DataCatalog

        df = make_synthetic_bars(50)
        write_bars(df, "NQ", "1m", data_dir=tmp_path)

        catalog = DataCatalog(data_dir=tmp_path)
        summary = catalog.summary()
        assert len(summary) == 1
        assert summary[0]["symbol"] == "NQ"
        assert summary[0]["interval"] == "1m"
        assert summary[0]["bar_count"] == 50


# ═══════════════════════════════════════════════════════════════════
#  TestProviderInterface
# ═══════════════════════════════════════════════════════════════════

class TestProviderInterface:
    """Test that providers implement DataProvider interface."""

    def test_yahoo_implements_interface(self):
        """YahooProvider implements DataProvider."""
        from engine.data.providers.base import DataProvider
        from engine.data.providers.yahoo import YahooProvider

        provider = YahooProvider()
        assert isinstance(provider, DataProvider)
        assert provider.name == "yahoo"
        assert "NQ" in provider.list_available_symbols()
        assert "MNQ" in provider.list_available_symbols()

    def test_databento_implements_interface(self):
        """DatabentoProvider implements DataProvider."""
        from engine.data.providers.base import DataProvider
        from engine.data.providers.databento import DatabentoProvider

        provider = DatabentoProvider()
        assert isinstance(provider, DataProvider)
        assert provider.name == "databento"
        assert "NQ" in provider.list_available_symbols()
        assert "ES" in provider.list_available_symbols()

    def test_databento_raises_without_key(self):
        """DatabentoProvider raises without API key."""
        import os
        from engine.data.providers.databento import DatabentoProvider

        # Temporarily clear the env var if set
        old_key = os.environ.pop("DATABENTO_API_KEY", None)
        try:
            provider = DatabentoProvider(api_key=None)
            with pytest.raises((ValueError, ImportError)):
                provider.get_historical_bars("NQ", "1m", "2025-01-01", "2025-01-02")
        finally:
            if old_key is not None:
                os.environ["DATABENTO_API_KEY"] = old_key

    def test_motivewave_implements_interface(self):
        """MotiveWaveProvider implements DataProvider."""
        from engine.data.providers.base import DataProvider
        from engine.data.providers.motivewave import MotiveWaveProvider

        provider = MotiveWaveProvider()
        assert isinstance(provider, DataProvider)
        assert provider.name == "motivewave"

    def test_get_provider_factory(self):
        """get_provider() factory returns correct types."""
        from engine.data import get_provider
        from engine.data.providers.yahoo import YahooProvider
        from engine.data.providers.databento import DatabentoProvider
        from engine.data.providers.motivewave import MotiveWaveProvider

        assert isinstance(get_provider("yahoo"), YahooProvider)
        assert isinstance(get_provider("databento"), DatabentoProvider)
        assert isinstance(get_provider("motivewave"), MotiveWaveProvider)

    def test_get_provider_unknown_raises(self):
        """get_provider() raises for unknown provider."""
        from engine.data import get_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_validate_output(self):
        """DataProvider.validate_output() enforces schema."""
        from engine.data.providers.base import DataProvider

        df = make_synthetic_bars(10)

        # Create a concrete subclass for testing
        class TestProvider(DataProvider):
            name = "test"
            def get_historical_bars(self, *a, **kw): ...
            def get_contract_specs(self, *a, **kw): ...
            def list_available_symbols(self): return []

        provider = TestProvider()
        result = provider.validate_output(df)
        assert len(result) == 10
        assert result["open"].dtype == pl.Float64
        assert result["volume"].dtype == pl.Int64

    def test_validate_output_missing_column_raises(self):
        """validate_output raises if required column is missing."""
        from engine.data.providers.base import DataProvider

        df = pl.DataFrame({
            "timestamp": [datetime(2025, 1, 6)],
            "open": [100.0],
            # missing high, low, close, volume, tick_count
        })

        class TestProvider(DataProvider):
            name = "test"
            def get_historical_bars(self, *a, **kw): ...
            def get_contract_specs(self, *a, **kw): ...
            def list_available_symbols(self): return []

        provider = TestProvider()
        with pytest.raises(ValueError, match="missing columns"):
            provider.validate_output(df)


# ═══════════════════════════════════════════════════════════════════
#  TestBackwardCompat
# ═══════════════════════════════════════════════════════════════════

class TestBackwardCompat:
    """Test backward compatibility of old import paths."""

    def test_data_pipeline_imports(self):
        """Old imports from engine.data_pipeline still work."""
        from engine.data_pipeline import DataCleaner, BarBuilder, MotiveWaveIngestor, load_parquet
        assert DataCleaner is not None
        assert BarBuilder is not None
        assert MotiveWaveIngestor is not None
        assert callable(load_parquet)

    def test_data_fetcher_imports(self):
        """Old imports from engine.data_fetcher still work."""
        try:
            from engine.data_fetcher import YFinanceFetcher
            assert YFinanceFetcher is not None
        except ImportError:
            # yfinance may not be installed in test environment
            pytest.skip("yfinance not installed")

    def test_new_package_imports(self):
        """New import paths from engine.data work."""
        from engine.data import get_provider, read_bars, write_bars, DataCleaner, ContractStitcher, DataCatalog
        assert callable(get_provider)
        assert callable(read_bars)
        assert callable(write_bars)
        assert DataCleaner is not None
        assert ContractStitcher is not None
        assert DataCatalog is not None


# ═══════════════════════════════════════════════════════════════════
#  TestContractSpecs
# ═══════════════════════════════════════════════════════════════════

class TestContractSpecs:
    """Test that CONTRACT_SPECS loads from YAML."""

    def test_hardcoded_specs_present(self):
        """MNQ and NQ specs are always present."""
        from engine.utils import CONTRACT_SPECS
        assert "MNQ" in CONTRACT_SPECS
        assert "NQ" in CONTRACT_SPECS

    def test_yaml_specs_loaded(self):
        """Additional specs from config/instruments/ are loaded."""
        from engine.utils import CONTRACT_SPECS
        # ES and GC should be loaded from YAML
        assert "ES" in CONTRACT_SPECS
        assert "GC" in CONTRACT_SPECS
        assert "CL" in CONTRACT_SPECS

    def test_yaml_spec_values(self):
        """YAML-loaded specs have correct values."""
        from engine.utils import CONTRACT_SPECS
        es = CONTRACT_SPECS.get("ES")
        if es:
            assert es.tick_size == 0.25
            assert es.tick_value == 12.50
            assert es.point_value == 50.0


# ═══════════════════════════════════════════════════════════════════
#  TestDatabentoMocked
# ═══════════════════════════════════════════════════════════════════

class TestDatabentoMocked:
    """Databento provider tests with mocked SDK — no real API calls."""

    def test_unknown_symbol_raises(self):
        """get_historical_bars raises for unknown symbol."""
        from engine.data.providers.databento import DatabentoProvider

        provider = DatabentoProvider(api_key="test-key-123")
        with pytest.raises(ValueError, match="Unknown symbol"):
            provider.get_historical_bars("INVALID", "1m", "2025-01-01", "2025-01-02")

    def test_unknown_interval_raises(self):
        """get_historical_bars raises for unknown interval."""
        from engine.data.providers.databento import DatabentoProvider

        provider = DatabentoProvider(api_key="test-key-123")
        with pytest.raises(ValueError, match="Unknown interval"):
            provider.get_historical_bars("NQ", "3m", "2025-01-01", "2025-01-02")

    def test_missing_dates_raises(self):
        """get_historical_bars raises when start/end dates missing."""
        from engine.data.providers.databento import DatabentoProvider

        provider = DatabentoProvider(api_key="test-key-123")
        with pytest.raises(ValueError, match="requires both start and end"):
            provider.get_historical_bars("NQ", "1m", start=None, end=None)

    def test_empty_response_raises(self):
        """get_historical_bars raises on empty API response."""
        from unittest.mock import MagicMock, patch
        from engine.data.providers.databento import DatabentoProvider
        import pandas as pd

        provider = DatabentoProvider(api_key="test-key-123")
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.to_df.return_value = pd.DataFrame()  # empty
        mock_client.timeseries.get_range.return_value = mock_result
        provider._client = mock_client

        with pytest.raises(ValueError, match="No data returned"):
            provider.get_historical_bars("NQ", "1m", "2025-01-01", "2025-01-02")

    def test_successful_fetch_with_mock(self):
        """get_historical_bars returns valid DataFrame with mocked response."""
        from unittest.mock import MagicMock
        from engine.data.providers.databento import DatabentoProvider
        import pandas as pd
        import numpy as np

        provider = DatabentoProvider(api_key="test-key-123")
        mock_client = MagicMock()

        # Create a mock pandas DataFrame matching Databento format
        n = 5
        mock_pdf = pd.DataFrame({
            "ts_event": pd.date_range("2025-01-06 09:30", periods=n, freq="1min"),
            "open": np.array([20000, 20001, 20002, 20003, 20004]) * int(1e9),
            "high": np.array([20005, 20006, 20007, 20008, 20009]) * int(1e9),
            "low": np.array([19995, 19996, 19997, 19998, 19999]) * int(1e9),
            "close": np.array([20002, 20003, 20004, 20005, 20006]) * int(1e9),
            "volume": [100, 150, 200, 125, 175],
        })

        mock_result = MagicMock()
        mock_result.to_df.return_value = mock_pdf
        mock_client.timeseries.get_range.return_value = mock_result
        provider._client = mock_client

        # Patch 'import databento' inside get_historical_bars
        import sys
        mock_db = MagicMock()
        sys.modules["databento"] = mock_db

        try:
            df = provider.get_historical_bars("NQ", "1m", "2025-01-06", "2025-01-07")

            assert len(df) == 5
            assert "timestamp" in df.columns
            assert "open" in df.columns
            assert "volume" in df.columns
            # Check fixed-point conversion (divided by 1e9)
            assert df["open"][0] == pytest.approx(20000.0, rel=1e-6)
        finally:
            del sys.modules["databento"]

    def test_stream_bars_raises_without_key(self):
        """stream_bars raises without API key."""
        from engine.data.providers.databento import DatabentoProvider

        import os
        old_key = os.environ.pop("DATABENTO_API_KEY", None)
        try:
            provider = DatabentoProvider(api_key=None)
            with pytest.raises(ValueError, match="API key required"):
                next(provider.stream_bars("NQ", "1m"))
        finally:
            if old_key is not None:
                os.environ["DATABENTO_API_KEY"] = old_key


# ═══════════════════════════════════════════════════════════════════
#  TestStoragePartitionPruning
# ═══════════════════════════════════════════════════════════════════

class TestStoragePartitionPruning:
    """Test that read_bars with date filters prunes irrelevant year partitions."""

    def test_year_pruning(self, tmp_path):
        """Date-filtered reads skip irrelevant year partitions."""
        from engine.data.storage import write_bars, read_bars, _filter_files_by_date
        from pathlib import Path

        # Create data spanning 2023 and 2024
        dates_2023 = [datetime(2023, 6, 1, i) for i in range(10)]
        dates_2024 = [datetime(2024, 6, 1, i) for i in range(10)]
        all_dates = dates_2023 + dates_2024

        df = pl.DataFrame({
            "timestamp": all_dates,
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.5] * 20,
            "volume": [1000] * 20,
        })
        write_bars(df, "TEST", "1h", data_dir=tmp_path)

        # Collect all parquet files
        all_files = sorted(tmp_path.rglob("*.parquet"))
        assert len(all_files) == 2  # one per year

        # Filter for 2024 only
        filtered = _filter_files_by_date(all_files, "2024-01-01", "2024-12-31")
        assert len(filtered) == 1
        assert "year=2024" in str(filtered[0])

    def test_roundtrip_no_extra_columns(self, tmp_path):
        """Write then read produces identical columns (no Hive partition leaks)."""
        from engine.data.storage import write_bars, read_bars

        df = make_synthetic_bars(50)
        write_bars(df, "X", "5m", data_dir=tmp_path)
        df2 = read_bars("X", "5m", data_dir=tmp_path)
        assert df2.columns == df.columns
