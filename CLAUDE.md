# CLAUDE.md — Instructions for Claude Code

## Project Overview

**MCQ Engine** (Monte Carlo Quantitative Trading Engine) is a strategy research engine that generates, tests, and validates intraday trading strategies for CME futures using Monte Carlo simulation. The primary goal is passing prop firm evaluations (Topstep, Apex, Lucid).

Two developers work on this repo independently and merge periodically:
- **Berj** — Engine internals, signals, strategy generation, Monte Carlo, validation
- **Project owner** — Data infrastructure, multi-asset scalability, pipeline robustness

## Tech Stack

- **Python 3.11+** (required)
- **Polars** for all DataFrames (NOT pandas — only use pandas at API boundaries when a provider returns it, then immediately convert to Polars)
- **PyArrow** for Parquet I/O
- **NumPy/SciPy** for numerical computation
- **Plotly** for visualization
- **PyYAML** for config files
- **pytest** for testing

## Architecture

```
engine/          Core backtester, risk manager, metrics
engine/data/     Data provider abstraction, storage, quality, catalog
  providers/     DataProvider ABC + adapters (databento, yahoo, motivewave)
  storage.py     Hive-partitioned Parquet read/write
  quality.py     Data validation and cleaning (re-exports DataCleaner)
  contracts.py   Continuous contract stitching (re-exports ContractStitcher)
  catalog.py     DataCatalog — list available symbols, intervals, date ranges
signals/         Signal library: trend, momentum, volatility, volume, orderflow, price_action, time_filters
strategies/      Strategy generator (combinatorial), serializer, saved strategies
monte_carlo/     MC simulator, scoring, visualization
validation/      Walk-forward analysis, correlation filtering, regime detection
pipeline/        Pipeline stages (cli, config, data_loader, search, monte_carlo, validation, paper_trading, reporting)
live/            Paper trader, signal engine, alerts, dashboard
config/          Prop firm rules, session definitions, events calendar, instrument specs
data/            raw/ → processed/ → features/ (Parquet, Hive-partitioned)
tests/           pytest tests organized by phase
```

## Key Design Patterns

### Strategy Protocol
All strategies implement this interface (defined in `engine/backtester.py`):
```python
class Strategy(Protocol):
    name: str
    def compute_signals(self, data: dict[str, pl.DataFrame]) -> pl.DataFrame: ...
    def get_stop_loss(self, entry_price: float, direction: str) -> float | None: ...
    def get_take_profit(self, entry_price: float, direction: str) -> float | None: ...
```

### ContractSpec
Asset-specific parameters live in `engine/utils.py` as frozen dataclasses. `CONTRACT_SPECS` is loaded dynamically from `config/instruments/*.yaml` at import time, with hardcoded fallbacks for MNQ and NQ:
```python
ContractSpec(symbol="MNQ", tick_size=0.25, tick_value=0.50, point_value=2.0)
```
Currently loaded: MNQ, NQ, ES, MES, GC, CL.

### Risk Manager
The `RiskManager` is always-on and cannot be bypassed. It enforces prop firm daily loss limits, trailing drawdown, session windows, event blackouts, and position limits. Every order goes through it.

### Data Format
The standard internal DataFrame format is:
```
timestamp: Datetime (UTC, timezone-naive internally)
open: Float64
high: Float64
low: Float64
close: Float64
volume: Int64
tick_count: Int64
symbol: Utf8 (optional, present after ingestion)
session_segment: Utf8 (optional, added by DataCleaner.tag_sessions)
```

## Data Layer

### Provider Abstraction (implemented)

The data layer uses a provider abstraction pattern with a factory function:

```python
from engine.data import get_provider, read_bars, write_bars, DataCatalog

# Factory returns the right provider
provider = get_provider("databento")  # or "yahoo", "motivewave"
df = provider.get_historical_bars("NQ", "1m", "2024-01-01", "2024-12-31")

# Direct storage access
df = read_bars("MNQ", "1m", start="2024-06-01", end="2024-12-31")
write_bars(df, "MNQ", "1m")
```

All providers implement `DataProvider` ABC from `engine/data/providers/base.py`:
```python
class DataProvider(ABC):
    @abstractmethod
    def get_historical_bars(self, symbol, interval, start, end) -> pl.DataFrame: ...
    @abstractmethod
    def get_contract_specs(self, symbol) -> ContractSpec: ...
    @abstractmethod
    def list_available_symbols(self) -> list[str]: ...
    def stream_bars(self, symbol, interval) -> Iterator[pl.DataFrame]: ...  # optional
    def validate_output(self, df) -> pl.DataFrame: ...  # schema enforcement
```

### Implemented Providers

| Provider | Module | Status | Notes |
|----------|--------|--------|-------|
| Databento | `engine/data/providers/databento.py` | Primary | 15yr+ history, live streaming, requires `databento` package + API key |
| Yahoo | `engine/data/providers/yahoo.py` | Fallback | ~60 days of 1m data, chunked fetching, unreliable |
| MotiveWave | `engine/data/providers/motivewave.py` | Legacy | CSV import from MotiveWave platform |

### Planned Providers (not yet implemented)
- **FirstRate Data** — bulk historical CSVs, one-time purchase
- **Interactive Brokers** — live streaming for paper trading

### Storage Layout (Hive-partitioned Parquet)

```
data/
  processed/
    symbol=ES/
      interval=1m/
        year=2024/
          2024-01.parquet
          2024-02.parquet
    symbol=NQ/
      interval=1m/
        year=2024/
          ...
  continuous/
    ES_back_adjusted.parquet
    NQ_back_adjusted.parquet
```

`read_bars()` tries the Hive layout first, then falls back to the legacy flat layout (`SYMBOL/INTERVAL/*.parquet`). Year-level partition pruning is applied when date filters are provided.

### Backward Compatibility

Old import paths still work:
- `from engine.data_pipeline import DataCleaner, BarBuilder, MotiveWaveIngestor, load_parquet`
- `from engine.data_fetcher import YFinanceFetcher`

### Key rules for the data layer
- All timestamps stored as UTC, timezone-naive in Polars
- Exchange timezone stored as metadata in Parquet file metadata, NOT in the timestamp column
- Session tagging (RTH/ETH/pre_market/core/post_close) happens at query time or as a computed column, not baked into storage
- Continuous contracts: use back-adjustment (Panama) for signal generation, keep unadjusted prices alongside for return calculations
- Roll logic: volume-based for equity indices (ES, NQ), calendar-based for commodities (GC, CL)
- Validate every incoming DataFrame: OHLCV sanity, timestamp monotonicity, gap detection, outlier flagging
- Use Zstd compression level 3 for Parquet writes
- Parquet partition granularity: year/month (one file per month per symbol per interval)

## Config System

### Prop Firms (`config/prop_firms.yaml`)
Each profile defines: account_size, daily_loss_limit, max_drawdown, drawdown_type, max_contracts per symbol, consistency rules, commissions. Add new firms by adding a new YAML block.

### Sessions (`config/sessions.yaml`)
Defines pre_market, core, post_close windows in US/Eastern. The `flat_by` time forces all positions closed.

### Instrument Configs (`config/instruments/`)
Six instruments configured via YAML files: NQ, MNQ, ES, MES, GC, CL. `CONTRACT_SPECS` in `engine/utils.py` loads these dynamically at import time. Example:
```yaml
# config/instruments/NQ.yaml
symbol: NQ
exchange: CME
asset_class: equity_index
tick_size: 0.25
tick_value: 5.00
point_value: 20.0
roll_method: volume
roll_days_before_expiry: 8
contract_months: [H, M, U, Z]
```

## Pipeline (`run_pipeline.py` + `pipeline/`)

The pipeline is a thin orchestrator (`run_pipeline.py`, ~150 lines) that imports from the `pipeline/` package:

| Module | Purpose |
|--------|---------|
| `pipeline/cli.py` | `parse_args()`, `banner()` |
| `pipeline/config.py` | Path constants, stage helpers, config builders |
| `pipeline/data_loader.py` | Stage 0: load data via `read_bars()` with Yahoo fallback |
| `pipeline/search.py` | Stage 1: strategy generation + parallel backtesting |
| `pipeline/monte_carlo.py` | Stage 2: MC simulation + scoring |
| `pipeline/validation.py` | Stage 3: walk-forward + regime analysis |
| `pipeline/paper_trading.py` | Stage 4: paper trading replay |
| `pipeline/reporting.py` | Final leaderboard + summary |

Run with: `python run_pipeline.py --stage {search|mc|validate|paper|all}`

## Coding Conventions

- **Type hints everywhere**. Use `str | None` syntax (not `Optional[str]`).
- **Dataclasses** for all data structures. Use `frozen=True` for immutable types.
- **Polars over pandas**. Convert at API boundaries only.
- **Pathlib** for all file paths. Never use string paths.
- **f-strings** for formatting.
- **Logging** via `logging.getLogger(__name__)`, not print statements (except in CLI entry points).
- Imports: stdlib → third-party → local, separated by blank lines.
- No wildcard imports.
- Docstrings on all public classes and functions.
- Test files mirror source structure: `engine/backtester.py` → `tests/test_engine.py`.

## Running Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/test_engine.py -v     # Phase 1 (engine core)
pytest tests/test_phase2.py -v     # Phase 2 (signals + generator)
pytest tests/test_phase3.py -v     # Phase 3 (Monte Carlo)
pytest tests/test_phase4.py -v     # Phase 4 (validation)
pytest tests/test_phase5.py -v     # Phase 5 (live/paper)
pytest tests/test_data.py -v      # Data layer (storage, providers, catalog)
```

Tests use synthetic data generated in `tests/test_engine.py::generate_synthetic_data()`. Data layer tests in `tests/test_data.py` use small synthetic DataFrames with mocked providers — no external API calls.

## Common Tasks

### Adding a new instrument
1. Create `config/instruments/<SYMBOL>.yaml` with contract spec fields
2. `CONTRACT_SPECS` auto-loads from YAML — no code changes needed
3. Add symbol mapping to each data provider adapter
4. Add session config if trading hours differ from existing instruments
5. Add to prop firm configs if max_contracts differs

### Adding a new data provider
1. Create `engine/data/providers/<name>.py`
2. Implement `DataProvider` ABC from `engine/data/providers/base.py`
3. Call `self.validate_output(df)` before returning from `get_historical_bars()`
4. Handle provider-specific timestamp formats, column names, and quirks in the adapter
5. Register in `engine/data/__init__.py` → `get_provider()` factory
6. Add tests in `tests/test_data.py` (mock all external calls)

### Adding a new signal
1. Add compute function in appropriate `signals/<category>.py` module
2. Register in `signals/registry.py` with parameter ranges
3. The strategy generator will automatically pick it up

## What NOT to Do

- Do NOT use pandas DataFrames inside the engine. Polars only.
- Do NOT hardcode "MNQ" or "NQ" anywhere new. Use `ContractSpec.symbol` or config.
- Do NOT store timezone-aware timestamps in Parquet. Store UTC naive.
- Do NOT make external API calls in tests. Mock or use synthetic data.
- Do NOT put large data files in git. The `data/` directory is gitignored except `.gitkeep`.
- Do NOT bypass the RiskManager. All orders flow through it.
- Do NOT use `print()` in library code. Use `logging`.
- Do NOT duplicate logic between `run_search.py`, `run_search2.py`, `run_search3.py` — these need consolidation (tech debt).

## Known Tech Debt

- `run_search.py`, `run_search2.py`, `run_search3.py` are three iterations of the same search script. Consolidate into one.
- `live/dashboard.py` line 176 has a Python 3.11 incompatible f-string (backslash in expression). Causes 4 test failures in `test_phase5.py`.
- 2 test failures in `test_phase4.py` due to Polars cgroup panic in containerized environments.
- Session config is hardcoded for US/Eastern equity index hours. Needs to be per-instrument.
