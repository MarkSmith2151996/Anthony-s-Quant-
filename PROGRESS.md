# BerjQuant Progress Tracker

## What This Project Is

**MCQ Engine** — A Monte Carlo Quantitative Trading Engine for NQ/MNQ futures, designed for prop firm evaluation accounts (Topstep, Apex, Lucid).

The engine generates, backtests, stress-tests, validates, and paper-trades intraday strategies using combinatorial signal composition and Monte Carlo simulation.

---

## Completed Phases

### Phase 1 — Core Engine ✅
- **Data Pipeline** (`engine/data_pipeline.py`): MotiveWave CSV ingestion, OHLCV cleaning, contract stitching, bar resampling (Polars)
- **Data Fetcher** (`engine/data_fetcher.py`): Yahoo Finance downloader for NQ/MNQ with chunked 1m data
- **Backtester** (`engine/backtester.py`): Vectorized signal computation + bar-by-bar execution with slippage, stops, EOD flatten
- **Risk Manager** (`engine/risk_manager.py`): 7 pre-trade checks — kill switch, session bounds, event blackout, daily loss, max DD, max contracts, EOD proximity
- **Metrics** (`engine/metrics.py`): Win rate, Sharpe, profit factor, max DD, consistency score, SQLite trade logger
- **Parallel Runner** (`engine/parallel_runner.py`): Multiprocessing batch runner with composite ranking
- **Config** (`config/`): Prop firm profiles (Topstep/Apex/Lucid), session times, FOMC/NFP/CPI event calendar
- **Tests**: 9 tests passing (test_engine.py)

### Phase 2 — Signal Library & Strategy Generator ✅
- **37 signals across 7 categories**:
  - Trend (6): EMA crossover, EMA slope, EMA ribbon, linear regression slope, Heikin-Ashi, Supertrend
  - Momentum (6): RSI, MACD, Stochastic, ROC, CCI, Williams %R
  - Volatility (6): ATR, Bollinger Bands, Keltner Channels, BB/KC Squeeze, ATR percentile, Historical Vol
  - Volume (5): VWAP, Volume Delta, Volume Profile, Relative Volume, Volume Climax
  - Order Flow (8): Delta divergence, Absorption, Exhaustion, Imbalance, Large trades, Delta momentum, Footprint imbalance, Trapped traders
  - Price Action (6): Session levels, Previous day levels, Range breakout, Pivot points, Opening range, Candle patterns
  - Time Filters (7): Time window, Day of week, Session segment, Minutes since open, First/Last N minutes, London overlap
- **Signal Registry** (`signals/registry.py`): Catalogs all signals with parameter search spaces, output columns, dependency graphs
- **Strategy Generator** (`strategies/generator.py`): Combinatorial engine — entries × filters × exit rules × sizing rules
- **Strategy Serializer** (`strategies/serializer.py`): JSON persistence for strategies and leaderboards
- **EMA Crossover Strategy** (`strategies/ema_crossover.py`): Hand-coded reference strategy
- **Tests**: 30 additional tests (test_phase2.py)

### Phase 3 — Monte Carlo Simulation ✅
- **Simulator** (`monte_carlo/simulator.py`): 10K bootstrap simulations with trade shuffling, slippage perturbation, parameter jitter, gap injection. Checks prop firm compliance per sim.
- **Scorer** (`monte_carlo/scoring.py`): 7-component scoring (Sharpe, PF, drawdown, consistency, pass rate, ruin, robustness) → composite score → letter grade (A+ through F)
- **Visualizer** (`monte_carlo/visualization.py`): Equity fan charts, drawdown/return/Sharpe distributions, radar plots, leaderboard comparison, session heatmap (Plotly + Matplotlib fallback)
- **Tests**: 15 additional tests (test_phase3.py)

### Phase 4 — Validation & Anti-Overfitting ✅
- **Walk-Forward Validator** (`validation/walk_forward.py`): Rolling train/test windows, WF efficiency ratio (OOS/IS Sharpe), overfitting detection (5 checks)
- **Regime Detector** (`validation/regime.py`): Classifies bars as trending/ranging/high-vol/breakout, measures strategy sensitivity across regimes
- **Correlation Analyzer** (`validation/correlation.py`): Pairwise strategy correlation, trade overlap, greedy diversified portfolio selection
- **Overfit Detector** (`validation/correlation.py`): Parameter sensitivity, time stability, IS/OOS degradation → overfit probability → verdict
- **Tests**: 9 additional tests (test_phase4.py)

### Phase 5 — Live/Paper Trading ✅
- **Signal Engine** (`live/signal_engine.py`): Real-time bar processing, multi-strategy confidence scoring, replay mode
- **Paper Trader** (`live/paper_trader.py`): Simulated fills, position management, kill switch, JSONL trade logging, export to JSON
- **Alert Manager** (`live/alerts.py`): Console, file, webhook (Slack/Discord), email channels with level filtering
- **Trading Dashboard** (`live/dashboard.py`): Terminal box-drawn status display + interactive HTML report with 6 Plotly charts
- **Tests**: 12 additional tests (test_phase5.py)

### Strategy Search Scripts ✅
- **`run_search.py`**: Brute-force MC search — single signals, combos, filters, mutations targeting $2K+/30 days
- **`run_search2.py`**: Walk-forward windowed search — 2yr data split into 30-day windows, requires monthly consistency
- **`run_search3.py`**: Deep full-period search — widest parameter sweep on full 2yr dataset
- **`run_top3.py`**: Top 3 winners deep-tested on 1h timeframe with full MC validation

### Deployment Setup ✅
- **`deploy/setup-vm.sh`**: One-command VM setup (Python, deps, venv, Claude Code)
- **`deploy/install-service.sh`**: Systemd service installation (auto-start on boot)
- **`deploy/berjquant.service`**: Systemd service template
- **`deploy/connect.sh`**: Quick SSH connect script
- **`deploy/Dockerfile`**: Container build
- **`deploy/docker-compose.yml`**: Docker Compose with persistent volumes
- **`deploy/DEPLOY.md`**: Step-by-step guide for Oracle Cloud & GCP free tier
- **GitHub Codespace**: Created and configured at `berjquant-jj756x9745qxc957`

---

## Test Status

**99 tests passing** across all phases:
- `test_engine.py` — 9 tests (data pipeline, risk manager, backtester)
- `test_phase2.py` — 51 tests (all signal categories, registry, generator, serializer)
- `test_phase3.py` — 15 tests (MC simulator, scorer, visualizer)
- `test_phase4.py` — 9 tests (walk-forward, regime, correlation, overfit detection)
- `test_phase5.py` — 15 tests (signal engine, paper trader, alerts, dashboard)

---

## Architecture

```
engine/          Core: data pipeline, backtester, risk manager, metrics, parallel runner
signals/         37 signals across 7 categories (trend, momentum, volatility, volume, orderflow, price action, time)
strategies/      EMA crossover reference + combinatorial generator + JSON serializer
monte_carlo/     MC simulator (10K sims) + 7-component scorer + Plotly visualizer
validation/      Walk-forward, regime detection, correlation analysis, overfit detection
live/            Signal engine, paper trader, alert manager, trading dashboard
config/          Prop firm rules, session times, economic event calendar
deploy/          VM setup, systemd service, Docker, Codespace config
data/            Raw → processed → features (Parquet)
tests/           99 tests across 5 test files
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Single backtest
python run_backtest.py

# Full pipeline (search → MC → validate → paper trade)
python run_pipeline.py --stage all --fetch

# Strategy searches
python run_search.py    # Brute-force 30-day search
python run_search2.py   # Walk-forward windowed search
python run_search3.py   # Deep full-period search
python run_top3.py      # Validate top 3 on hourly data
```

---

## Cloud Deployment

See `deploy/DEPLOY.md` for full instructions. Quick start:

```bash
# On a fresh Ubuntu VM:
git clone https://github.com/MarkSmith2151996/Anthony-s-Quant-.git
cd Anthony-s-Quant-
bash deploy/setup-vm.sh

# Or use the existing GitHub Codespace:
gh codespace ssh --codespace berjquant-jj756x9745qxc957
```

---

## What's Next

Potential next steps (not yet started):
- Real market data integration (Databento, Polygon, etc.)
- Live broker connection (NinjaTrader, TradeStation API)
- Portfolio-level strategy allocation
- Automated daily strategy search cron job
- Web dashboard for remote monitoring
