# BerjQuant Forward Test — Setup Guide

HTF Swing v3 forward test on **LucidFlex 150K**. Two scripts, no server.

---

## LucidFlex 150K Rules

| Rule | Value |
|------|-------|
| Account size | $150,000 |
| Daily loss limit | **None** |
| Max Loss Limit | -$4,500 (EOD trailing) |
| Drawdown checks | End-of-day only (4:45 PM close) |
| Eval profit target | $9,000 |
| Eval consistency | Best day ≤ 50% of total profit |
| Funded consistency | **None** |
| Flatten time | 4:45 PM ET |
| Profit split | 90/10 |
| Platforms | Tradovate, Rithmic, NinjaTrader |

---

## Setup

### 1. Create project

```bash
mkdir -p ~/berjquant/data ~/berjquant/logs
cd ~/berjquant
python3 -m venv .venv
source .venv/bin/activate
pip install numpy polars databento python-dotenv
```

### 2. Add your Databento API key

```bash
echo 'DATABENTO_API_KEY=db-your-key-here' > ~/berjquant/.env
```

Get a free key at [databento.com](https://databento.com). MNQ daily pulls
cost ~$0.02/day.

### 3. Place the scripts

Put both files in `~/berjquant/`:

```
~/berjquant/
├── .env              # API key
├── fetch_bars.py     # Pulls MNQ 15m bars from Databento
├── ingest.py         # Runs strategies, tracks trades in SQLite
├── data/             # Downloaded CSVs land here
└── forward_test.db   # Auto-created on first run
```

---

## Daily Workflow

### After market close (~5 PM ET):

```bash
cd ~/berjquant && source .venv/bin/activate

# 1. Fetch today's bars
python fetch_bars.py

# 2. Feed them to the engine
python ingest.py feed data/mnq_15m_2026-03-25.csv

# 3. Check status
python ingest.py status
```

### First time — backfill 2-4 weeks:

```bash
python fetch_bars.py --backfill 30
python ingest.py feed data/mnq_15m_backfill.csv
python ingest.py status
```

---

## Commands

```bash
python ingest.py feed <file>            # Ingest bars
python ingest.py status                  # Balance, PnL, MLL check
python ingest.py trades                  # Last 7 days of trades
python ingest.py trades --days 30        # Last 30 days
python ingest.py trades --strategy RSI   # Filter by strategy
python ingest.py performance             # Monthly breakdown
python ingest.py compare                 # Live vs backtest table
python ingest.py reset --confirm RESET   # Wipe everything
```

---

## Checking In with Claude Code

Just ask Claude Code to look at the database directly:

> "Run `python ~/berjquant/ingest.py compare` and tell me how the
> forward test is tracking against backtest expectations."

Or for deeper analysis:

> "Open ~/berjquant/forward_test.db and analyze the last month of
> trades. Check if slippage assumptions are holding, if the first-hour
> edge is still dominant, and whether any strategy is underperforming."

Claude Code can run Python and query SQLite directly — no MCP server needed.

---

## Automate (Optional)

### Cron job — auto-fetch after market close

```bash
crontab -e
```

Add (runs 5:30 PM ET weekdays):

```
30 17 * * 1-5 cd ~/berjquant && .venv/bin/python fetch_bars.py >> logs/fetch.log 2>&1
```

### Auto-fetch AND ingest

```bash
cat > ~/berjquant/daily.sh << 'EOF'
#!/bin/bash
cd ~/berjquant && source .venv/bin/activate
python fetch_bars.py
LATEST=$(ls -t data/mnq_15m_*.csv 2>/dev/null | head -1)
[ -n "$LATEST" ] && python ingest.py feed "$LATEST"
python ingest.py status
EOF
chmod +x ~/berjquant/daily.sh
```

```
30 17 * * 1-5 ~/berjquant/daily.sh >> ~/berjquant/logs/daily.log 2>&1
```

---

## Strategy Parameters

| Strategy | Signal | SL | TP | Max Hold | R:R |
|----------|--------|----|----|----------|-----|
| RSI(7,70,30) | RSI < 30 → long, > 70 → short | 15 pts | 100 pts | 5 bars | 6.67 |
| IB Breakout | Break above/below first-30-min range | 20 pts | 120 pts | 10 bars | 6.00 |
| Momentum | Big bar + volume + trend alignment | 15 pts | 100 pts | 5 bars | 6.67 |

3 contracts per strategy (9 MNQ total). $11.34 round-trip cost per trade.
Session: 9:30 AM – 4:45 PM ET. Flatten at 4:45 PM.

---

## What to Watch For

After 4-6 weeks, the numbers that matter:

| Metric | Backtest Y2 OOS | Worry If |
|--------|----------------|----------|
| Monthly PnL | ~$7,186 | Below $3,000 consistently |
| Win rate | ~26% | Below 20% or above 35% |
| R:R ratio | 3.3–3.8 | Below 2.5 |
| Max DD | -$1,927 | Beyond -$3,000 |
| Trades/day | ~7 | Below 4 or above 12 |
| First-hour % of PnL | ~69% | Below 40% |

If live numbers diverge more than ~30% from backtest for 2+ weeks straight,
something structural changed — investigate before continuing.
