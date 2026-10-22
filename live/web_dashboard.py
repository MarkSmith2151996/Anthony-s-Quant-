"""TradingView-style web dashboard for live/paper trading monitoring.

Serves a single-page dashboard at http://localhost:8080 with:
- TradingView Lightweight Charts candlestick chart (dark theme)
- Real-time WebSocket updates for bars, trades, and account state
- Account panel, position panel, and trade log

Stack: FastAPI + WebSocket + lightweight-charts v4 from CDN (no npm).

Usage:
    dashboard = WebDashboard(symbol="MNQ", interval="15m", firm_name="Topstep 50K")
    dashboard.start(port=8080)

    # Push data from the trading loop:
    dashboard.push_bar(bar_dict)
    dashboard.push_trade(trade_dict)
    dashboard.push_account(account_dict)
"""

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ── HTML Template ────────────────────────────────────────────────

def _build_html(symbol: str, interval: str, firm_name: str) -> str:
    """Build the single-page dashboard HTML with inline CSS and JS."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MCQ Engine — {{symbol}} {{interval}}</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #131722;
    color: #d1d4dc;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
    overflow: hidden;
    height: 100vh;
}}

/* Top bar */
.topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    background: #1e222d;
    border-bottom: 1px solid #2a2e39;
    height: 40px;
}}
.topbar .symbol {{
    font-size: 18px;
    font-weight: 700;
    color: #fff;
}}
.topbar .interval {{
    font-size: 14px;
    color: #787b86;
    margin-left: 12px;
}}
.topbar .firm {{
    font-size: 13px;
    color: #787b86;
}}
.topbar .status {{
    display: flex;
    align-items: center;
    gap: 6px;
}}
.topbar .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #f44336;
}}
.topbar .dot.connected {{
    background: #26a69a;
}}

/* Chart container */
#chart-container {{
    width: 100%;
    height: calc(70vh - 40px);
}}

/* Bottom panels */
.panels {{
    display: flex;
    height: 30vh;
    border-top: 1px solid #2a2e39;
}}
.panel {{
    flex: 1;
    padding: 12px;
    border-right: 1px solid #2a2e39;
    overflow-y: auto;
}}
.panel:last-child {{
    border-right: none;
    flex: 1.5;
}}
.panel h3 {{
    color: #787b86;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}}

/* Account panel */
.metric {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #1e222d;
}}
.metric .label {{
    color: #787b86;
}}
.metric .value {{
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}}
.positive {{ color: #26a69a; }}
.negative {{ color: #ef5350; }}
.flat {{ color: #787b86; }}

/* Progress bars for limits */
.limit-bar {{
    margin: 4px 0;
}}
.limit-bar .bar-bg {{
    height: 4px;
    background: #2a2e39;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 2px;
}}
.limit-bar .bar-fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
}}
.bar-safe {{ background: #26a69a; }}
.bar-warning {{ background: #f59e0b; }}
.bar-danger {{ background: #ef5350; }}

/* Position panel */
.position-card {{
    background: #1e222d;
    border-radius: 6px;
    padding: 10px;
    margin-top: 6px;
}}
.pos-direction {{
    font-size: 16px;
    font-weight: 700;
}}
.pos-direction.long {{ color: #26a69a; }}
.pos-direction.short {{ color: #ef5350; }}

/* Trade log */
.trade-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}
.trade-table th {{
    color: #787b86;
    font-weight: 500;
    text-align: left;
    padding: 4px 6px;
    border-bottom: 1px solid #2a2e39;
    position: sticky;
    top: 0;
    background: #131722;
}}
.trade-table td {{
    padding: 4px 6px;
    border-bottom: 1px solid #1e222d;
    font-variant-numeric: tabular-nums;
}}
.trade-table tr:hover {{
    background: #1e222d;
}}
</style>
</head>
<body>

<!-- Top Bar -->
<div class="topbar">
    <div style="display:flex;align-items:center;">
        <span class="symbol">{symbol}</span>
        <span class="interval">{interval}</span>
    </div>
    <span class="firm">{firm_name}</span>
    <div class="status">
        <span id="status-text">Disconnected</span>
        <div class="dot" id="status-dot"></div>
    </div>
</div>

<!-- Chart -->
<div id="chart-container"></div>

<!-- Bottom Panels -->
<div class="panels">
    <!-- Account Panel -->
    <div class="panel" id="account-panel">
        <h3>Account</h3>
        <div class="metric">
            <span class="label">Balance</span>
            <span class="value" id="acct-balance">$0.00</span>
        </div>
        <div class="metric">
            <span class="label">Daily P&L</span>
            <span class="value" id="acct-daily-pnl">$0.00</span>
        </div>
        <div class="metric">
            <span class="label">Total P&L</span>
            <span class="value" id="acct-total-pnl">$0.00</span>
        </div>
        <div class="metric">
            <span class="label">Drawdown</span>
            <span class="value" id="acct-drawdown">$0.00</span>
        </div>
        <div class="metric">
            <span class="label">HWM</span>
            <span class="value" id="acct-hwm">$0.00</span>
        </div>
        <div class="metric">
            <span class="label">Kill Switch</span>
            <span class="value" id="acct-kill">OK</span>
        </div>
        <div class="limit-bar">
            <div style="display:flex;justify-content:space-between;">
                <span class="label">Daily Limit</span>
                <span class="value" id="daily-limit-pct">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill bar-safe" id="daily-limit-bar" style="width:0%"></div></div>
        </div>
        <div class="limit-bar">
            <div style="display:flex;justify-content:space-between;">
                <span class="label">Max Drawdown</span>
                <span class="value" id="dd-limit-pct">0%</span>
            </div>
            <div class="bar-bg"><div class="bar-fill bar-safe" id="dd-limit-bar" style="width:0%"></div></div>
        </div>
    </div>

    <!-- Position Panel -->
    <div class="panel" id="position-panel">
        <h3>Position</h3>
        <div id="position-content">
            <div class="position-card">
                <div class="pos-direction flat">FLAT</div>
            </div>
        </div>
    </div>

    <!-- Trade Log -->
    <div class="panel" id="tradelog-panel">
        <h3>Trade Log</h3>
        <table class="trade-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Dir</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Net P&L</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody id="trade-log-body">
            </tbody>
        </table>
    </div>
</div>

<script>
// ── Chart Setup ─────────────────────────────────────────────
const chartContainer = document.getElementById('chart-container');

const chart = LightweightCharts.createChart(chartContainer, {{
    layout: {{
        background: {{ type: 'solid', color: '#131722' }},
        textColor: '#d1d4dc',
    }},
    grid: {{
        vertLines: {{ color: '#1e222d' }},
        horzLines: {{ color: '#1e222d' }},
    }},
    crosshair: {{
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: {{ color: '#758696', width: 1, style: 3 }},
        horzLine: {{ color: '#758696', width: 1, style: 3 }},
    }},
    timeScale: {{
        borderColor: '#2a2e39',
        timeVisible: true,
        secondsVisible: false,
    }},
    rightPriceScale: {{
        borderColor: '#2a2e39',
    }},
}});

const candleSeries = chart.addCandlestickSeries({{
    upColor: '#26a69a',
    downColor: '#ef5350',
    borderUpColor: '#26a69a',
    borderDownColor: '#ef5350',
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
}});

const volumeSeries = chart.addHistogramSeries({{
    priceFormat: {{ type: 'volume' }},
    priceScaleId: 'volume',
}});

chart.priceScale('volume').applyOptions({{
    scaleMargins: {{ top: 0.85, bottom: 0 }},
}});

// Trade markers array
let markers = [];

// Resize handler
function resizeChart() {{
    chart.applyOptions({{
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
    }});
}}
window.addEventListener('resize', resizeChart);
resizeChart();

// ── WebSocket ───────────────────────────────────────────────
let ws = null;
let reconnectDelay = 1000;

function connect() {{
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(protocol + '//' + location.host + '/ws');

    ws.onopen = () => {{
        document.getElementById('status-dot').classList.add('connected');
        document.getElementById('status-text').textContent = 'Connected';
        reconnectDelay = 1000;
    }};

    ws.onclose = () => {{
        document.getElementById('status-dot').classList.remove('connected');
        document.getElementById('status-text').textContent = 'Reconnecting...';
        setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 30000);
    }};

    ws.onmessage = (event) => {{
        const msg = JSON.parse(event.data);
        switch (msg.type) {{
            case 'bar': handleBar(msg.data); break;
            case 'trade': handleTrade(msg.data); break;
            case 'account': handleAccount(msg.data); break;
            case 'signal': handleSignal(msg.data); break;
            case 'history': handleHistory(msg.data); break;
        }}
    }};
}}

// ── Handlers ────────────────────────────────────────────────
function handleBar(bar) {{
    candleSeries.update({{
        time: bar.time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
    }});
    volumeSeries.update({{
        time: bar.time,
        value: bar.volume,
        color: bar.close >= bar.open ? 'rgba(38,166,154,0.3)' : 'rgba(239,83,80,0.3)',
    }});
}}

function handleHistory(bars) {{
    const candles = bars.map(b => ({{
        time: b.time, open: b.open, high: b.high, low: b.low, close: b.close,
    }}));
    candleSeries.setData(candles);

    const volumes = bars.map(b => ({{
        time: b.time,
        value: b.volume,
        color: b.close >= b.open ? 'rgba(38,166,154,0.3)' : 'rgba(239,83,80,0.3)',
    }}));
    volumeSeries.setData(volumes);

    chart.timeScale().fitContent();
}}

function handleTrade(trade) {{
    // Add entry marker
    markers.push({{
        time: trade.entry_time,
        position: trade.direction === 'long' ? 'belowBar' : 'aboveBar',
        color: trade.direction === 'long' ? '#26a69a' : '#ef5350',
        shape: trade.direction === 'long' ? 'arrowUp' : 'arrowDown',
        text: trade.direction.toUpperCase() + ' ' + trade.entry_price.toFixed(2),
    }});

    // Add exit marker
    markers.push({{
        time: trade.exit_time,
        position: 'inBar',
        color: trade.net_pnl >= 0 ? '#26a69a' : '#ef5350',
        shape: 'circle',
        text: (trade.net_pnl >= 0 ? '+' : '') + '$' + trade.net_pnl.toFixed(2),
    }});

    // Sort markers by time and apply
    markers.sort((a, b) => a.time - b.time);
    candleSeries.setMarkers(markers);

    // Add to trade log
    addTradeRow(trade);
}}

function handleAccount(acct) {{
    const fmt = (v) => '$' + Math.abs(v).toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
    const sign = (v) => v >= 0 ? '+' + fmt(v) : '-' + fmt(v);

    setVal('acct-balance', fmt(acct.balance));
    setValPnl('acct-daily-pnl', acct.daily_pnl);
    setValPnl('acct-total-pnl', acct.balance - acct.starting_balance);
    setValPnl('acct-drawdown', acct.drawdown);
    setVal('acct-hwm', fmt(acct.high_water_mark));

    const killEl = document.getElementById('acct-kill');
    killEl.textContent = acct.is_killed ? 'ACTIVE' : 'OK';
    killEl.className = 'value ' + (acct.is_killed ? 'negative' : 'positive');

    // Daily limit bar
    if (acct.daily_loss_limit < 0) {{
        const pct = Math.min(100, Math.abs(acct.daily_pnl / acct.daily_loss_limit) * 100);
        setBar('daily-limit', pct);
    }}

    // Drawdown bar
    if (acct.max_drawdown < 0) {{
        const pct = Math.min(100, Math.abs(acct.drawdown / acct.max_drawdown) * 100);
        setBar('dd-limit', pct);
    }}

    // Position
    updatePosition(acct.open_position);
}}

function handleSignal(sig) {{
    // Optional: could add signal markers to chart
}}

// ── Helpers ─────────────────────────────────────────────────
function setVal(id, text) {{
    document.getElementById(id).textContent = text;
}}

function setValPnl(id, value) {{
    const el = document.getElementById(id);
    const prefix = value >= 0 ? '+$' : '-$';
    el.textContent = prefix + Math.abs(value).toLocaleString('en-US', {{
        minimumFractionDigits: 2, maximumFractionDigits: 2,
    }});
    el.className = 'value ' + (value > 0 ? 'positive' : value < 0 ? 'negative' : 'flat');
}}

function setBar(prefix, pct) {{
    const bar = document.getElementById(prefix + '-bar');
    const label = document.getElementById(prefix + '-pct');
    bar.style.width = pct.toFixed(0) + '%';
    label.textContent = pct.toFixed(0) + '%';
    bar.className = 'bar-fill ' + (pct > 80 ? 'bar-danger' : pct > 60 ? 'bar-warning' : 'bar-safe');
}}

function updatePosition(pos) {{
    const el = document.getElementById('position-content');
    if (!pos) {{
        el.innerHTML = '<div class="position-card"><div class="pos-direction flat">FLAT</div></div>';
        return;
    }}
    const upnl = pos.unrealized_pnl || 0;
    const upnlClass = upnl >= 0 ? 'positive' : 'negative';
    el.innerHTML = `
        <div class="position-card">
            <div class="pos-direction ${{pos.direction}}">${{pos.direction.toUpperCase()}} ${{pos.contracts}} @ ${{pos.entry_price.toFixed(2)}}</div>
            <div class="metric"><span class="label">Unrealized</span><span class="value ${{upnlClass}}">${{upnl >= 0 ? '+' : ''}}$${{upnl.toFixed(2)}}</span></div>
            ${{pos.stop_loss ? '<div class="metric"><span class="label">Stop Loss</span><span class="value">' + pos.stop_loss.toFixed(2) + '</span></div>' : ''}}
            ${{pos.take_profit ? '<div class="metric"><span class="label">Take Profit</span><span class="value">' + pos.take_profit.toFixed(2) + '</span></div>' : ''}}
        </div>
    `;
}}

function addTradeRow(trade) {{
    const tbody = document.getElementById('trade-log-body');
    const pnlClass = trade.net_pnl >= 0 ? 'positive' : 'negative';
    const dirClass = trade.direction === 'long' ? 'positive' : 'negative';

    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${{new Date(trade.exit_time * 1000).toLocaleTimeString()}}</td>
        <td class="${{dirClass}}">${{trade.direction.toUpperCase()}}</td>
        <td>${{trade.entry_price.toFixed(2)}}</td>
        <td>${{trade.exit_price.toFixed(2)}}</td>
        <td class="${{pnlClass}}">${{trade.net_pnl >= 0 ? '+' : ''}}$${{trade.net_pnl.toFixed(2)}}</td>
        <td>${{trade.exit_reason}}</td>
    `;
    tbody.insertBefore(row, tbody.firstChild);

    // Keep max 100 rows
    while (tbody.children.length > 100) {{
        tbody.removeChild(tbody.lastChild);
    }}
}}

// ── Init ────────────────────────────────────────────────────
connect();

// Load historical bars on page load
fetch('/api/history')
    .then(r => r.json())
    .then(bars => {{ if (bars.length) handleHistory(bars); }});
</script>
</body>
</html>"""


# ── WebDashboard Server ──────────────────────────────────────────

class WebDashboard:
    """FastAPI-based web dashboard with WebSocket push.

    Thread-safe: the push methods can be called from any thread.
    The dashboard runs in a background thread via ``start()``.
    """

    def __init__(
        self,
        symbol: str = "MNQ",
        interval: str = "15m",
        firm_name: str = "Topstep 50K",
    ):
        if not HAS_FASTAPI:
            raise ImportError(
                "FastAPI and uvicorn are required for the web dashboard. "
                "Install with: pip install fastapi uvicorn"
            )

        self.symbol = symbol
        self.interval = interval
        self.firm_name = firm_name

        # State
        self._bars: list[dict] = []
        self._trades: list[dict] = []
        self._account: dict = {}
        self._clients: list[WebSocket] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server_thread: threading.Thread | None = None

        # Build FastAPI app
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        """Construct the FastAPI application."""
        app = FastAPI(title="MCQ Trading Dashboard")
        html_content = _build_html(self.symbol, self.interval, self.firm_name)

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return html_content

        @app.get("/api/history")
        async def history():
            with self._lock:
                return JSONResponse(list(self._bars))

        @app.get("/api/status")
        async def status():
            with self._lock:
                return JSONResponse(dict(self._account))

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self._clients.append(ws)
            logger.info("WebSocket client connected (%d total)", len(self._clients))
            try:
                # Send current state on connect
                with self._lock:
                    if self._bars:
                        await ws.send_json({"type": "history", "data": self._bars})
                    if self._account:
                        await ws.send_json({"type": "account", "data": self._account})

                # Keep connection alive
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                if ws in self._clients:
                    self._clients.remove(ws)
                logger.info("WebSocket client disconnected (%d remaining)", len(self._clients))

        return app

    # ── Push methods (thread-safe) ───────────────────────────────

    def push_bar(self, bar: dict) -> None:
        """Push a new bar to all connected clients."""
        with self._lock:
            self._bars.append(bar)
            # Keep last 2000 bars
            if len(self._bars) > 2000:
                self._bars = self._bars[-2000:]

        self._broadcast({"type": "bar", "data": bar})

    def push_trade(self, trade: dict) -> None:
        """Push a completed trade to all connected clients."""
        with self._lock:
            self._trades.append(trade)

        self._broadcast({"type": "trade", "data": trade})

    def push_account(self, account: dict) -> None:
        """Push account state update to all connected clients."""
        with self._lock:
            self._account = account

        self._broadcast({"type": "account", "data": account})

    def push_signal(self, signal: dict) -> None:
        """Push a signal notification to all connected clients."""
        self._broadcast({"type": "signal", "data": signal})

    def set_history(self, bars: list[dict]) -> None:
        """Set initial historical bars for the chart."""
        with self._lock:
            self._bars = list(bars)

    # ── Broadcast ────────────────────────────────────────────────

    def _broadcast(self, message: dict) -> None:
        """Send a message to all connected WebSocket clients."""
        if not self._clients:
            return

        if self._loop is None:
            return

        asyncio.run_coroutine_threadsafe(
            self._async_broadcast(message),
            self._loop,
        )

    async def _async_broadcast(self, message: dict) -> None:
        """Async broadcast to all clients."""
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            if ws in self._clients:
                self._clients.remove(ws)

    # ── Server lifecycle ─────────────────────────────────────────

    def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the dashboard server in a background thread."""
        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="warning",
                loop="asyncio",
            )
            server = uvicorn.Server(config)
            self._loop.run_until_complete(server.serve())

        self._server_thread = threading.Thread(
            target=_run, daemon=True, name="web-dashboard",
        )
        self._server_thread.start()
        logger.info("Web dashboard started at http://%s:%d", host, port)

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("Web dashboard stopped")


# ── Standalone mode ──────────────────────────────────────────────

def _bar_to_tv(bar: dict) -> dict:
    """Convert an internal bar dict to TradingView format."""
    ts = bar.get("timestamp")
    if isinstance(ts, datetime):
        time_val = int(ts.timestamp())
    elif isinstance(ts, (int, float)):
        time_val = int(ts)
    else:
        time_val = 0

    return {
        "time": time_val,
        "open": bar["open"],
        "high": bar["high"],
        "low": bar["low"],
        "close": bar["close"],
        "volume": bar.get("volume", 0),
    }


def _trade_to_ws(trade) -> dict:
    """Convert a Trade object or dict to WebSocket format."""
    if hasattr(trade, "entry_time"):
        # Trade dataclass
        entry_time = trade.entry_time
        exit_time = trade.exit_time
        if isinstance(entry_time, datetime):
            entry_time = int(entry_time.timestamp())
        if isinstance(exit_time, datetime):
            exit_time = int(exit_time.timestamp())

        return {
            "trade_id": trade.trade_id,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "contracts": trade.contracts,
            "net_pnl": trade.net_pnl,
            "gross_pnl": trade.gross_pnl,
            "exit_reason": trade.exit_reason,
        }
    # Already a dict
    return trade


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCQ Web Dashboard (standalone)")
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--firm", default="Topstep 50K")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dashboard = WebDashboard(
        symbol=args.symbol,
        interval=args.interval,
        firm_name=args.firm,
    )

    print(f"Starting dashboard at http://localhost:{args.port}")
    print("Press Ctrl+C to stop")

    # Run in foreground for standalone mode
    uvicorn.run(dashboard.app, host="0.0.0.0", port=args.port, log_level="info")
