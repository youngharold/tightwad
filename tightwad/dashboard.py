"""Live web dashboard for the speculative decoding proxy.

Supports both SSE (``/v1/tightwad/events``) and WebSocket
(``/v1/tightwad/ws``) for real-time updates.  SSE is the default
(simpler, works through more proxies).  WebSocket enables bidirectional
communication for interactive dashboard features.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

logger = logging.getLogger("tightwad.dashboard")


def _get_proxy():
    from .proxy import _get_proxy as gp
    return gp()


async def handle_dashboard(request: Request):
    return HTMLResponse(DASHBOARD_HTML)


async def handle_history(request: Request):
    proxy = _get_proxy()
    records = [asdict(r) for r in proxy.stats.request_history]
    return JSONResponse(records)


async def handle_events(request: Request):
    async def event_stream():
        last_history_len = 0
        while True:
            proxy = _get_proxy()
            stats = proxy.stats

            # Stats event
            stats_data = {
                "total_rounds": stats.total_rounds,
                "total_drafted": stats.total_drafted,
                "total_accepted": stats.total_accepted,
                "total_bonus": stats.total_bonus,
                "total_resampled": stats.total_resampled,
                "total_tokens_output": stats.total_tokens_output,
                "acceptance_rate": round(stats.acceptance_rate, 3),
                "effective_tokens_per_round": round(stats.effective_tokens_per_round, 2),
                "uptime_seconds": round(stats.uptime_seconds, 1),
                "drafter_wins": stats.drafter_wins,
            }
            yield f"event: stats\ndata: {json.dumps(stats_data)}\n\n"

            # Health event
            health_data = {"servers": []}
            if proxy._multi_drafter:
                for endpoint, _ in proxy.draft_clients:
                    h = await proxy.check_server(endpoint.url, endpoint.backend)
                    health_data["servers"].append({
                        "role": "draft",
                        "url": endpoint.url,
                        "model": endpoint.model_name,
                        "backend": endpoint.backend,
                        "alive": h.get("alive", False),
                        "wins": stats.drafter_wins.get(endpoint.url, 0),
                    })
            else:
                h = await proxy.check_server(proxy.config.draft.url, proxy.config.draft.backend)
                health_data["servers"].append({
                    "role": "draft",
                    "url": proxy.config.draft.url,
                    "model": proxy.config.draft.model_name,
                    "backend": proxy.config.draft.backend,
                    "alive": h.get("alive", False),
                })
            th = await proxy.check_server(proxy.config.target.url, proxy.config.target.backend)
            health_data["servers"].append({
                "role": "target",
                "url": proxy.config.target.url,
                "model": proxy.config.target.model_name,
                "backend": proxy.config.target.backend,
                "alive": th.get("alive", False),
            })
            yield f"event: health\ndata: {json.dumps(health_data)}\n\n"

            # New requests since last check
            current_len = len(stats.request_history)
            if current_len > last_history_len:
                new_records = stats.request_history[last_history_len:]
                records_data = [asdict(r) for r in new_records]
                yield f"event: requests\ndata: {json.dumps(records_data)}\n\n"
                last_history_len = current_len

            await asyncio.sleep(2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def handle_websocket(websocket: WebSocket):
    """WebSocket endpoint for bidirectional dashboard communication.

    Sends the same stats/health/requests events as SSE, but also accepts
    commands from the client (e.g. adjusting max_draft_tokens).

    Outbound messages: ``{"event": "stats"|"health"|"requests", "data": {...}}``
    Inbound commands:  ``{"command": "set_draft_tokens", "value": 32}``
    """
    await websocket.accept()
    last_history_len = 0

    try:
        while True:
            proxy = _get_proxy()
            stats = proxy.stats

            # Send stats
            stats_data = {
                "total_rounds": stats.total_rounds,
                "total_drafted": stats.total_drafted,
                "total_accepted": stats.total_accepted,
                "total_bonus": stats.total_bonus,
                "total_resampled": stats.total_resampled,
                "total_tokens_output": stats.total_tokens_output,
                "acceptance_rate": round(stats.acceptance_rate, 3),
                "effective_tokens_per_round": round(stats.effective_tokens_per_round, 2),
                "uptime_seconds": round(stats.uptime_seconds, 1),
                "drafter_wins": stats.drafter_wins,
                "consensus_accepted": stats.consensus_accepted,
                "consensus_fallback": stats.consensus_fallback,
                "current_draft_tokens": proxy.draft_n,
                "auto_draft_tokens": proxy.config.auto_draft_tokens,
            }
            await websocket.send_json({"event": "stats", "data": stats_data})

            # Send health
            health_data = {"servers": []}
            if proxy._multi_drafter:
                for endpoint, _ in proxy.draft_clients:
                    h = await proxy.check_server(endpoint.url, endpoint.backend)
                    health_data["servers"].append({
                        "role": "draft", "url": endpoint.url,
                        "model": endpoint.model_name, "alive": h.get("alive", False),
                    })
            else:
                h = await proxy.check_server(proxy.config.draft.url, proxy.config.draft.backend)
                health_data["servers"].append({
                    "role": "draft", "url": proxy.config.draft.url,
                    "model": proxy.config.draft.model_name,
                    "alive": h.get("alive", False),
                })
            th = await proxy.check_server(proxy.config.target.url, proxy.config.target.backend)
            health_data["servers"].append({
                "role": "target", "url": proxy.config.target.url,
                "model": proxy.config.target.model_name,
                "alive": th.get("alive", False),
            })
            await websocket.send_json({"event": "health", "data": health_data})

            # Send new requests
            current_len = len(stats.request_history)
            if current_len > last_history_len:
                new_records = stats.request_history[last_history_len:]
                records_data = [asdict(r) for r in new_records]
                await websocket.send_json({"event": "requests", "data": records_data})
                last_history_len = current_len

            # Check for inbound commands (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                _handle_ws_command(proxy, msg)
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as exc:
        logger.debug("WebSocket error: %s", exc)


def _handle_ws_command(proxy, msg: dict) -> None:
    """Process an inbound WebSocket command."""
    command = msg.get("command")
    if command == "set_draft_tokens":
        value = msg.get("value")
        if isinstance(value, int) and 1 <= value <= 256:
            if proxy._adaptive:
                proxy._adaptive.current = value
                logger.info("WebSocket: set draft tokens to %d", value)
            else:
                proxy.config.max_draft_tokens = value
                logger.info("WebSocket: set draft tokens to %d (fixed mode)", value)


DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Tightwad Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;min-height:100vh}
a{color:#e94560;text-decoration:none}
a:hover{text-decoration:underline}

#header{padding:14px 24px;background:#16213e;border-bottom:1px solid #0f3460;display:flex;align-items:center;gap:16px}
#header h1{font-size:20px;color:#e94560;font-weight:700}
#header .subtitle{font-size:13px;color:#888}
#header .uptime{font-size:13px;color:#4ecca3;margin-left:auto}
#header .nav{margin-left:12px}
#header .nav a{font-size:13px;color:#888;padding:4px 10px;border:1px solid #0f3460;border-radius:6px}
#header .nav a:hover{color:#e94560;border-color:#e94560;text-decoration:none}

.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:20px 24px}
@media(max-width:800px){.grid{grid-template-columns:1fr}}

.card{background:#16213e;border:1px solid #0f3460;border-radius:10px;padding:16px 20px}
.card h2{font-size:14px;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}

.server-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:14px}
.dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.dot.alive{background:#4ecca3}
.dot.dead{background:#e94560}
.server-label{color:#ccc}
.server-meta{color:#666;font-size:12px}
.wins-badge{background:#0f3460;color:#4ecca3;font-size:11px;padding:2px 8px;border-radius:10px;margin-left:auto}

.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.stat-item{text-align:center;padding:8px;background:#1a1a2e;border-radius:8px}
.stat-value{font-size:22px;font-weight:700;color:#fff}
.stat-label{font-size:11px;color:#888;margin-top:2px}

.chart-container{margin-top:12px;position:relative;height:120px}
.chart-container svg{width:100%;height:100%}
.chart-legend{display:flex;gap:16px;margin-top:6px;font-size:11px;color:#888}
.chart-legend span{display:flex;align-items:center;gap:4px}
.chart-legend .swatch{width:10px;height:3px;border-radius:2px}

.full-width{grid-column:1/-1}
#log-table{width:100%;border-collapse:collapse;font-size:13px}
#log-table th{text-align:left;padding:8px 10px;color:#888;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;border-bottom:1px solid #0f3460}
#log-table td{padding:6px 10px;border-bottom:1px solid #0f3460}
#log-table tr:hover{background:#1a1a2e}
.rate-high{color:#4ecca3}
.rate-mid{color:#f0c040}
.rate-low{color:#e94560}
.time-cell{color:#888;font-size:12px}
.ms-cell{font-family:monospace;font-size:12px;color:#aaa}
.empty-log{text-align:center;padding:32px;color:#555}
</style></head><body>
<div id="header">
  <h1>TIGHTWAD</h1>
  <span class="subtitle">Speculative Decoding Dashboard</span>
  <span class="nav"><a href="/">Chat UI</a></span>
  <span class="uptime" id="uptime">--</span>
</div>

<div class="grid">
  <div class="card" id="health-card">
    <h2>Server Health</h2>
    <div id="health-list"><div class="server-row"><span class="server-label" style="color:#555">Connecting...</span></div></div>
  </div>

  <div class="card">
    <h2>Speculation Stats</h2>
    <div class="stat-grid">
      <div class="stat-item"><div class="stat-value" id="s-rate">--</div><div class="stat-label">Acceptance Rate</div></div>
      <div class="stat-item"><div class="stat-value" id="s-tpr">--</div><div class="stat-label">Tokens/Round</div></div>
      <div class="stat-item"><div class="stat-value" id="s-rounds">--</div><div class="stat-label">Rounds</div></div>
      <div class="stat-item"><div class="stat-value" id="s-tokens">--</div><div class="stat-label">Tokens Output</div></div>
    </div>
    <div class="chart-container">
      <svg id="chart" viewBox="0 0 400 120" preserveAspectRatio="none">
        <polyline id="line-rate" fill="none" stroke="#4ecca3" stroke-width="2" points=""/>
        <polyline id="line-tpr" fill="none" stroke="#5b8dee" stroke-width="2" points=""/>
      </svg>
    </div>
    <div class="chart-legend">
      <span><span class="swatch" style="background:#4ecca3"></span> Acceptance %</span>
      <span><span class="swatch" style="background:#5b8dee"></span> Tokens/Round</span>
    </div>
  </div>

  <div class="card full-width">
    <h2>Request Log</h2>
    <div style="overflow-x:auto">
    <table id="log-table">
      <thead><tr>
        <th>Time</th><th>Rounds</th><th>Drafted</th><th>Accepted</th><th>Rate</th>
        <th>Draft ms</th><th>Verify ms</th><th>Total ms</th><th>Tokens</th><th>Model</th>
      </tr></thead>
      <tbody id="log-body">
        <tr><td colspan="10" class="empty-log">No requests yet. Send a message via the <a href="/">Chat UI</a> or API.</td></tr>
      </tbody>
    </table>
    </div>
  </div>
</div>

<script>
var ratePoints = [];
var tprPoints = [];
var MAX_POINTS = 60;
var logCount = 0;
var MAX_LOG = 50;

function formatUptime(s) {
  if (s < 60) return Math.round(s) + 's';
  if (s < 3600) return Math.round(s/60) + 'm';
  var h = Math.floor(s/3600), m = Math.round((s%3600)/60);
  return h + 'h ' + m + 'm';
}

function formatTime(ts) {
  var d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

function rateClass(r) {
  if (r >= 0.7) return 'rate-high';
  if (r >= 0.4) return 'rate-mid';
  return 'rate-low';
}

function updateChart() {
  var w = 400, h = 120, pad = 2;
  function toPoints(arr, maxVal) {
    if (!arr.length) return '';
    var pts = [];
    for (var i = 0; i < arr.length; i++) {
      var x = arr.length === 1 ? w/2 : (i / (arr.length - 1)) * (w - pad*2) + pad;
      var y = h - pad - (arr[i] / maxVal) * (h - pad*2);
      pts.push(x.toFixed(1) + ',' + y.toFixed(1));
    }
    return pts.join(' ');
  }
  document.getElementById('line-rate').setAttribute('points', toPoints(ratePoints, 100));
  var maxTpr = Math.max(10, Math.max.apply(null, tprPoints.length ? tprPoints : [10]));
  document.getElementById('line-tpr').setAttribute('points', toPoints(tprPoints, maxTpr));
}

function updateHealth(data) {
  var el = document.getElementById('health-list');
  // Clear existing content safely (no dynamic data involved).
  while (el.firstChild) { el.removeChild(el.firstChild); }

  data.servers.forEach(function(s) {
    // Build the row entirely with DOM methods so server-supplied strings
    // (role, model, backend, url) are treated as plain text and cannot
    // inject HTML or execute scripts.
    var row = document.createElement('div');
    row.className = 'server-row';

    var dot = document.createElement('span');
    dot.className = 'dot ' + (s.alive ? 'alive' : 'dead');
    row.appendChild(dot);

    var role = s.role.charAt(0).toUpperCase() + s.role.slice(1);
    var label = document.createElement('span');
    label.className = 'server-label';
    label.textContent = role + ': ' + s.model;  // textContent auto-escapes
    row.appendChild(label);

    var meta = document.createElement('span');
    meta.className = 'server-meta';
    meta.textContent = s.backend + ' @ ' + s.url;  // textContent auto-escapes
    row.appendChild(meta);

    if (s.wins !== undefined && s.wins > 0) {
      var badge = document.createElement('span');
      badge.className = 'wins-badge';
      badge.textContent = s.wins;  // numeric, but textContent is still safest
      row.appendChild(badge);
    }

    el.appendChild(row);
  });
}

function updateStats(data) {
  document.getElementById('s-rate').textContent = (data.acceptance_rate * 100).toFixed(1) + '%';
  document.getElementById('s-tpr').textContent = data.effective_tokens_per_round.toFixed(1);
  document.getElementById('s-rounds').textContent = data.total_rounds;
  document.getElementById('s-tokens').textContent = data.total_tokens_output;
  document.getElementById('uptime').textContent = 'uptime: ' + formatUptime(data.uptime_seconds);

  ratePoints.push(data.acceptance_rate * 100);
  tprPoints.push(data.effective_tokens_per_round);
  if (ratePoints.length > MAX_POINTS) { ratePoints.shift(); tprPoints.shift(); }
  updateChart();
}

/**
 * Append one request-record row to the log table.
 *
 * All cell values come from server-supplied JSON, so each <td> is
 * populated with textContent (never innerHTML) to prevent XSS.
 * The only exception is clearing the placeholder row on first use,
 * which is done safely via body.innerHTML = '' (no dynamic data).
 */
function addLogRow(r) {
  var body = document.getElementById('log-body');
  // Remove the static "No requests yet" placeholder on first real row.
  if (logCount === 0) { body.innerHTML = ''; }

  var rc = rateClass(r.acceptance_rate);

  // Helper: create a <td> with a CSS class and plain-text content.
  function td(cls, text) {
    var cell = document.createElement('td');
    if (cls) { cell.className = cls; }
    cell.textContent = text;
    return cell;
  }

  var tr = document.createElement('tr');
  // Numeric fields (timestamp, rounds, drafted, …) are safe to coerce via
  // toFixed/toString, but we still use textContent as the insertion method
  // so that a compromised SSE payload cannot break out into raw HTML.
  tr.appendChild(td('time-cell', formatTime(r.timestamp)));
  tr.appendChild(td('', r.rounds));
  tr.appendChild(td('', r.drafted));
  tr.appendChild(td('', r.accepted));
  tr.appendChild(td(rc, (r.acceptance_rate * 100).toFixed(1) + '%'));
  tr.appendChild(td('ms-cell', r.draft_ms.toFixed(0)));
  tr.appendChild(td('ms-cell', r.verify_ms.toFixed(0)));
  tr.appendChild(td('ms-cell', r.total_ms.toFixed(0)));
  tr.appendChild(td('', r.tokens_output));
  // r.model is a server-supplied string — textContent prevents injection.
  tr.appendChild(td('server-meta', r.model));

  body.insertBefore(tr, body.firstChild);
  logCount++;
  while (body.children.length > MAX_LOG) { body.removeChild(body.lastChild); }
}

// Load initial history
fetch('/v1/tightwad/history').then(function(r){ return r.json(); }).then(function(records){
  records.forEach(function(r){ addLogRow(r); });
}).catch(function(){});

// SSE connection
var es = new EventSource('/v1/tightwad/events');
es.addEventListener('stats', function(e) { updateStats(JSON.parse(e.data)); });
es.addEventListener('health', function(e) { updateHealth(JSON.parse(e.data)); });
es.addEventListener('requests', function(e) {
  JSON.parse(e.data).forEach(function(r){ addLogRow(r); });
});
es.onerror = function() {
  document.getElementById('uptime').textContent = 'reconnecting...';
  document.getElementById('uptime').style.color = '#e94560';
};
es.onopen = function() {
  document.getElementById('uptime').style.color = '#4ecca3';
};
</script>
</body></html>"""
