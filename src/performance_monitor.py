
import threading
import time
from collections import deque

import threading
import time
from collections import deque

import psutil
from flask import Flask, jsonify, request, render_template_string
from voice_stats import get_voice_stats

app = Flask(__name__)

@app.route("/api/voice-stats")
def api_voice_stats():
    return jsonify(get_voice_stats())
_lock = threading.Lock()
_started_at = time.time()

_stats = {
    "fps": 0.0,
    "latency_ms": 0.0,
    "frames": 0,
    "left_clicks": 0,
    "right_clicks": 0,
    "double_clicks": 0,
    "accidental_clicks": 0,
    "cpu_percent": 0.0,
    "memory_percent": 0.0,
    "calibration_samples": 0,
    "calibration_error_px": None,
    "last_gesture": "Waiting for webcam",
}

_frame_times = deque(maxlen=30)
_dashboard_started = False


def record_frame(frame_seconds, latency_ms, gesture=None):
    """Record one processed webcam frame."""
    with _lock:
        _stats["frames"] += 1
        _frame_times.append(frame_seconds)

        average_frame_time = (
            sum(_frame_times) / len(_frame_times)
            if _frame_times else 0
        )
        _stats["fps"] = (
            1.0 / average_frame_time if average_frame_time > 0 else 0.0
        )
        _stats["latency_ms"] = max(0.0, latency_ms)

        if gesture is not None:
            _stats["last_gesture"] = str(gesture)


def record_click(kind):
    """Record a click triggered by the gesture controller."""
    key = {
        "left": "left_clicks",
        "right": "right_clicks",
        "double": "double_clicks",
    }.get(kind)

    if key:
        with _lock:
            _stats[key] += 1


def mark_last_click_accidental():
    """Label one previously recorded click as accidental."""
    with _lock:
        total = (
            _stats["left_clicks"]
            + _stats["right_clicks"]
            + _stats["double_clicks"]
        )
        if _stats["accidental_clicks"] < total:
            _stats["accidental_clicks"] += 1
            return True
        return False


def record_calibration_error(target_x, target_y, actual_x, actual_y):
    """Record pixel error for one manually measured calibration point."""
    import math

    error = math.sqrt(
        (float(target_x) - float(actual_x)) ** 2
        + (float(target_y) - float(actual_y)) ** 2
    )

    with _lock:
        previous_count = _stats["calibration_samples"]
        previous_error = _stats["calibration_error_px"]

        if previous_count == 0 or previous_error is None:
            _stats["calibration_error_px"] = error
        else:
            _stats["calibration_error_px"] = (
                previous_error * previous_count + error
            ) / (previous_count + 1)

        _stats["calibration_samples"] += 1

    return error


def get_stats():
    with _lock:
        result = dict(_stats)

    result["successful_clicks"] = max(
        0,
        result["left_clicks"]
        + result["right_clicks"]
        + result["double_clicks"]
        - result["accidental_clicks"],
    )
    result["uptime_seconds"] = int(time.time() - _started_at)
    result["cpu_percent"] = psutil.cpu_percent(interval=None)
    result["memory_percent"] = psutil.virtual_memory().percent
    return result


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Virtual Mouse Performance</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 28px;
  background: #0b1020; color: #e9eefb;
  font: 15px/1.5 Segoe UI, Arial, sans-serif;
}
main { max-width: 1150px; margin: auto; }
header { display:flex; justify-content:space-between; gap:16px; align-items:center; flex-wrap:wrap; }
h1 { margin:0; font-size:28px; }
.sub { color:#9ba9c7; margin:6px 0 22px; }
.status { color:#82e6bb; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; }
.card, section {
  background:#131b30; border:1px solid #263451;
  border-radius:14px; padding:18px;
}
.label { color:#9ba9c7; font-size:13px; }
.value { font-size:30px; font-weight:700; margin:7px 0; }
.note { color:#9ba9c7; font-size:12px; }
section { margin-top:16px; }
h2 { font-size:18px; margin:0 0 12px; }
button {
  background:#5978f4; border:0; color:white; padding:10px 13px;
  border-radius:8px; cursor:pointer; margin:5px 4px 5px 0;
}
button.secondary { background:#34415e; }
input {
  width:105px; background:#0b1020; color:white;
  border:1px solid #34415e; padding:9px; border-radius:7px;
  margin:4px;
}
label { color:#c7d1e7; font-size:13px; }
.message { color:#82e6bb; min-height:22px; }
.meter { height:8px; background:#27344e; border-radius:10px; overflow:hidden; }
.meter span { display:block; height:100%; width:0%; background:#7d9bff; }
footer { color:#8190ad; font-size:12px; margin-top:18px; }
</style>
</head>
<body>
<main>
<header>
  <div>
    <h1>Virtual Mouse Performance</h1>
    <p class="sub">Local live monitoring · Gesture-Controlled Virtual Mouse</p>
  </div>
  <div class="status" id="connection">Connecting…</div>
</header>

<div class="grid">
  <div class="card"><div class="label">Processing FPS</div><div class="value" id="fps">—</div><div class="note">Recent 30-frame average</div></div>
  <div class="card"><div class="label">Gesture processing latency</div><div class="value" id="latency">—</div><div class="note">Webcam processing through control handling</div></div>
  <div class="card"><div class="label">Successful clicks</div><div class="value" id="successful">0</div><div class="note">Recorded clicks minus clicks marked accidental</div></div>
  <div class="card"><div class="label">Accidental clicks</div><div class="value" id="accidental">0</div><button class="secondary" onclick="markAccidental()">Mark one click accidental</button></div>
  <div class="card"><div class="label">CPU usage</div><div class="value" id="cpu">—</div><div class="meter"><span id="cpuBar"></span></div></div>
  <div class="card"><div class="label">Memory usage</div><div class="value" id="memory">—</div><div class="meter"><span id="memoryBar"></span></div></div>
</div>

<section>
  <h2>Click counts</h2>
  <p>Left-clicks: <b id="left">0</b> · Right-clicks: <b id="right">0</b> · Double-clicks: <b id="double">0</b></p>
  <p>Last recognized gesture: <b id="gesture">Waiting for webcam</b></p>
  <p>Frames processed: <b id="frames">0</b> · Uptime: <b id="uptime">0s</b></p>
</section>

<section>
  <h2>Cursor calibration error</h2>
  <p class="note">Enter screen-pixel coordinates for a known target and the cursor's measured position at the time of the test. The dashboard calculates Euclidean pixel error and averages recorded samples. These values are manually supplied, not inferred automatically.</p>
  <div>
    <label>Target X <input id="tx" type="number" min="0" value="500"></label>
    <label>Target Y <input id="ty" type="number" min="0" value="400"></label>
  </div>
  <div>
    <label>Actual X <input id="ax" type="number" min="0" value="500"></label>
    <label>Actual Y <input id="ay" type="number" min="0" value="400"></label>
  </div>
  <button onclick="recordCalibration()">Record calibration sample</button>
  <button class="secondary" onclick="resetCalibration()">Reset calibration results</button>
  <p>Average error: <b id="error">Not measured</b> · Samples: <b id="samples">0</b></p>
  <p class="message" id="message"></p>
</section>

<footer>Dashboard binds to localhost only. Metrics are kept in memory and reset when the Python process restarts.</footer>
</main>
<script>
async function refresh() {
  try {
    const response = await fetch('/api/metrics');
    if (!response.ok) throw new Error('Metrics endpoint unavailable');
    const d = await response.json();
    document.getElementById('connection').textContent = '● Connected';
    for (const [id, value] of Object.entries({
      fps: d.fps.toFixed(1),
      latency: d.latency_ms.toFixed(1) + ' ms',
      successful: d.successful_clicks,
      accidental: d.accidental_clicks,
      cpu: d.cpu_percent.toFixed(1) + '%',
      memory: d.memory_percent.toFixed(1) + '%',
      left: d.left_clicks, right: d.right_clicks, double: d.double_clicks,
      gesture: d.last_gesture, frames: d.frames, samples: d.calibration_samples,
      uptime: d.uptime_seconds + 's'
    })) document.getElementById(id).textContent = value;
    document.getElementById('cpuBar').style.width = d.cpu_percent + '%';
    document.getElementById('memoryBar').style.width = d.memory_percent + '%';
    document.getElementById('error').textContent =
      d.calibration_error_px === null ? 'Not measured' : d.calibration_error_px.toFixed(1) + ' px';
  } catch (e) {
    document.getElementById('connection').textContent = '● Disconnected — start the virtual mouse';
  }
}
async function post(url, body={}) {
  const r = await fetch(url, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)
  });
  const d = await r.json();
  document.getElementById('message').textContent = d.message || d.error || '';
  await refresh();
}
function markAccidental() { post('/api/accidental'); }
function recordCalibration() {
  const val = id => Number(document.getElementById(id).value);
  post('/api/calibration', {tx:val('tx'), ty:val('ty'), ax:val('ax'), ay:val('ay')});
}
function resetCalibration() { post('/api/calibration/reset'); }
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/api/metrics")
def metrics_endpoint():
    return jsonify(get_stats())


@app.post("/api/accidental")
def accidental_endpoint():
    if mark_last_click_accidental():
        return jsonify(message="One click marked accidental.")
    return jsonify(message="No unlabelled click is available yet.")


@app.post("/api/calibration")
def calibration_endpoint():
    data = request.get_json(silent=True) or {}
    try:
        error = record_calibration_error(
            data["tx"], data["ty"], data["ax"], data["ay"]
        )
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Enter valid numeric target and actual coordinates."), 400
    return jsonify(message=f"Calibration sample recorded: {error:.1f} px error.")


@app.post("/api/calibration/reset")
def reset_calibration_endpoint():
    with _lock:
        _stats["calibration_samples"] = 0
        _stats["calibration_error_px"] = None
    return jsonify(message="Calibration results reset.")


def start_dashboard():
    """Start Flask once in a background thread."""
    global _dashboard_started
    if _dashboard_started:
        return
    _dashboard_started = True

    thread = threading.Thread(
        target=lambda: app.run(
            host="127.0.0.1", port=5000, debug=False, use_reloader=False
        ),
        daemon=True,
    )
    thread.start()
