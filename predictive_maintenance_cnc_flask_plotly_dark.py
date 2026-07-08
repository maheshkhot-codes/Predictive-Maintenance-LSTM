# predictive_maintenance_cnc_flask_plotly_dark.py
# Single-file Flask app with:
# - Dark / Light mode toggle
# - Dashboard-style cards
# - Interactive Plotly charts (time-series)
# - CSV upload (timestamp,temperature,vibration,acoustic)
# - Rolling-max + linear-projection forecast
# - Telegram alerting (optional via form or env vars)
# - Downloadable maintenance report
#
# Dependencies:
# pip install flask pandas numpy scikit-learn matplotlib requests plotly

import os
from flask import Flask, request, render_template_string, redirect, url_for, send_file, flash
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO, StringIO
import base64
from datetime import datetime
import requests

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace-with-a-secure-random-key")

# ----------------------------
# HTML template (Plotly + Dark)
# ----------------------------
HTML = """
<!doctype html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8">
  <title>Predictive Maintenance • CNC</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Bootstrap -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
  <!-- Plotly -->
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

  <style>
    :root {
      --bg-light: #f6f8fb;
      --card-radius: 12px;
      --accent: linear-gradient(90deg,#6a11cb,#2575fc);
    }
    html[data-theme='dark'] {
      --bg: #0b1221;
      --surface: #0f1724;
      --muted: #9aa4b2;
      color-scheme: dark;
    }
    html[data-theme='light'] {
      --bg: #f6f8fb;
      --surface: #ffffff;
      --muted: #6c757d;
    }

    body {
      background: var(--bg, #f6f8fb);
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      -webkit-font-smoothing:antialiased;
      -moz-osx-font-smoothing:grayscale;
      padding-bottom:40px;
      color: var(--muted);
    }
    .container-small { max-width:1200px; margin:auto; padding-top:22px; }
    .hero {
      background: var(--accent);
      color: white;
      padding:22px; border-radius:14px;
      box-shadow: 0 10px 30px rgba(37,117,252,0.12);
      margin-bottom:18px;
    }
    .card-pretty { border-radius: var(--card-radius); box-shadow: 0 6px 18px rgba(2,6,23,0.06); background: var(--surface, #fff); color: inherit; }
    .kpi { padding:12px; border-radius:10px; }
    .badge-eta { background: linear-gradient(90deg,#22c55e,#06b6d4); color:white; padding:6px 10px; border-radius:999px; font-weight:600; }
    .param-good { background:#e9fbf0; color:#065f46; padding:8px 10px; border-radius:8px; display:inline-block; }
    .param-warn { background:#fff7ed; color:#92400e; padding:8px 10px; border-radius:8px; display:inline-block; }
    .param-danger { background:#fff1f2; color:#9f1239; padding:8px 10px; border-radius:8px; display:inline-block; }
    .pulse { width:10px; height:10px; border-radius:50%; background:#dc2626; display:inline-block; box-shadow:0 0 8px rgba(220,38,38,0.35); margin-right:6px; }
    footer { opacity:0.8; margin-top:18px; color:var(--muted); }

    /* dark-mode adjustments */
    html[data-theme='dark'] .card-pretty { background: #071226; box-shadow: 0 6px 18px rgba(2,6,23,0.6); color:#d1d9e6; }
    html[data-theme='dark'] body { color: #d1d9e6; }
    html[data-theme='dark'] .kpi { background: rgba(255,255,255,0.02); }
    html[data-theme='dark'] .hero { box-shadow: 0 10px 30px rgba(0,0,0,0.6); }
    html[data-theme='dark'] .table thead { background: rgba(255,255,255,0.02); }
  </style>
</head>
<body>
  <div class="container container-small">

    <div class="d-flex align-items-center justify-content-between mb-3">
      <div class="hero flex-grow-1 me-3">
        <h2 class="mb-0"><i class="bi bi-gear-fill"></i> Predictive Maintenance — CNC</h2>
        <small class="text-white-50">Dashboard</small>
      </div>

      <div class="d-flex gap-2 align-items-center">
        <div class="form-check form-switch">
          <input class="form-check-input" type="checkbox" id="darkToggle">
          <label class="form-check-label" for="darkToggle">Dark mode</label>
        </div>
        <a class="btn btn-outline-secondary" href="{{ url_for('index') }}">Reset</a>
      </div>
    </div>

    <!-- Upload & settings -->
    <div class="card card-pretty p-3 mb-3">
      <form method="post" enctype="multipart/form-data" action="{{ url_for('upload') }}">
        <div class="row g-2">
          <div class="col-md-5">
            <label class="form-label">CSV file <small class="text-muted">(timestamp,temperature,vibration,acoustic)</small></label>
            <input class="form-control" type="file" name="csvfile" accept=".csv" required>
          </div>
          <div class="col-md-2">
            <label class="form-label">Lookback</label>
            <input class="form-control" type="number" name="lookback" value="{{ lookback }}" min="3" max="1000">
          </div>
          <div class="col-md-2">
            <label class="form-label">Alert horizon (days)</label>
            <input class="form-control" type="number" name="horizon" value="{{ horizon }}" step="0.5">
          </div>
          <div class="col-md-3 d-flex align-items-end">
            <button class="btn btn-primary w-100" type="submit"><i class="bi bi-upload"></i> Upload & Analyze</button>
          </div>
        </div>

        <hr>

        <div class="row g-2">
          <div class="col-md-4"><label class="form-label">Temperature threshold (°C)</label><input class="form-control" name="threshold_temp" type="number" step="0.1" value="{{ threshold_temp }}"></div>
          <div class="col-md-4"><label class="form-label">Vibration threshold (G)</label><input class="form-control" name="threshold_vib" type="number" step="0.01" value="{{ threshold_vib }}"></div>
          <div class="col-md-4"><label class="form-label">Acoustic threshold (dB)</label><input class="form-control" name="threshold_ac" type="number" step="0.1" value="{{ threshold_ac }}"></div>
        </div>

        <hr>

        <div class="row g-2">
          <div class="col-md-6"><label class="form-label">Telegram Bot Token (optional)</label><input class="form-control" type="text" name="bot_token" placeholder="Token or env var"></div>
          <div class="col-md-4"><label class="form-label">Telegram Chat ID (optional)</label><input class="form-control" type="text" name="chat_id" placeholder="Chat ID or env var"></div>
          <div class="col-md-2 d-flex align-items-end"><a class="btn btn-outline-secondary w-100" href="{{ url_for('download_report') }}"><i class="bi bi-download"></i> Report</a></div>
        </div>
      </form>
    </div>

    <!-- Flash -->
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for cat, msg in messages %}
          <div class="alert alert-{{ cat }} alert-dismissible fade show" role="alert">
            {{ msg|safe }}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
          </div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    {% if summary %}
    <!-- Dashboard cards -->
    <div class="row g-3 mb-3">
      <div class="col-md-3">
        <div class="card card-pretty p-3 kpi">
          <small class="text-muted">Latest Temp</small>
          <div class="d-flex align-items-center justify-content-between">
            <div><h3 class="mb-0">{{ summary.latest_temp }} °C</h3><small class="text-muted">Rolling max: {{ summary.max_temp }}</small></div>
            <div><span class="{{ summary.temp_style }}"><i class="bi bi-thermometer-half"></i> Status</span></div>
          </div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card card-pretty p-3 kpi">
          <small class="text-muted">Latest Vib</small>
          <div class="d-flex align-items-center justify-content-between">
            <div><h3 class="mb-0">{{ summary.latest_vib }} G</h3><small class="text-muted">Rolling max: {{ summary.max_vib }}</small></div>
            <div><span class="{{ summary.vib_style }}"><i class="bi bi-sliders"></i> Status</span></div>
          </div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card card-pretty p-3 kpi">
          <small class="text-muted">Latest Acoustic</small>
          <div class="d-flex align-items-center justify-content-between">
            <div><h3 class="mb-0">{{ summary.latest_ac }} dB</h3><small class="text-muted">Rolling max: {{ summary.max_ac }}</small></div>
            <div><span class="{{ summary.ac_style }}"><i class="bi bi-megaphone"></i> Status</span></div>
          </div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card card-pretty p-3 kpi">
          <small class="text-muted">Maintenance ETA</small>
          <div class="d-flex align-items-center justify-content-between">
            <div><h4 class="mb-0">{{ maintenance_eta }}</h4><small class="text-muted">Nearest ETA</small></div>
            <div><span class="badge-eta">{{ nearest_eta_badge }}</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Immediate alerts -->
    {% if immediate_alerts %}
      <div class="card card-pretty p-3 mb-3">
        <div class="d-flex align-items-center mb-2">
          <div class="pulse"></div>
          <h5 class="mb-0">Immediate Alerts</h5>
        </div>
        <ul>
          {% for a in immediate_alerts %}<li><strong>{{ a }}</strong></li>{% endfor %}
        </ul>
      </div>
    {% endif %}

    <!-- Forecast table + interactive charts -->
    <div class="row g-3 mb-3">
      <div class="col-md-6">
        <div class="card card-pretty p-3">
          <h5>Predictive Forecast</h5>
          <div class="table-responsive mt-2">
            <table class="table table-sm">
              <thead class="table-light"><tr><th>Parameter</th><th>Latest</th><th>Rolling Max</th><th>Threshold</th><th>ETA</th></tr></thead>
              <tbody>
                {% for r in table %}
                  <tr class="{% if r.alert_level=='danger' %}table-danger{% elif r.alert_level=='warn' %}table-warning{% else %}table-success{% endif %}">
                    <td class="text-capitalize fw-bold">{{ r.parameter }}</td>
                    <td>{{ r.latest_value }}</td>
                    <td>{{ r.rolling_max }}</td>
                    <td>{{ r.threshold }}</td>
                    <td><span class="badge-eta">{{ r.readable_eta }}</span></td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
          {% if pred_alerts %}
            <div class="alert alert-warning mt-2"><strong>Predictive:</strong><ul class="mb-0">{% for p in pred_alerts %}<li>{{ p }}</li>{% endfor %}</ul></div>
          {% endif %}
        </div>
      </div>

      <div class="col-md-6">
        <div class="card card-pretty p-3">
          <h5>Interactive sensor charts</h5>
          <div id="plotly-chart" style="height:420px;"></div>
        </div>
      </div>
    </div>

    <div class="d-flex gap-2 mb-4">
      <a class="btn btn-outline-dark" href="{{ url_for('download_report') }}"><i class="bi bi-download"></i> Download report</a>
      <a class="btn btn-outline-secondary" href="{{ url_for('index') }}"><i class="bi bi-arrow-clockwise"></i> Analyze another file</a>
    </div>

    {% endif %}

    <footer class="text-center">
      <small class="text-muted">Dashboard • Dark mode • Interactive charts</small>
    </footer>
  </div>

  <script>
    // dark mode toggle using localStorage
    const toggle = document.getElementById('darkToggle');
    function applyTheme(theme) {
      document.documentElement.setAttribute('data-theme', theme);
      toggle.checked = (theme === 'dark');
    }
    // init from storage or system
    const saved = localStorage.getItem('pm_theme');
    if (saved) applyTheme(saved);
    else {
      const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      applyTheme(prefersDark ? 'dark' : 'light');
    }
    toggle.addEventListener('change', () => {
      const theme = toggle.checked ? 'dark' : 'light';
      applyTheme(theme);
      localStorage.setItem('pm_theme', theme);
    });

    // Plotly: build interactive plot from server-provided arrays
    {% if df_json %}
    (function(){
      const timestamps = {{ df_json.timestamps | tojson }};
      const temperature = {{ df_json.temperature | tojson }};
      const vibration = {{ df_json.vibration | tojson }};
      const acoustic = {{ df_json.acoustic | tojson }};
      const temp_roll = {{ df_json.temp_rollmax | tojson }};
      const vib_roll = {{ df_json.vib_rollmax | tojson }};
      const ac_roll = {{ df_json.ac_rollmax | tojson }};
      const threshold_temp = {{ threshold_temp }};
      const threshold_vib = {{ threshold_vib }};
      const threshold_ac = {{ threshold_ac }};

      const traces = [
        { x: timestamps, y: temperature, name: 'Temperature', mode: 'lines', line:{width:2} },
        { x: timestamps, y: temp_roll, name: 'Temp (rollmax)', mode:'lines', line:{dash:'dash'} },
        { x: timestamps, y: vibration, name: 'Vibration', mode: 'lines', visible:'legendonly' },
        { x: timestamps, y: vib_roll, name: 'Vib (rollmax)', mode:'lines', visible:'legendonly', line:{dash:'dash'} },
        { x: timestamps, y: acoustic, name: 'Acoustic', mode:'lines', visible:'legendonly' },
        { x: timestamps, y: ac_roll, name: 'Ac (rollmax)', mode:'lines', visible:'legendonly', line:{dash:'dash'} }
      ];

      const shapes = [
        { type:'line', xref:'paper', x0:0, x1:1, yref:'y', y0:threshold_temp, y1:threshold_temp, line:{color:'rgba(255,99,71,0.7)', dash:'dot'} , yaxis:'y'},
      ];

      const layout = {
        margin:{t:30,b:40,l:60,r:30},
        legend:{orientation:'h'},
        xaxis:{type:'date', title:''},
        yaxis:{title:'Values'},
        shapes: [
          { type:'line', xref:'paper', x0:0, x1:1, yref:'y', y0:threshold_temp, y1:threshold_temp, line:{color:'rgba(255,99,71,0.7)', dash:'dot'} },
          { type:'line', xref:'paper', x0:0, x1:1, yref:'y2', y0:threshold_vib, y1:threshold_vib, line:{color:'rgba(255,165,0,0.6)', dash:'dot'} },
          { type:'line', xref:'paper', x0:0, x1:1, yref:'y3', y0:threshold_ac, y1:threshold_ac, line:{color:'rgba(30,144,255,0.6)', dash:'dot'} }
        ],
        grid:{rows:3, columns:1},
      };

      // Because we use mixed metrics, create subplots with shared x
      const dataTemp = [
        { x: timestamps, y: temperature, name: 'Temperature (°C)', xaxis:'x', yaxis:'y', type:'scatter' },
        { x: timestamps, y: temp_roll, name: 'Temp (rollmax)', xaxis:'x', yaxis:'y', type:'scatter', line:{dash:'dash'} }
      ];
      const dataVib = [
        { x: timestamps, y: vibration, name:'Vibration (G)', xaxis:'x', yaxis:'y2', type:'scatter' },
        { x: timestamps, y: vib_roll, name:'Vib (rollmax)', xaxis:'x', yaxis:'y2', type:'scatter', line:{dash:'dash'} }
      ];
      const dataAc = [
        { x: timestamps, y: acoustic, name:'Acoustic (dB)', xaxis:'x', yaxis:'y3', type:'scatter' },
        { x: timestamps, y: ac_roll, name:'Ac (rollmax)', xaxis:'x', yaxis:'y3', type:'scatter', line:{dash:'dash'} }
      ];

      const layoutSub = {
        grid: {rows:3, columns:1, pattern:'independent'},
        margin: {t:20, b:30, l:50, r:20},
        xaxis: {type:'date'},
        xaxis2: {type:'date'},
        xaxis3: {type:'date'},
        yaxis: {title:'°C'},
        yaxis2: {title:'G'},
        yaxis3: {title:'dB'},
        legend: {orientation:'h', x:0, y:-0.2}
      };

      const dataAll = dataTemp.concat(dataVib).concat(dataAc);
      Plotly.newPlot('plotly-chart', dataAll, layoutSub, {responsive:true});
    })();
    {% endif %}
  </script>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# -------------------------
# Application logic (same ideas as before)
# -------------------------
LAST_REPORT_CSV = None

def load_and_prepare(file_stream):
    df = pd.read_csv(file_stream)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'timestamp' not in df.columns:
        raise ValueError("CSV must include a 'timestamp' column")
    for col in ['temperature', 'vibration', 'acoustic']:
        if col not in df.columns:
            raise ValueError(f"CSV must include '{col}' column")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['temp_rollmax'] = df['temperature'].cummax()
    df['vib_rollmax'] = df['vibration'].cummax()
    df['ac_rollmax'] = df['acoustic'].cummax()
    return df

def forecast_time_to_threshold(ts_series, value_series, threshold, points):
    # returns days until threshold (float), float('inf') if not predicted
    if value_series.iloc[-1] >= threshold:
        return 0.0
    n = min(points, len(value_series))
    x_ts = ts_series.iloc[-n:].astype('int64') // 10**9
    y = value_series.iloc[-n:].values
    X = x_ts.values.reshape(-1, 1)
    if len(X) < 2:
        return float('inf')
    model = LinearRegression()
    model.fit(X, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    if slope <= 0:
        return float('inf')
    t_pred = (threshold - intercept) / slope
    last_ts_sec = x_ts.iloc[-1]
    seconds_until = t_pred - last_ts_sec
    days_until = seconds_until / (3600 * 24)
    if days_until < 0:
        return 0.0
    return days_until

def human_friendly_days(days):
    if days == float('inf'):
        return "Not predicted"
    if days <= 0:
        return "Now"
    if days < 1:
        hours = int(round(days * 24))
        return f"~{hours}h"
    return f"~{days:.1f}d"

def send_telegram(bot_token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
        return (resp.status_code == 200, resp.text)
    except Exception as e:
        return (False, f"exception: {e}")

@app.route('/', methods=['GET'])
def index():
    context = {
        'threshold_temp': 45.0,
        'threshold_vib': 2.0,
        'threshold_ac': 20.0,
        'lookback': 20,
        'horizon': 7.0,
        'summary': None,
        'df_json': None,
        'maintenance_eta': '—',
        'nearest_eta_badge': '—'
    }
    return render_template_string(HTML, **context)

@app.route('/', methods=['POST'])
def upload():
    global LAST_REPORT_CSV
    f = request.files.get('csvfile')
    if not f:
        flash("No file uploaded", "danger")
        return redirect(url_for('index'))

    try:
        threshold_temp = float(request.form.get('threshold_temp', 45.0))
        threshold_vib = float(request.form.get('threshold_vib', 2.0))
        threshold_ac = float(request.form.get('threshold_ac', 20.0))
        lookback = int(request.form.get('lookback', 20))
        horizon = float(request.form.get('horizon', 7.0))
        bot_token = (request.form.get('bot_token') or "").strip() or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id_val = (request.form.get('chat_id') or "").strip() or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    except Exception:
        flash("Invalid numeric inputs", "danger")
        return redirect(url_for('index'))

    try:
        df = load_and_prepare(f)
    except Exception as e:
        flash(f"Error reading CSV: {e}", "danger")
        return redirect(url_for('index'))

    summary = {
        'latest_temp': round(float(df['temperature'].iloc[-1]), 3),
        'max_temp': round(float(df['temp_rollmax'].iloc[-1]), 3),
        'latest_vib': round(float(df['vibration'].iloc[-1]), 3),
        'max_vib': round(float(df['vib_rollmax'].iloc[-1]), 3),
        'latest_ac': round(float(df['acoustic'].iloc[-1]), 3),
        'max_ac': round(float(df['ac_rollmax'].iloc[-1]), 3),
    }

    def style_for(value, threshold):
        if value >= threshold:
            return "param-danger"
        if value >= 0.9*threshold:
            return "param-warn"
        return "param-good"

    summary['temp_style'] = style_for(summary['max_temp'], threshold_temp)
    summary['vib_style'] = style_for(summary['max_vib'], threshold_vib)
    summary['ac_style'] = style_for(summary['max_ac'], threshold_ac)

    immediate_alerts = []
    if summary['max_temp'] >= threshold_temp:
        immediate_alerts.append(f"Temperature rolling max {summary['max_temp']}°C >= threshold {threshold_temp}°C")
    if summary['max_vib'] >= threshold_vib:
        immediate_alerts.append(f"Vibration rolling max {summary['max_vib']}G >= threshold {threshold_vib}G")
    if summary['max_ac'] >= threshold_ac:
        immediate_alerts.append(f"Acoustic rolling max {summary['max_ac']}dB >= threshold {threshold_ac}dB")

    days_temp = forecast_time_to_threshold(df['timestamp'], df['temp_rollmax'], threshold_temp, lookback)
    days_vib = forecast_time_to_threshold(df['timestamp'], df['vib_rollmax'], threshold_vib, lookback)
    days_ac = forecast_time_to_threshold(df['timestamp'], df['ac_rollmax'], threshold_ac, lookback)

    def row_level(latest, rollmax, threshold, days):
        if rollmax >= threshold or days == 0.0:
            return 'danger'
        if days != float('inf') and days <= horizon:
            return 'warn'
        return 'good'

    table = []
    etas = []
    for param, latest, rollmax, thr, days in [
        ('temperature', summary['latest_temp'], summary['max_temp'], threshold_temp, days_temp),
        ('vibration', summary['latest_vib'], summary['max_vib'], threshold_vib, days_vib),
        ('acoustic', summary['latest_ac'], summary['max_ac'], threshold_ac, days_ac),
    ]:
        table.append({
            'parameter': param,
            'latest_value': latest,
            'rolling_max': rollmax,
            'threshold': thr,
            'predicted_days_until_threshold': (round(days, 3) if days != float('inf') else "inf"),
            'readable_eta': human_friendly_days(days),
            'alert_level': row_level(latest, rollmax, thr, days)
        })
        etas.append(days)

    # nearest ETA (smallest non-inf positive days)
    finite_etas = [d for d in etas if d != float('inf')]
    nearest = None
    if finite_etas:
        nearest = min(finite_etas)
        nearest_str = human_friendly_days(nearest)
    else:
        nearest_str = "Not predicted"

    # prepare plotly-friendly JSON arrays
    df_json = {
        'timestamps': df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist(),
        'temperature': df['temperature'].tolist(),
        'vibration': df['vibration'].tolist(),
        'acoustic': df['acoustic'].tolist(),
        'temp_rollmax': df['temp_rollmax'].tolist(),
        'vib_rollmax': df['vib_rollmax'].tolist(),
        'ac_rollmax': df['ac_rollmax'].tolist()
    }

    # create downloadable report (table)
    report_df = pd.DataFrame(table)
    report_buf = StringIO()
    report_df.to_csv(report_buf, index=False)
    report_buf.seek(0)
    LAST_REPORT_CSV = report_buf.getvalue()

    # Telegram: friendly messaging only (no raw JSON)
    if bot_token and chat_id_val and (immediate_alerts or [p for p in table if p['alert_level']!='good']):
        now = datetime.utcnow().isoformat() + "Z"
        lines = [f"Predictive Maintenance Alert",
                 f"Time: {now}", ""]
        lines.append("Sensors:")
        lines.append(f"Temperature: {summary['latest_temp']} °C (max {summary['max_temp']})")
        lines.append(f"Vibration: {summary['latest_vib']} G (max {summary['max_vib']})")
        lines.append(f"Acoustic: {summary['latest_ac']} dB (max {summary['max_ac']})")
        lines.append("")
        if immediate_alerts:
            lines.append("Immediate Alerts:")
            for a in immediate_alerts:
                lines.append(f"- {a}")
            lines.append("")
        # ETA summary
        lines.append("ETA to threshold:")
        lines.append(f"- Temperature: {human_friendly_days(days_temp)}")
        lines.append(f"- Vibration: {human_friendly_days(days_vib)}")
        lines.append(f"- Acoustic: {human_friendly_days(days_ac)}")
        lines.append("")
        lines.append("Recommendation: schedule maintenance before ETA to avoid breakdown.")
        message = "\n".join(lines)

        ok, info = send_telegram(bot_token, chat_id_val, message)
        if ok:
            flash("Telegram alert sent.", "success")
        else:
            # try to parse info for friendly message
            try:
                import json
                j = json.loads(info)
                reason = f"{j.get('error_code')} {j.get('description')}" if isinstance(j, dict) and not j.get('ok', True) else info
            except Exception:
                reason = info
            flash(f"Telegram failed: {reason}", "warning")

    # render page with interactive data
    context = {
        'threshold_temp': threshold_temp,
        'threshold_vib': threshold_vib,
        'threshold_ac': threshold_ac,
        'lookback': lookback,
        'horizon': horizon,
        'summary': summary,
        'immediate_alerts': immediate_alerts,
        'table': table,
        'pred_alerts': [],  # messages are in table/alerts
        'plots': None,
        'df_json': df_json,
        'maintenance_eta': nearest_str,
        'nearest_eta_badge': nearest_str
    }
    return render_template_string(HTML, **context)

@app.route('/download_report')
def download_report():
    global LAST_REPORT_CSV
    if not LAST_REPORT_CSV:
        flash("No report available. Upload and analyze a CSV first.", "warning")
        return redirect(url_for('index'))
    buf = BytesIO()
    buf.write(LAST_REPORT_CSV.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="maintenance_report.csv")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8501)