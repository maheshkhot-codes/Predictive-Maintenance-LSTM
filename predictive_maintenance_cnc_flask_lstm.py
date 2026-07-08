# predictive_maintenance_cnc_flask_lstm.py
"""
Flask app that uses LSTM model (if available) for ETA prediction.
If model files are missing, falls back to linear regression projection.
Run:
 python predictive_maintenance_cnc_flask_lstm.py
"""
import os
import json
from flask import Flask, request, render_template_string, redirect, url_for, send_file, flash
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO, StringIO

# ML libs
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import requests

# Optional TF model import if available
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace-with-a-secure-random-key")

# Expected model files
LSTM_MODEL_FILE = "lstm_cnc_model.h5"
SCALER_FILE = "scaler_cnc.pkl"

FEATURES = ["temperature", "vibration", "acoustic"]

# Simple HTML template (interactive Plotly disabled to keep file short)
HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Predictive Maintenance • LSTM</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body style="background:#f7fafc;color:#111">
<div class="container" style="max-width:1000px;padding:20px">
  <h3>Predictive Maintenance (LSTM capable)</h3>
  <form method="post" enctype="multipart/form-data">
    <div class="mb-2">
      <label>CSV file (timestamp,temperature,vibration,acoustic)</label>
      <input class="form-control" type="file" name="csvfile" accept=".csv" required>
    </div>
    <div class="row g-2">
      <div class="col-md-3"><label>Lookback</label><input class="form-control" type="number" name="lookback" value="60"></div>
      <div class="col-md-3"><label>Pred steps</label><input class="form-control" type="number" name="pred_steps" value="30"></div>
      <div class="col-md-3"><label>Horizon (days)</label><input class="form-control" type="number" name="horizon" value="7" step="0.5"></div>
      <div class="col-md-3 d-flex align-items-end"><button class="btn btn-primary w-100">Upload & Analyze</button></div>
    </div>
  </form>

  <div class="mt-3">
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for cat,msg in messages %}
          <div class="alert alert-{{cat}}">{{ msg|safe }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}
  </div>

  {% if table %}
    <h5>Results</h5>
    <table class="table table-striped">
      <thead><tr><th>Param</th><th>Latest</th><th>Rolling max</th><th>Threshold</th><th>ETA</th></tr></thead>
      <tbody>
        {% for r in table %}
        <tr>
          <td>{{ r.parameter }}</td>
          <td>{{ r.latest_value }}</td>
          <td>{{ r.rolling_max }}</td>
          <td>{{ r.threshold }}</td>
          <td>{{ r.readable_eta }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    <a class="btn btn-outline-secondary" href="{{ url_for('download_report') }}">Download CSV report</a>
  {% endif %}
</div>
</body>
</html>
"""

LAST_REPORT = None

# ---------------------------
# Utility / forecasting code
# ---------------------------
def load_and_prepare(file_stream):
    df = pd.read_csv(file_stream)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'timestamp' not in df.columns:
        raise ValueError("CSV must include 'timestamp' column")
    for c in FEATURES:
        if c not in df.columns:
            raise ValueError(f"CSV must include '{c}' column")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df[FEATURES] = df[FEATURES].ffill().bfill()
    # rolling max columns
    df['temp_rollmax'] = df['temperature'].cummax()
    df['vib_rollmax'] = df['vibration'].cummax()
    df['ac_rollmax'] = df['acoustic'].cummax()
    return df

def forecast_time_to_threshold_linear(ts_series, value_series, threshold, points):
    # linear regression approach (fallback)
    if value_series.iloc[-1] >= threshold:
        return 0.0
    n = min(points, len(value_series))
    x = (ts_series.iloc[-n:].astype('int64') // 10**9).values.reshape(-1,1)
    y = value_series.iloc[-n:].values
    if len(x) < 2:
        return float('inf')
    model = LinearRegression()
    model.fit(x, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    if slope <= 0:
        return float('inf')
    t_pred = (threshold - intercept) / slope
    last = x[-1,0]
    secs = t_pred - last
    days = secs / (3600*24)
    return max(days, 0.0)

# LSTM helper functions (if model exists)
def load_lstm_and_scaler():
    model = None
    scaler = None
    if os.path.exists(LSTM_MODEL_FILE) and os.path.exists(SCALER_FILE) and TF_AVAILABLE:
        try:
            model = load_model(LSTM_MODEL_FILE)
            with open(SCALER_FILE,'rb') as f:
                scaler = pickle.load(f)
            print("Loaded LSTM model and scaler.")
        except Exception as e:
            print("Failed to load LSTM model/scaler:", e)
            model = None
            scaler = None
    else:
        print("LSTM model or scaler not found or TF not available.")
    return model, scaler

def predict_future_multi(model, window_scaled, pred_steps, features):
    x = np.expand_dims(window_scaled, axis=0)  # (1, timesteps, features)
    y_flat = model.predict(x, verbose=0)[0]
    y_seq = y_flat.reshape((pred_steps, len(features)))
    return y_seq

def predict_eta_lstm_autoregressive(model, scaler, last_window_scaled, feature_names, timesteps, pred_steps,
                                    thresholds, sample_freq_seconds=60, max_blocks=48):
    """
    Return dict of ETA days per feature or None if not predicted in horizon.
    last_window_scaled: numpy array shape (timesteps, features) in scaled space
    """
    cur_window = last_window_scaled.copy()
    total_seconds = 0
    eta_seconds = {f: None for f in feature_names}
    for block in range(max_blocks):
        y_scaled = predict_future_multi(model, cur_window, pred_steps, feature_names)  # (pred_steps, features)
        # invert scaling per-step
        y_flat = np.array(y_scaled)  # scaled
        try:
            y_inv = scaler.inverse_transform(y_flat)
        except Exception:
            # if scaler expects 2D flat input, reshape accordingly
            y_inv = scaler.inverse_transform(y_flat)
        for step in range(pred_steps):
            total_seconds += sample_freq_seconds
            for i, fname in enumerate(feature_names):
                if eta_seconds[fname] is not None:
                    continue
                val = y_inv[step, i]
                if val >= thresholds.get(fname, float('inf')):
                    eta_seconds[fname] = total_seconds
        if all(v is not None for v in eta_seconds.values()):
            break
        # append predicted block (scaled) and trim
        cur_window = np.vstack([cur_window, y_scaled])
        cur_window = cur_window[-timesteps:, :]
    # convert to days
    eta_days = {k: (v/(3600*24) if v is not None else float('inf')) for k,v in eta_seconds.items()}
    return eta_days

# load LSTM once if available
LSTM_MODEL, SCALER = load_lstm_and_scaler()

@app.route('/', methods=['GET'])
def index():
    ctx = {
        'table': None
    }
    return render_template_string(HTML, **ctx)

@app.route('/', methods=['POST'])
def upload():
    global LAST_REPORT
    f = request.files.get('csvfile')
    if not f:
        flash("No file uploaded", "danger")
        return redirect(url_for('index'))
    try:
        lookback = int(request.form.get('lookback', 60))
        pred_steps = int(request.form.get('pred_steps', 30))
        horizon_days = float(request.form.get('horizon', 7.0))
    except Exception:
        flash("Invalid numeric inputs", "danger")
        return redirect(url_for('index'))

    try:
        df = load_and_prepare(f)
    except Exception as e:
        flash(f"Error reading CSV: {e}", "danger")
        return redirect(url_for('index'))

    # thresholds (defaults)
    thresholds = {'temperature':45.0, 'vibration':2.0, 'acoustic':20.0}

    # summary and rolling max
    summary = {
        'latest_temp': float(df['temperature'].iloc[-1]),
        'max_temp': float(df['temp_rollmax'].iloc[-1]),
        'latest_vib': float(df['vibration'].iloc[-1]),
        'max_vib': float(df['vib_rollmax'].iloc[-1]),
        'latest_ac': float(df['acoustic'].iloc[-1]),
        'max_ac': float(df['ac_rollmax'].iloc[-1]),
    }

    # choose LSTM if available and model dims match lookback/pred_steps
    eta_results = {}
    used_model = "linear"
    if LSTM_MODEL is not None and SCALER is not None:
        try:
            # scale full historical features and get last window
            scaled_all = SCALER.transform(df[FEATURES].values)
            if len(scaled_all) >= lookback:
                last_window = scaled_all[-lookback: , :]  # shape (lookback, features)
                eta_days = predict_eta_lstm_autoregressive(LSTM_MODEL, SCALER, last_window, FEATURES, lookback, pred_steps,
                                                           thresholds, sample_freq_seconds=60, max_blocks=48)
                eta_results = eta_days
                used_model = "lstm"
            else:
                # insufficient history: fallback to linear
                used_model = "linear"
        except Exception as e:
            print("LSTM prediction failed:", e)
            used_model = "linear"

    if used_model == "linear":
        # compute ETA per parameter using linear regression on rolling max
        eta_results = {}
        for name, rollcol, thr in [
            ('temperature', df['temp_rollmax'], thresholds['temperature']),
            ('vibration', df['vib_rollmax'], thresholds['vibration']),
            ('acoustic', df['ac_rollmax'], thresholds['acoustic']),
        ]:
            days = forecast_time_to_threshold_linear(df['timestamp'], rollcol, thr, lookback)
            eta_results[name] = days

    # prepare table and human-friendly eta
    table = []
    def human_days(d):
        if d == float('inf'):
            return "Not predicted"
        if d <= 0:
            return "Now"
        if d < 1:
            h = int(round(d*24))
            return f"~{h}h"
        return f"~{d:.1f}d"

    table.append({'parameter':'temperature','latest_value':summary['latest_temp'],'rolling_max':summary['max_temp'],'threshold':thresholds['temperature'],'readable_eta':human_days(eta_results.get('temperature', float('inf')))})
    table.append({'parameter':'vibration','latest_value':summary['latest_vib'],'rolling_max':summary['max_vib'],'threshold':thresholds['vibration'],'readable_eta':human_days(eta_results.get('vibration', float('inf')))})
    table.append({'parameter':'acoustic','latest_value':summary['latest_ac'],'rolling_max':summary['max_ac'],'threshold':thresholds['acoustic'],'readable_eta':human_days(eta_results.get('acoustic', float('inf')))})
    LAST_REPORT = "\n".join([",".join(["parameter","latest","rolling_max","threshold","eta"])] + [f"{r['parameter']},{r['latest_value']},{r['rolling_max']},{r['threshold']},{r['readable_eta']}" for r in table])

    flash(f"Analysis complete (model used: {used_model})", "success")
    return render_template_string(HTML, table=table)

@app.route('/download_report')
def download_report():
    global LAST_REPORT
    if not LAST_REPORT:
        flash("No report available. Run an analysis first.", "warning")
        return redirect(url_for('index'))
    buf = BytesIO()
    buf.write(LAST_REPORT.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='maintenance_report.csv')

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=8501)
