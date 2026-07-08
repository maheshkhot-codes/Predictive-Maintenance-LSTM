# train_lstm_cnc.py
"""
Train an LSTM multi-step model for CNC sensor forecasting.

Outputs:
 - lstm_cnc_model.h5
 - scaler_cnc.pkl

Usage:
 python train_lstm_cnc.py
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ---------------- USER CONFIG ----------------
INPUT_CSV = "cnc_log_threshold_breaking.csv"   # update if needed
FEATURES = ["temperature", "vibration", "acoustic"]
TIMESTEPS = 60           # how many past steps to use
PRED_STEPS = 30          # how many future steps predicted per block
SAMPLE_FREQ_SECONDS = 60 # seconds between rows (used later to convert steps -> time)
BATCH_SIZE = 64
EPOCHS = 200
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
MODEL_OUT = "lstm_cnc_model.h5"
SCALER_OUT = "scaler_cnc.pkl"
MAX_AUTOREG_BLOCKS = 48
# ---------------------------------------------

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'timestamp' not in df.columns:
        raise ValueError("CSV must include a 'timestamp' column")
    for c in FEATURES:
        if c not in df.columns:
            raise ValueError(f"CSV must include column: {c}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Optionally resample if timestamps irregular (uncomment and adjust)
    # df = df.set_index('timestamp').resample('1T').ffill().reset_index()

    df[FEATURES] = df[FEATURES].ffill().bfill()
    return df

def create_scaler_and_scale(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES].values)
    return scaler, scaled

def create_sequences(scaled, timesteps, pred_steps):
    X, y = [], []
    L = len(scaled)
    for i in range(0, L - timesteps - pred_steps + 1):
        X.append(scaled[i : i + timesteps])
        y.append(scaled[i + timesteps : i + timesteps + pred_steps])
    X = np.array(X)    # (samples, timesteps, features)
    y = np.array(y)    # (samples, pred_steps, features)
    return X, y

def reshape_target(y):
    # flatten target to (samples, pred_steps * features)
    s = y.shape
    return y.reshape((s[0], s[1]*s[2]))

def build_model(timesteps, features, pred_steps):
    inp = Input(shape=(timesteps, features))
    x = LSTM(128, return_sequences=True)(inp)
    x = LSTM(64)(x)
    x = Dense(pred_steps * features, activation='linear')(x)
    model = Model(inp, x)
    model.compile(optimizer=Adam(0.001), loss='mse')
    model.summary()
    return model

def train():
    print("Loading:", INPUT_CSV)
    df = load_and_prepare(INPUT_CSV)
    scaler, scaled = create_scaler_and_scale(df)
    X, y = create_sequences(scaled, TIMESTEPS, PRED_STEPS)
    y_flat = reshape_target(y)
    print("Samples:", len(X))

    # time-based split
    N = len(X)
    test_n = int(N * TEST_SPLIT)
    val_n = int(N * VALID_SPLIT)
    train_end = N - val_n - test_n

    X_train, y_train = X[:train_end], y_flat[:train_end]
    X_val, y_val = X[train_end:train_end+val_n], y_flat[train_end:train_end+val_n]
    X_test, y_test = X[train_end+val_n:], y_flat[train_end+val_n:]

    print("Train/Val/Test:", len(X_train), len(X_val), len(X_test))

    model = build_model(TIMESTEPS, len(FEATURES), PRED_STEPS)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_OUT, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    print("Saving model & scaler...")
    model.save(MODEL_OUT)
    with open(SCALER_OUT, 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved:", MODEL_OUT, SCALER_OUT)

    # Eval
    loss = model.evaluate(X_test, y_test, verbose=0)
    print("Test MSE:", loss)

if __name__ == "__main__":
    train()
