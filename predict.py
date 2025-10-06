# predict.py
"""
Phase 3: Traffic / Latency Predictor
------------------------------------
Reads 'network_data.csv' (from capture.py), aggregates packets into
time windows, trains an LSTM model to predict next-window total_bytes,
and evaluates performance. Saves model to 'model.keras' and plots to 'plots/'.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime

# ---------------- CONFIG ----------------
CSV_FILE = "network_data.csv"
AGG_SECONDS = 10
SEQ_LEN = 6
TARGET_COL = "total_bytes"
MODEL_FILE = "model.keras"
PLOTS_DIR = "plots"
EPOCHS = 50
BATCH_SIZE = 16
# ----------------------------------------

os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 1️⃣ Load and preprocess data ---
print(f"📂 Loading: {CSV_FILE}")
if not os.path.exists(CSV_FILE):
    raise SystemExit("❌ Error: network_data.csv not found. Please run capture.py first.")

df = pd.read_csv(CSV_FILE)
if "timestamp" not in df.columns:
    raise SystemExit("❌ Error: 'timestamp' column missing in network_data.csv")

# Parse timestamps safely
try:
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("int64") // 10**9
except Exception:
    df["ts"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)

# Rename columns if needed
if "packet_size" not in df.columns and "len" in df.columns:
    df.rename(columns={"len": "packet_size"}, inplace=True)
if "latency_ms" not in df.columns:
    print("⚠️ 'latency_ms' not found — generating synthetic latency values.")
    df["latency_ms"] = np.random.uniform(5, 50, size=len(df))

df = df[["ts", "packet_size", "latency_ms"]].copy()

# --- 2️⃣ Aggregate into time windows ---
df["time_group"] = (df["ts"] // AGG_SECONDS) * AGG_SECONDS
agg = df.groupby("time_group").agg(
    total_bytes=("packet_size", "sum"),
    avg_latency=("latency_ms", "mean"),
    pkt_count=("packet_size", "count")
).reset_index().sort_values("time_group")

print(f"✅ Aggregated into {len(agg)} windows (each {AGG_SECONDS}s).")

if len(agg) < SEQ_LEN + 5:
    print("⚠️ Not enough data for sequence modeling — consider collecting more samples.")

# --- 3️⃣ Prepare sequences for supervised learning ---
features = ["total_bytes", "avg_latency"]
data = agg[features].values.astype(float)

X, y = [], []
for i in range(len(data) - SEQ_LEN):
    X.append(data[i:i + SEQ_LEN])
    y.append(data[i + SEQ_LEN][0])  # predict next total_bytes

X = np.array(X)
y = np.array(y)
print("🔹 X shape:", X.shape, "y shape:", y.shape)

use_lstm = X.shape[0] >= 10  # fallback if data too small

# --- 4️⃣ Scaling ---
if use_lstm:
    scaler = StandardScaler()
    n_features = X.shape[2]
    X_2d = X.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_2d)
    X = X_scaled.reshape(-1, SEQ_LEN, n_features)
else:
    scaler = StandardScaler()
    X = scaler.fit_transform(data[:-1])

y = y.reshape(-1, 1)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y).ravel()

# --- 5️⃣ Split data ---
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"🧠 Train: {X_train.shape}, Test: {X_test.shape}")

# --- 6️⃣ Build model ---
if use_lstm:
    print("🧩 Building LSTM model...")
    model = Sequential([
        Input(shape=(SEQ_LEN, len(features))),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")
    ])
else:
    print("⚙️ Fallback to MLP (insufficient sequence data)...")
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")
    ])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# --- 7️⃣ Train ---
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor="val_loss")
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print(f"✅ Model saved to {MODEL_FILE}")

# --- 8️⃣ Evaluate ---
print("🔍 Loading best model for evaluation...")
best = load_model(MODEL_FILE, compile=False)
best.compile(optimizer="adam", loss="mse", metrics=["mae"])

y_pred_scaled = best.predict(X_test).ravel()
y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_pred_inv = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"📊 Evaluation -> MAE: {mae:.2f} bytes, RMSE: {rmse:.2f} bytes")

# --- 9️⃣ Plot results ---
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label="Actual total_bytes")
plt.plot(y_pred_inv, label="Predicted total_bytes", linestyle="--")
plt.title("Actual vs Predicted Total Bytes (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"))
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "training_loss.png"))
plt.close()

print(f"📈 Plots saved in '{PLOTS_DIR}' folder")
print("✅ Done.")
