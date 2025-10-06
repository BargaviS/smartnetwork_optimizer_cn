
"""
Smart Network Optimizer — Interactive Dashboard
Clear, understandable, and visually rich dashboard
to show network activity, predictions, and alerts.
"""

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import os


DATA_FILE = "network_data.csv"
ALERT_FILE = "alerts.csv"
MODEL_FILE = "model.keras"
AGG_SECONDS = 10
SEQ_LEN = 6


st.set_page_config(page_title="Smart Network Optimizer", layout="wide")


st.title("📊 Smart Network Optimizer — Dashboard")
st.write("A real-time AI system for monitoring and predicting network traffic and latency.")
st.divider()


if not os.path.exists(DATA_FILE):
    st.error("❌ No network data found. Run capture.py first.")
    st.stop()

df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["ts"] = df["timestamp"].astype("int64") // 10**9
df["time_group"] = (df["ts"] // AGG_SECONDS) * AGG_SECONDS

agg = df.groupby("time_group").agg(
    total_bytes=("packet_size", "sum"),
    avg_latency=("latency_ms", "mean"),
).reset_index().sort_values("time_group")


agg["total_MB"] = agg["total_bytes"] / (1024 * 1024)


st.subheader("🧠 AI Model Status")
if os.path.exists(MODEL_FILE):
    st.success("✅ AI model loaded successfully!")
    model = load_model(MODEL_FILE)
else:
    st.error("❌ Model not found. Please run predict.py first.")
    st.stop()


features = ["total_bytes", "avg_latency"]
data = agg[features].values.astype(float)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


if len(data_scaled) >= SEQ_LEN:
    X_input = np.array([data_scaled[-SEQ_LEN:]])
    pred_scaled = model.predict(X_input).ravel()[0]
    y_pred = scaler.inverse_transform([[pred_scaled, 0]])[0][0]
else:
    y_pred = None


col1, col2, col3 = st.columns(3)
latest_MB = agg["total_MB"].iloc[-1]
avg_latency = agg["avg_latency"].iloc[-1]

status = "✅ Network Normal"
status_color = "green"

if latest_MB > agg["total_MB"].mean() * 2:
    status = "🚨 Abnormal Traffic Detected!"
    status_color = "red"

with col1:
    st.metric("Current Traffic", f"{latest_MB:.2f} MB")
with col2:
    st.metric("Average Latency", f"{avg_latency:.1f} ms")
with col3:
    st.markdown(f"<h4 style='color:{status_color}'>{status}</h4>", unsafe_allow_html=True)


if y_pred:
    st.info(f"🔮 AI Prediction (Next 10s): **{y_pred/1e6:.2f} MB expected traffic**")


st.subheader("📈 Network Traffic Trend")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(agg["time_group"], agg["total_MB"], label="Actual Traffic (MB)", color="blue")
if y_pred:
    ax.axhline(y=y_pred/1e6, color="orange", linestyle="--", label="Predicted Next (MB)")
ax.set_xlabel("Time (epoch)")
ax.set_ylabel("Traffic (MB)")
ax.legend()
st.pyplot(fig)

st.divider()
st.subheader("🚨 Recent Alerts")
if os.path.exists(ALERT_FILE):
    alerts = pd.read_csv(ALERT_FILE)
    if len(alerts) == 0:
        st.success("✅ No recent anomalies detected.")
    else:
        for _, a in alerts.tail(5).iterrows():
            st.warning(f"**{a['timestamp']}** — {a['message']} (Severity: {a['severity']})")
else:
    st.info("No alert file found yet. Run alert.py after predict.py to detect anomalies.")

st.caption("🔁 Dashboard auto-refreshes every few seconds for live updates.")
time.sleep(2)
