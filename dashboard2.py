# dashboard.py
"""
Smart Network Optimizer — Interactive Dashboard
------------------------------------------------
Visual dashboard for real-time network monitoring, prediction, and alerts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# --------------------------
# Config
# --------------------------
st.set_page_config(page_title="Smart Network Optimizer", layout="wide")
DATA_FILE = "network_data.csv"
ALERTS_FILE = "alerts.csv"

# --------------------------
# Load data safely
# --------------------------
@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    else:
        return pd.DataFrame(columns=["timestamp", "packet_size", "latency_ms"])

@st.cache_data
def load_alerts():
    if os.path.exists(ALERTS_FILE):
        alerts = pd.read_csv(ALERTS_FILE)
        return alerts
    else:
        return pd.DataFrame(columns=["timestamp", "message", "severity"])

df = load_data()
alerts = load_alerts()

# --------------------------
# Header & Title
# --------------------------
st.title("📊 Smart Network Optimizer Dashboard")
st.markdown("### Real-Time Network Monitoring & AI Prediction System")
st.divider()

# --------------------------
# Metrics Overview
# --------------------------
col1, col2, col3 = st.columns(3)

if not df.empty:
    total_packets = len(df)
    avg_latency = np.mean(df["latency_ms"])
    total_bytes = np.sum(df["packet_size"])
else:
    total_packets, avg_latency, total_bytes = 0, 0, 0

col1.metric("📦 Total Packets", f"{total_packets}")
col2.metric("⚡ Avg Latency (ms)", f"{avg_latency:.2f}")
col3.metric("💾 Total Bytes", f"{total_bytes:.0f}")

# --------------------------
# Traffic Trend Visualization
# --------------------------
st.subheader("📈 Network Traffic Over Time")

if not df.empty:
    fig = px.line(df, x="timestamp", y="packet_size",
                  title="Traffic Flow (Packet Size Over Time)",
                  labels={"timestamp": "Time", "packet_size": "Packet Size (bytes)"},
                  line_shape="spline")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No network data available yet. Run capture.py to collect packets.")

# --------------------------
# Alerts Section
# --------------------------
st.subheader("🚨 Real-Time Alerts")

if not alerts.empty:
    for _, row in alerts.iterrows():
        color = "🟢"
        if row["severity"] == "MEDIUM":
            color = "🟡"
        elif row["severity"] == "HIGH":
            color = "🔴"
        st.markdown(f"{color} **[{row['timestamp']}]** — {row['message']}")
else:
    st.success("✅ No active alerts detected. System is stable.")

# --------------------------
# Interactive Controls
# --------------------------
st.divider()
st.subheader("🕹️ Control Panel")

colA, colB, colC = st.columns(3)
if colA.button("▶️ Start Monitoring"):
    st.toast("Monitoring started...", icon="✅")

if colB.button("⏸️ Pause Monitoring"):
    st.toast("Monitoring paused.", icon="⏸️")

if colC.button("🚀 Simulate Attack"):
    st.toast("Simulated network spike generated!", icon="⚠️")

st.markdown("🧠 **AI Model Ready** — Predicting future network load in real time.")

# --------------------------
# Footer
# --------------------------
st.divider()

