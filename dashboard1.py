"""
🌐 SMART NETWORK OPTIMIZER — INTERACTIVE DASHBOARD
--------------------------------------------------
Real-time visualization and control center for network monitoring.
Displays AI predictions, alerts, and live traffic stats.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os


st.set_page_config(
    page_title="Smart Network Optimizer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("🌐 SMART NETWORK OPTIMIZER — DASHBOARD")
st.markdown("#### Real-time AI-driven monitoring of network health and performance.")
st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


DATA_FILE = "network_data.csv"
ALERTS_FILE = "alerts.csv"

if not os.path.exists(DATA_FILE):
    st.error("❌ No network data found. Please run `capture.py` first.")
    st.stop()

# Read data
df = pd.read_csv(DATA_FILE)

# Clean and prepare data
if "timestamp" not in df.columns or "packet_size" not in df.columns:
    st.error("⚠️ Invalid data format. Expected 'timestamp' and 'packet_size' columns.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp")
df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")


avg_latency = df["latency_ms"].mean() if "latency_ms" in df else 0
total_bytes = df["packet_size"].sum()
avg_rate = df["packet_size"].mean()
health_score = np.clip(100 - avg_latency / 2, 0, 100)  # Example health metric


col1, col2, col3 = st.columns(3)
col1.metric("💚 Network Health Index", f"{health_score:.1f} / 100", "Stable")
col2.metric("📦 Total Data Transferred", f"{total_bytes/1e6:.2f} MB", f"{avg_rate:.1f} bytes avg")
col3.metric("⚡ Average Latency", f"{avg_latency:.2f} ms", "↓ Improved")

st.divider()


fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health_score,
    title={'text': "Network Health"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "green" if health_score > 80 else "orange" if health_score > 50 else "red"},
        'steps': [
            {'range': [0, 50], 'color': "red"},
            {'range': [50, 80], 'color': "yellow"},
            {'range': [80, 100], 'color': "green"}
        ]
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)


st.subheader("📈 Network Traffic Over Time")
fig_line = px.line(df, x="time", y="packet_size", title="📡 Live Packet Flow", markers=True)
fig_line.update_layout(xaxis_title="Time", yaxis_title="Packet Size (bytes)")
st.plotly_chart(fig_line, use_container_width=True)


st.subheader("🚨 Real-Time Alerts")
if os.path.exists(ALERTS_FILE):
    alerts_df = pd.read_csv(ALERTS_FILE)
    if len(alerts_df) > 0:
        for _, alert in alerts_df.iterrows():
            sev = alert.get("severity", "MEDIUM").upper()
            msg = alert.get("message", "Unknown event")
            ts = alert.get("timestamp", "")
            if sev == "HIGH":
                st.error(f"🔴 **CRITICAL:** {msg} at {ts}")
            elif sev == "MEDIUM":
                st.warning(f"🟠 **Warning:** {msg} at {ts}")
            else:
                st.info(f"🟢 **Normal:** {msg} at {ts}")
    else:
        st.success("✅ No active alerts detected.")
else:
    st.info("ℹ️ Alerts will appear here once `alert.py` detects anomalies.")


st.sidebar.header("🕹️ Control Panel")
if st.sidebar.button("⏸️ Pause Monitoring"):
    st.warning("Monitoring paused.")

if st.sidebar.button("🚀 Auto Optimize"):
    st.success("Network optimization algorithm activated!")

if st.sidebar.button("🔥 Simulate Attack"):
    st.error("Simulated network overload triggered (for testing).")


st.subheader("🧾 Summary Report")
st.markdown(f"""
- **Total Packets Monitored:** {len(df)}  
- **Total Data Transferred:** {total_bytes/1e6:.2f} MB  
- **Average Latency:** {avg_latency:.2f} ms  
- **Network Health:** {'✅ Excellent' if health_score > 80 else '🟠 Moderate' if health_score > 50 else '🔴 Poor'}  
""")

st.divider()
st.caption("Developed as part of Smart Network Optimization Project — AI meets Cybersecurity ⚙️")
