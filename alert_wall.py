# alert_wall.py
"""
Alert Wall
----------
Displays all recent alerts detected by alert.py in real time.
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime

st.set_page_config(page_title="🚨 Alert Wall", layout="wide")
st.title("🚨 Network Alert Wall")

st.info("Auto-refreshing every 5 seconds...")

placeholder = st.empty()

while True:
    try:
        alerts = pd.read_csv("alerts.csv")
    except FileNotFoundError:
        st.warning("⚠️ No alerts found yet. Run alert.py or wait for detection.")
        time.sleep(5)
        continue

    with placeholder.container():
        st.subheader(f"Active Alerts (Last updated: {datetime.now().strftime('%H:%M:%S')})")
        if alerts.empty:
            st.success("✅ No active alerts. Network stable.")
        else:
            for _, row in alerts.iterrows():
                color = "red" if row["severity"] == "HIGH" else "orange"
                st.markdown(
                    f"""
                    <div style="background-color:{color};padding:10px;border-radius:10px;margin-bottom:10px">
                    <b>Time:</b> {row['timestamp']} <br>
                    <b>Severity:</b> {row['severity']} <br>
                    <b>Bytes:</b> {row['total_bytes']:.0f} <br>
                    <b>Latency:</b> {row['avg_latency']:.2f} ms <br>
                    <b>Message:</b> {row['message']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    time.sleep(5)
