
"""
Alert Detection Module
----------------------
Detects abnormal spikes in network usage or latency from network_data.csv.
Saves alerts to alerts.csv for dashboard and alert wall to read.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


CSV_FILE = "network_data.csv"
ALERTS_FILE = "alerts.csv"
AGG_SECONDS = 10
Z_THRESHOLD = 2.5  # anomaly threshold (z-score > 2.5 = alert)

def detect_anomalies():
    # Load data safely
    if not os.path.exists(CSV_FILE):
        print("❌ network_data.csv not found. Run capture.py first.")
        return

    df = pd.read_csv(CSV_FILE)
    if len(df) < 10:
        print("⚠️ Not enough data to detect anomalies.")
        return

    # Parse timestamps
    try:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("int64") // 10**9
    except Exception:
        df["ts"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)

    # Aggregate data by time window
    df["time_group"] = (df["ts"] // AGG_SECONDS) * AGG_SECONDS
    agg = df.groupby("time_group").agg(
        total_bytes=("packet_size", "sum"),
        avg_latency=("latency_ms", "mean")
    ).reset_index().sort_values("time_group")

    if agg.empty:
        print("⚠️ No valid data for anomaly detection.")
        return

  
    agg["bytes_z"] = (agg["total_bytes"] - agg["total_bytes"].mean()) / (agg["total_bytes"].std() + 1e-6)
    agg["latency_z"] = (agg["avg_latency"] - agg["avg_latency"].mean()) / (agg["avg_latency"].std() + 1e-6)

    
    alerts = []
    for _, row in agg.iterrows():
        if abs(row["bytes_z"]) > Z_THRESHOLD or abs(row["latency_z"]) > Z_THRESHOLD:
            severity = "HIGH" if abs(row["bytes_z"]) > 3 or abs(row["latency_z"]) > 3 else "MEDIUM"
            message = (
                f"Spike detected! Bytes={row['total_bytes']:.0f}, "
                f"Latency={row['avg_latency']:.1f} ms (Severity={severity})"
            )
            alerts.append({
                "timestamp": datetime.utcfromtimestamp(row["time_group"]).strftime("%Y-%m-%d %H:%M:%S"),
                "total_bytes": row["total_bytes"],
                "avg_latency": row["avg_latency"],
                "severity": severity,
                "message": message
            })

    
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        if os.path.exists(ALERTS_FILE):
            old = pd.read_csv(ALERTS_FILE)
            alerts_df = pd.concat([old, alerts_df], ignore_index=True).drop_duplicates(subset=["timestamp"])
        alerts_df.to_csv(ALERTS_FILE, index=False)
        print(f"🚨 {len(alerts)} anomalies detected. Saved to {ALERTS_FILE}")
    else:
        print("✅ No anomalies detected.")

if __name__ == "__main__":
    detect_anomalies()
