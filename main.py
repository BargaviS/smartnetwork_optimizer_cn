import subprocess
import time

print("🚀 Starting network packet capture...")
capture_process = subprocess.Popen(["python", "capture.py"])

time.sleep(3)

print("📊 Launching Streamlit dashboard...")
subprocess.run(["streamlit", "run", "dashboard.py"])

print("🛑 Stopping packet capture...")
capture_process.terminate()
