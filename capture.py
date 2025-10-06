import pandas as pd
import random
import time

def simulate_packet():
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_ip": f"192.168.1.{random.randint(1, 255)}",
        "dest_ip": f"10.0.0.{random.randint(1, 255)}",
        "source_port": random.randint(1000, 9999),
        "dest_port": random.randint(80, 8080),
        "packet_size": random.randint(40, 1500),
        "latency_ms": round(random.uniform(1, 100), 2)
    }

data = []

print("Simulating network packets... Press Ctrl+C to stop.")
try:
    while True:
        packet = simulate_packet()
        data.append(packet)
        print(packet)
        time.sleep(1)

except KeyboardInterrupt:
    df = pd.DataFrame(data)
    df.to_csv("network_data.csv", index=False)
    print("\n✅ Saved simulated data to network_data.csv")
