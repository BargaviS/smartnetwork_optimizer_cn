import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("network_data.csv")

print("✅ Data Loaded Successfully!\n")
print(df.head())


print("\n📊 Dataset Info:")
print(df.describe())


plt.figure(figsize=(10, 6))
sns.lineplot(x='timestamp', y='latency_ms', data=df)
plt.title('Network Latency Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['packet_size'], bins=20, kde=True)
plt.title('Distribution of Packet Sizes')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='packet_size', y='latency_ms', data=df)
plt.title('Packet Size vs Latency')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='timestamp', y='latency_ms', data=df)
plt.title('Network Latency Over Time')
plt.xticks(rotation=45)
plt.tight_layout()

plt.figure(figsize=(10, 6))
sns.histplot(df['packet_size'], bins=20, kde=True)
plt.title('Distribution of Packet Sizes')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='packet_size', y='latency_ms', data=df)
plt.title('Packet Size vs Latency')

plt.show()
