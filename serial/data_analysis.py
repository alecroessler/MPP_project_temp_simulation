import matplotlib.pyplot as plt
import pandas as pd

data = {
    "iteration": [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                  10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
                  18000, 19000, 20000, 21000, 22000, 23000],
    "max_change": [0.09110, 0.01534, 0.01226, 0.00975, 0.00793, 0.00661,
                   0.00562, 0.00486, 0.00426, 0.00376, 0.00336, 0.00302,
                   0.00272, 0.00247, 0.00225, 0.00205, 0.00188, 0.00173,
                   0.00159, 0.00146, 0.00135, 0.00124, 0.00115, 0.00106],
    "max_temp": [25.09110, 44.52675, 57.09434, 67.55969, 76.19184, 83.37226,
                 89.42951, 94.60199, 99.07686, 102.98722, 106.42706, 109.47416,
                 112.19540, 114.62837, 116.81220, 118.78867, 120.57765,
                 122.19867, 123.67093, 125.01086, 126.23569, 127.36137,
                 128.39181, 129.33582]
}

df = pd.DataFrame(data)

# Plot Max Temperature
plt.figure(figsize=(10, 5))
plt.plot(df["iteration"], df["max_temp"], marker='o', label='Max Temperature')
plt.xlabel("Iteration")
plt.ylabel("Max Temperature (Celsius)")
plt.title("Max Temperature vs Iteration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot Max Change
plt.figure(figsize=(10, 5))
plt.plot(df["iteration"], df["max_change"], marker='o', color='orange', label='Max Change')
plt.xlabel("Iteration")
plt.ylabel("Max Change")
plt.title("Convergence Rate (Max Change per Iteration)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
