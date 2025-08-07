import numpy as np
import matplotlib.pyplot as plt

# Load temperature data
T = np.loadtxt("final_temperature_map.csv", delimiter=",")

# Plot
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(T, cmap='inferno', origin='lower')
plt.colorbar(heatmap, label='Temperature (°C)')
plt.title('Steady-State Temperature Distribution')
plt.xlabel('X position (pixels)')
plt.ylabel('Y position (pixels)')
plt.savefig("temperature_heatmap.png", dpi=300)
plt.show()
