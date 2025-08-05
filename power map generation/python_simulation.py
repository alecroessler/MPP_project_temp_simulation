import numpy as np
import csv
import matplotlib.pyplot as plt

def load_power_map_from_csv(filename):
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)

        # Read all rows and convert each to a list of floats
        power_density_map = []
        for row in reader:
            power_density_map.append([float(val) for val in row])

    # Return as np array
    return np.array(power_density_map, dtype=np.float64)



# Simulation parameters
GRID_SIZE = 256  # Number of pixels in each dimension
k = 150  # Thermal conductivity W/mK (example for silicon)
DIE_WIDTH_M = 0.016
DIE_HEIGHT_M = 0.016
h = DIE_WIDTH_M / GRID_SIZE # Grid spacing (m)

# Initial temperature map (Kelvin)
T_init_celsius = 25
T_amb = T_init_celsius + 273.15
T = np.full((GRID_SIZE, GRID_SIZE), T_amb) 

# Load the power density map from CSV
q = load_power_map_from_csv("power_map_simplified_256.csv")

# Jacobi iteration for steady-state temperature
num_iters = 10000

for iteration in range(num_iters):
    T_new = T.copy()
    for i in range(1, GRID_SIZE - 1):
        for j in range(1, GRID_SIZE - 1):
            T_new[i, j] = (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] + (h**2 / k) * q[i, j]) / 4
            #T_new[i, j] = (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1]) / 4 + (h**2 / (4 * k)) * q[i, j]

    # Dirichlet BCs: fixed ambient temp at edges
    T_new[0, :] = T_amb
    T_new[-1, :] = T_amb
    T_new[:, 0] = T_amb
    T_new[:, -1] = T_amb

    if iteration % 100 == 0:
        print(f"Iteration {iteration} max temp (C): {np.max(T) - 273.15:.8f}")
        print(f"Max temperature change: {np.max(np.abs(T_new - T)):.8f}")

    if np.max(np.abs(T_new - T)) < 1e-3:
        print(f"Converged after {iteration} iterations")
        break

    T = T_new


# Final result
print(f"Final max temperature (C): {np.max(T) - 273.15:.5f}")
print(f"Final min temperature (C): {np.min(T) - 273.15:.5f}")
print(f"Final average temperature (C): {np.mean(T) - 273.15:.5f}")


# After the simulation loop ends, save the heatmap image:
plt.figure(figsize=(8, 6))
# Convert temperature back to Celsius for visualization
temp_celsius = T - 273.15
heatmap = plt.imshow(temp_celsius, cmap='inferno', origin='lower')
plt.colorbar(heatmap, label='Temperature (°C)')
plt.title('Steady-State Temperature Distribution')
plt.xlabel('X position (pixels)')
plt.ylabel('Y position (pixels)')
plt.savefig('heatmap.png', dpi=300)  # Save the image file with high resolution
plt.close()