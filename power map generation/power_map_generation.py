import numpy as np
import matplotlib.pyplot as plt
import csv

np.random.seed(42) # Set seed for consistent data

# EV6 floorplan units: name, width (m), height (m), x (m), y (m)
floorplan = [
    ("L2_left", 0.004900, 0.006200, 0.000000, 0.009800),
    ("L2", 0.016000, 0.009800, 0.000000, 0.000000),
    ("L2_right", 0.004900, 0.006200, 0.011100, 0.009800),
    ("Icache", 0.003100, 0.002600, 0.004900, 0.009800),
    ("Dcache", 0.003100, 0.002600, 0.008000, 0.009800),
    ("Bpred_0", 0.001033, 0.000700, 0.004900, 0.012400),
    ("Bpred_1", 0.001033, 0.000700, 0.005933, 0.012400),
    ("Bpred_2", 0.001033, 0.000700, 0.006967, 0.012400),
    ("DTB_0", 0.001033, 0.000700, 0.008000, 0.012400),
    ("DTB_1", 0.001033, 0.000700, 0.009033, 0.012400),
    ("DTB_2", 0.001033, 0.000700, 0.010067, 0.012400),
    ("FPAdd_0", 0.001100, 0.000900, 0.004900, 0.013100),
    ("FPAdd_1", 0.001100, 0.000900, 0.006000, 0.013100),
    ("FPReg_0", 0.000550, 0.000380, 0.004900, 0.014000),
    ("FPReg_1", 0.000550, 0.000380, 0.005450, 0.014000),
    ("FPReg_2", 0.000550, 0.000380, 0.006000, 0.014000),
    ("FPReg_3", 0.000550, 0.000380, 0.006550, 0.014000),
    ("FPMul_0", 0.001100, 0.000950, 0.004900, 0.014380),
    ("FPMul_1", 0.001100, 0.000950, 0.006000, 0.014380),
    ("FPMap_0", 0.001100, 0.000670, 0.004900, 0.015330),
    ("FPMap_1", 0.001100, 0.000670, 0.006000, 0.015330),
    ("IntMap", 0.000900, 0.001350, 0.007100, 0.014650),
    ("IntQ", 0.001300, 0.001350, 0.008000, 0.014650),
    ("IntReg_0", 0.000900, 0.000670, 0.009300, 0.015330),
    ("IntReg_1", 0.000900, 0.000670, 0.010200, 0.015330),
    ("IntExec", 0.001800, 0.002230, 0.009300, 0.013100),
    ("FPQ", 0.000900, 0.001550, 0.007100, 0.013100),
    ("LdStQ", 0.001300, 0.000950, 0.008000, 0.013700),
    ("ITB_0", 0.000650, 0.000600, 0.008000, 0.013100),
    ("ITB_1", 0.000650, 0.000600, 0.008650, 0.013100),
]
# Power values in Watts for each unit "block" of the floorplan
power_values = {
    "L2_left": 1.44,
    "L2": 7.37,
    "L2_right": 1.44,
    "Icache": 8.27,
    "Dcache": 14.3,
    "Bpred_0": 1.51666666666667,
    "Bpred_1": 1.51666666666667,
    "Bpred_2": 1.51666666666667,
    "DTB_0": 0.0596666666666667,
    "DTB_1": 0.0596666666666667,
    "DTB_2": 0.0596666666666667,
    "FPAdd_0": 0.62,
    "FPAdd_1": 0.62,
    "FPReg_0": 0.19375,
    "FPReg_1": 0.19375,
    "FPReg_2": 0.19375,
    "FPReg_3": 0.19375,
    "FPMul_0": 0.665,
    "FPMul_1": 0.665,
    "FPMap_0": 0.02355,
    "FPMap_1": 0.02355,
    "IntMap": 1.07,
    "IntQ": 0.365,
    "IntReg_0": 2.585,
    "IntReg_1": 2.585,
    "IntExec": 7.7,
    "FPQ": 0.0354,
    "LdStQ": 3.46,
    "ITB_0": 0.2,
    "ITB_1": 0.2,
}

GRID_SIZE = 256

max_x = 0
max_y = 0
for _, w, h, x, y in floorplan:
    max_x = max(max_x, x + w)
    max_y = max(max_y, y + h)

def scale_to_grid(x_m, y_m):
    return int(x_m / max_x * GRID_SIZE), int(y_m / max_y * GRID_SIZE)

def scale_size_to_grid(w_m, h_m):
    return max(1, int(w_m / max_x * GRID_SIZE)), max(1, int(h_m / max_y * GRID_SIZE))

power_map = np.zeros((GRID_SIZE, GRID_SIZE))

for name, w_m, h_m, x_m, y_m in floorplan:
    px, py = scale_to_grid(x_m, y_m)
    pw, ph = scale_size_to_grid(w_m, h_m)

    x_end = min(px + pw, GRID_SIZE)
    y_end = min(py + ph, GRID_SIZE)

    base_power = power_values.get(name, 0)

    block_width = x_end - px
    block_height = y_end - py

    # Start with ones for equal distribution
    block_power_map = np.ones((block_height, block_width))

    # Add random multiplicative variations in fine blocks
    num_fine_blocks = np.random.randint(20, 40)
    for _ in range(num_fine_blocks):
        fw = np.random.randint(2, max(3, block_width // 6))
        fh = np.random.randint(2, max(3, block_height // 6))
        fx = np.random.randint(0, block_width - fw + 1)
        fy = np.random.randint(0, block_height - fh + 1)
        deviation = np.random.uniform(0.6, 1.4)
        block_power_map[fy:fy+fh, fx:fx+fw] *= deviation

    # Add small multiplicative noise (around 1)
    pixel_noise = np.random.normal(loc=1.0, scale=0.05, size=(block_height, block_width))
    block_power_map *= pixel_noise

    # Clip negative values to zero
    block_power_map = np.clip(block_power_map, 0, None)

    # Normalize so sum equals the base power of the block
    block_power_map *= base_power / np.sum(block_power_map)
    
    power_map[py:y_end, px:x_end] = block_power_map



# Compute power density map
DIE_WIDTH_M = 0.016
DIE_HEIGHT_M = 0.016
pixel_area = (DIE_WIDTH_M * DIE_HEIGHT_M) / (GRID_SIZE * GRID_SIZE)  # m^2

# Power density = Power / Area per pixel
pixel_volume = pixel_area * 0.001  # Assuming thickness of 1 mm
pixel_volume = pixel_area * 0.0005  # FOR 256
power_density_map = power_map / pixel_volume  # Units: W/m^3

print("Max power density (W/m^3):", np.max(power_density_map))
print("Min power density (W/m^3):", np.min(power_density_map))
print("Total power (W):", np.sum(power_map))  

# Visualize power density
plt.figure(figsize=(10, 10))
plt.imshow(power_density_map, cmap='hot', interpolation='nearest', origin='lower')
plt.colorbar(label="Power Density (W/m^3)")
plt.title("Power Density Map")
plt.xlabel("Grid X")
plt.ylabel("Grid Y")

# Save the figure as a PNG file
plt.savefig("power map generation/power_density_map.png", dpi=300, bbox_inches='tight')


plt.show()

# Save power density map to CSV
with open("power_map_simplified_256.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for row in power_density_map:
        writer.writerow([f"{val:.8f}" for val in row])



