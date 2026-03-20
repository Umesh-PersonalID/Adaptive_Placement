import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import numpy as np

# --------------------------
# Define region (example polygon)
# --------------------------
region = Polygon([
    (0, 0), (8, 0), (10, 4),
    (7, 8), (2, 7), (0, 3)
])

# --------------------------
# Disk parameters
# --------------------------
r = 2.2

centers = np.array([
    [3, 3],
    [6, 4],
    [5, 6]
])

# --------------------------
# Create disks
# --------------------------
disks = [Point(c).buffer(r, resolution=128) for c in centers]

# Union of disks (Ω_m(x))
union_disks = unary_union(disks)

# --------------------------
# Plotting
# --------------------------
fig, ax = plt.subplots(figsize=(6, 6))

# Plot region
x_reg, y_reg = region.exterior.xy
ax.fill(x_reg, y_reg, color="yellow", alpha=0.4, label="Region A")

# Plot individual disks
colors = ["red", "blue", "green"]

for disk, c, color in zip(disks, centers, colors):
    x_d, y_d = disk.exterior.xy
    ax.fill(x_d, y_d, color=color, alpha=0.3)
    ax.plot(c[0], c[1], 'ko')  # center

# Plot union boundary (Ω_m(x))
if union_disks.geom_type == "Polygon":
    x_u, y_u = union_disks.exterior.xy
    ax.plot(x_u, y_u, color="black", linewidth=4, label=r"$\Omega_m(x)$")
else:
    for geom in union_disks.geoms:
        x_u, y_u = geom.exterior.xy
        ax.plot(x_u, y_u, color="black", linewidth=2)

# Formatting
ax.set_aspect('equal')
ax.set_title(r"Union of Disks $\Omega_m(x)$")
ax.legend()
ax.grid(True)

# Save figure (for paper)
plt.savefig("union_of_disks.png", dpi=300, bbox_inches='tight')
plt.show()