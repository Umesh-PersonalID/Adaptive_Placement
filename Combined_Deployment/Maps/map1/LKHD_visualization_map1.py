import numpy as np
import matplotlib.pyplot as plt
from grid_generator_map1 import grid
import math

# Load the waypoint path from LKH-D_order_map1.txt
def load_path(filename="LKH-D_order_map1.txt"):
    path = []
    with open(filename) as f:
        f.readline()  # skip first line
        for line in f:
            parts = line.strip().split("\t")
            x = float(parts[8]) * 1e5
            y = float(parts[9]) * 1e5
            path.append((int(round(x)), int(round(y))))
    return path

# Count how many cells in the grid are free (-1)
def count_free_cells(grid):
    return np.sum(grid == -1)

# Traverse and mark coverage/overlap
def simulate_path(grid, path, sensor_range=15):
    grid_copy = grid.copy()
    coverage = 0
    overlap = 0

    for px, py in path:
        for dx in range(-sensor_range, sensor_range+1):
            for dy in range(-sensor_range, sensor_range+1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    if dx**2 + dy**2 <= sensor_range**2:
                        if grid_copy[nx, ny] == -1:
                            grid_copy[nx, ny] = 0
                            coverage += 1
                        elif grid_copy[nx, ny] == 0:
                            grid_copy[nx, ny] = 2
                            overlap += 1
    return grid_copy, coverage, overlap

# Compute total path length
def compute_path_length(path):
    length = 0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        length += math.hypot(x2 - x1, y2 - y1)
    return length

# Plotting
def plot_path(grid, path, title="Drone Path Traversal"):
    color_map = {
        -1: [0.8, 0.8, 0.8],  # free (gray)
         0: [1.0, 1.0, 1.0],  # covered (white)
         1: [0.0, 0.0, 0.0],  # obstacle (black)
         2: [1.0, 0.0, 0.0]   # overlap (red)
    }
    rgb_image = np.zeros((*grid.shape, 3))
    for val, color in color_map.items():
        rgb_image[grid == val] = color

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image, origin="lower")
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, color='blue', linewidth=1.5, label='Drone Path')
    plt.scatter(path_y, path_x, c='blue', s=3)
    plt.title(title)
    plt.xlabel("Y-axis")
    plt.ylabel("X-axis")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# Run everything
if __name__ == "__main__":
    drone_path = load_path("LKH-D_order_map1.txt")
    free_cells = count_free_cells(grid)
    updated_grid, covered, overlapped = simulate_path(grid, drone_path, sensor_range=15)
    path_length = compute_path_length(drone_path)

    print(f"Total path length: {path_length:.2f} units")
    print(f"Total coverage: {covered} cells")
    print(f"Total overlap: {overlapped} cells")
    print(f"Percentage overlap over free area: {overlapped / free_cells * 100:.2f}%")

    plot_path(updated_grid, drone_path)