import numpy as np
import matplotlib.pyplot as plt
from grid_generator_map1 import grid  # Make sure you're using the same grid

SENSOR_RANGE = 15

# Load positions
def load_positions(path="Final_Combined_Position.txt"):
    positions = []
    with open(path, "r") as f:
        for line in f:
            x, y = map(float, line.strip().split(","))
            positions.append((int(x), int(y)))
    return positions

def plot_deployment(grid, positions, sensor_range):
    plt.figure(figsize=(8, 8))
    
    # Display the environment grid
    plt.imshow(grid, cmap='gray_r', origin='lower')

    # Plot sensor positions
    for x, y in positions:
        plt.plot(y, x, 'ro')  # 'ro' = red dot (note: y and x are swapped for imshow)
        # Draw sensing range circle (optional)
        circle = plt.Circle((y, x), sensor_range, color='red', fill=False, linestyle='--', linewidth=0.8)
        plt.gca().add_patch(circle)

    plt.title("Final Deployment Visualization")
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.grid(False)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    positions = load_positions("Final_Combined_Position.txt")
    plot_deployment(grid, positions, SENSOR_RANGE)
