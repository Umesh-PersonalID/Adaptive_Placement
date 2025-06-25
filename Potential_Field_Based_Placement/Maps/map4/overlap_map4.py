import numpy as np
import matplotlib.pyplot as plt
from grid_generator_map4 import grid

# Global Variables
overlap = 0
total_grid_without_obstacle = 0
global_sensor_range = 10
coverage_list = []
overlap_list = []

class Robot:
    def __init__(self, id, position, sensor_range):
        self.id = id
        self.position = position
        self.sensor_range = sensor_range

    def sense(self, occupancy_grid):
        global global_sensor_range
        global_sensor_range = self.sensor_range
        x, y = map(int, self.position)
        sensor_range_sq = self.sensor_range ** 2
        min_x = max(0, x - self.sensor_range)
        max_x = min(occupancy_grid.shape[0] - 1, x + self.sensor_range)
        min_y = max(0, y - self.sensor_range)
        max_y = min(occupancy_grid.shape[1] - 1, y + self.sensor_range)

        for nx in range(int(min_x), int(max_x) + 1):
            for ny in range(int(min_y), int(max_y) + 1):
                dx = x - nx
                dy = y - ny
                distance_sq = dx ** 2 + dy ** 2
                if distance_sq <= sensor_range_sq:
                    if occupancy_grid[nx][ny] == -1:
                        occupancy_grid[nx][ny] = 0  # Mark as covered


def count_desired_area(grid_data):
    return np.sum(grid_data == -1)

overlap = 0
def calculate_overlap(grid_data, robot_positions, sensor_range):
    global overlap
    """Count overlapping cells due to multiple sensor coverage"""
    coverage_count = np.zeros_like(grid_data, dtype=int)
    for x, y in robot_positions:
        sensor_range_sq = sensor_range ** 2
        min_x = max(0, x - sensor_range)
        max_x = min(grid_data.shape[0] - 1, x + sensor_range)
        min_y = max(0, y - sensor_range)
        max_y = min(grid_data.shape[1] - 1, y + sensor_range)

        for nx in range(min_x, max_x + 1):
            for ny in range(min_y, max_y + 1):
                dx = x - nx
                dy = y - ny
                if dx * dx + dy * dy <= sensor_range_sq:
                    coverage_count[nx, ny] += 1
    for nx in range(len(coverage_count)):
        for ny in range(len(coverage_count[0])):
            if grid_data[nx][ny] != 1 and coverage_count[nx,ny] > 1:
                overlap += coverage_count[nx][ny]
    
    print(overlap)
    overlap_cells = np.sum((coverage_count > 1) & (grid_data != 1))
    
    total_free_cells = np.sum(grid_data == -1)
    print(total_free_cells)
    return overlap/total_free_cells * 100


def deploy_robots_from_file(filename, sensor_range):
    # Read positions from file
    positions = []
    with open(filename, 'r') as f:
        for line in f:
            if ',' in line:
                x_str, y_str = line.strip().split(',')
                try:
                    x = float(x_str)
                    y = float(y_str)
                    positions.append((int(x), int(y)))
                except ValueError:
                    continue

    print(f"Loaded {len(positions)} robot positions from {filename}")

    robots = [Robot(i, pos, sensor_range) for i, pos in enumerate(positions)]

    occupancy_grid = grid.copy()

    total_free = count_desired_area(occupancy_grid)

    for robot in robots:
        robot.sense(occupancy_grid)

    covered = np.sum(occupancy_grid == 0)
    coverage_percent = covered / total_free * 100
    print(f"Total Coverage: {coverage_percent:.2f}%")
    coverage_list.append(coverage_percent)

    overlap_percent = calculate_overlap(grid, positions, sensor_range)
    print(f"Overlap Percentage: {overlap_percent:.2f}%")
    overlap_list.append(overlap_percent)

    visualize_grid(occupancy_grid, robots, coverage_percent, overlap_percent)
    return coverage_percent, overlap_percent


def visualize_grid(grid_data, robots, coverage, overlap):
    plt.figure(figsize=(10, 10))
    img = np.zeros((*grid_data.shape, 3))
    img[grid_data == -1] = [0.7, 0.7, 0.7]  # Free space (gray)
    img[grid_data == 0] = [1, 1, 1]         # Covered (white)
    img[grid_data == 1] = [0, 0, 0]         # Obstacles (black)

    for robot in robots:
        x, y = robot.position
        plt.plot(y, x, 'ro', markersize=3)

    plt.title(f"Final Deployment\nCoverage: {coverage:.2f}%, Overlap: {overlap:.2f}%")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-0.5, grid_data.shape[1] - 0.5)
    plt.ylim(-0.5, grid_data.shape[0] - 0.5)
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    deploy_robots_from_file("Final_Position.txt", sensor_range=10)
