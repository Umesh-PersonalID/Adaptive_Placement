import numpy as np
import matplotlib.pyplot as plt
from grid_generator_map7 import grid  # Assuming this is available
import sys
import matplotlib.animation as animation

import time
# Global Variablescd .
overlap = 0
count = 0
total_grid_without_obstacle = 0
global_sensor_range = 0
coverage_list = []
overlap_list = []

# Simulation settings
NUM_NODES = 88
SENSOR_RANGE = 10.0
COMM_RANGE = 40.0
AREA_SIZE = (200, 200)
K_NEIGHBORS = 6
TIME_STEPS = 10000
DELTA_T = 0.03

# Force parameters
K_COVER = 25.0
K_DEGREE = 60.0
Q = 1
V = 0.1
MASS = 1.0


last_coverage = 0.0
coverage_check_interval = 1000
coverage_threshold = 0.5  # percent


np.random.seed(42)
center_x, center_y = int(AREA_SIZE[0] / 2), int(AREA_SIZE[1] / 2)

if grid[center_x][center_y] == 1:
    print("Obstacle at the starting position. Exiting.")
    sys.exit(1)

# Generate initial positions avoiding obstacles
positions = []
while len(positions) < NUM_NODES:
    pos = np.random.rand(2) * 6 + [center_x - 3, center_y - 3]
    x, y = int(pos[0]), int(pos[1])
    if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x][y] != 1:
        positions.append(pos)
positions = np.array(positions)
velocities = np.zeros_like(positions)
final_position = []

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


from scipy.spatial import KDTree


def get_neighbors(pos):
    neighbors = []
    for i in range(len(pos)):
        n = []
        for j in range(len(pos)):
            if i != j and distance(pos[i], pos[j]) <= COMM_RANGE:
                n.append(j)
        neighbors.append(n)
    return neighbors


def compute_forces(pos, neighbors):
    forces = np.zeros_like(pos)
    count = 0
    for i in range(len(pos)):
        nbrs = neighbors[i]
        dists = [(j, distance(pos[i], pos[j])) for j in nbrs]
        sorted_nbrs = sorted(dists, key=lambda x: x[1])
        critical = [j for j, _ in sorted_nbrs[:K_NEIGHBORS]]
        if len(nbrs) > K_NEIGHBORS:
            
            for j, d in dists:
                if d == 0:
                    continue
                if d < 2 * SENSOR_RANGE:
                    f = (K_COVER + 2*d) / (d ** 2 + 1e-5)
                    dir = (pos[i] - pos[j]) / d
                    forces[i] += f * dir
        else:
            count +=1
            
            for j, d in dists:
                if d == 0:
                    continue
                dir = (pos[j] - pos[i]) / d
                if d < 2 * SENSOR_RANGE:
                    f = (K_COVER + 2*d) / (d ** 2 + 1e-5)
                    forces[i] -= f * dir
                if j in critical and (d >= Q * COMM_RANGE or d >= (Q - 0.1) * COMM_RANGE):
                    f = K_DEGREE / ((COMM_RANGE - d) ** 2 + 1e-5)
                    forces[i] += f * dir
   
    return forces


def is_velocity_stable(velocity):
    return np.all(np.abs(velocity) < 0.04)


def all_have_k_neighbors(neighbors):
    return all(len(nbrs) >= K_NEIGHBORS for nbrs in neighbors)


def is_in_obstacle(pos, grid_data):
    x, y = int(pos[0]), int(pos[1])
    if 0 <= x < grid_data.shape[0] and 0 <= y < grid_data.shape[1]:
        return grid_data[x][y] == 1
    return True  # Out of bounds treated as obstacle


def get_obstacle_boundary_normal(pos, grid_data, search_radius=2):
    x, y = int(pos[0]), int(pos[1])
    normals = []
    shape = grid_data.shape
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
                if grid_data[nx][ny] == 1:
                    vec = np.array([pos[0] - nx, pos[1] - ny])
                    if np.linalg.norm(vec) > 0:
                        normals.append(vec / np.linalg.norm(vec))
    if not normals:
        return None
    avg_normal = np.mean(normals, axis=0)
    return avg_normal / np.linalg.norm(avg_normal)


def project_to_tangent(force, normal):
    force_along_normal = np.dot(force, normal) * normal
    return force - force_along_normal


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
                    if occupancy_grid[nx, ny] == -1:
                        occupancy_grid[nx, ny] = 0


class ConstrainDeployment:
    def __init__(self):
        self.grid = grid.copy()
        self.num_robots = NUM_NODES
        self.sensor_range = SENSOR_RANGE
        self.positions = positions.copy()
        self.velocities = velocities.copy()
        self.robots = [Robot(i, self.positions[i], self.sensor_range) for i in range(self.num_robots)]

    def count_desired_area(self):
        global total_grid_without_obstacle
        total_grid_without_obstacle = np.sum(self.grid == -1)

    def update_occupancy(self):
        self.grid[self.grid == 0] = -1
        for robot in self.robots:
            robot.sense(self.grid)

    def find_safe_position(self, pos, max_search_radius=5):
        x, y = int(pos[0]), int(pos[1])
        shape = self.grid.shape
        for r in range(1, max_search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
                        if self.grid[nx][ny] != 1:
                            return np.array([nx, ny])
        return None

    def run_step(self):
        neighbors = get_neighbors(self.positions)
        forces = compute_forces(self.positions, neighbors)
        accels = forces / MASS
        self.velocities = (1 - V) * self.velocities + accels * DELTA_T
        new_positions = self.positions + self.velocities * DELTA_T
        new_positions = np.clip(new_positions, [0, 0], AREA_SIZE)

        for i in range(self.num_robots):
            if is_in_obstacle(new_positions[i], self.grid):
                normal = get_obstacle_boundary_normal(self.positions[i], self.grid)
                if normal is not None:
                    tangential_force = project_to_tangent(forces[i], normal)
                    tangential_accel = tangential_force / MASS
                    self.velocities[i] = (1 - V) * self.velocities[i] + tangential_accel * DELTA_T
                    new_positions[i] = self.positions[i] + self.velocities[i] * DELTA_T
                    new_positions[i] = np.clip(new_positions[i], [0, 0], AREA_SIZE)

                if is_in_obstacle(new_positions[i], self.grid):  # Fallback
                    safe_pos = self.find_safe_position(new_positions[i])
                    if safe_pos is not None:
                        new_positions[i] = safe_pos
                    else:
                        self.velocities[i] *= 0  # Stop

            self.robots[i].position = new_positions[i]

        self.positions = new_positions
        self.update_occupancy()
        return all_have_k_neighbors(neighbors) and is_velocity_stable(self.velocities)

    def count_coverage(self):
        return np.sum(self.grid == 0)


def visualize(deployment):
    deployment.count_desired_area()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define variables to track coverage change
    last_coverage = 0.0
    step_counter = 0

    def update(frame):
        global end_time
        nonlocal last_coverage, step_counter
        ax.clear()
        done = deployment.run_step()
        img = np.zeros((*deployment.grid.shape, 3))
        img[deployment.grid == -1] = [0.7, 0.7, 0.7]
        img[deployment.grid == 0] = [1, 1, 1]
        img[deployment.grid == 1] = [0, 0, 0]
        for robot in deployment.robots:
            x, y = robot.position
            ax.plot(y, x, 'ro', markersize=3)
        ax.imshow(img)
        coverage = deployment.count_coverage() / total_grid_without_obstacle * 100
        ax.set_title(f"Step {frame}: Coverage={coverage:.2f}%")
        ax.set_xlim(-0.5, AREA_SIZE[1] - 0.5)
        ax.set_ylim(-0.5, AREA_SIZE[0] - 0.5)
        ax.grid(True)

        # Check coverage improvement every 1000 steps
        step_counter += 1
        if step_counter >= coverage_check_interval:
            improvement = coverage - last_coverage
            print(f"[Coverage Check] Last: {last_coverage:.2f}%, Current: {coverage:.2f}%, Improvement: {improvement:.2f}%")
            if improvement < coverage_threshold:
                for robot in deployment.robots:
                    x, y = robot.position
                    final_position.append([x,y])
                with open("Final_Position.txt", "w") as f: 
                    for x, y in final_position:
                        f.write(f"{x:.2f}, {y:.2f}\n")

                print("Coverage improvement below threshold. Stopping simulation.")
                end_time = time.time()
                ani.event_source.stop()
            last_coverage = coverage
            step_counter = 0  # Reset counter

        if done:
            ani.event_source.stop()

    ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, interval=1)
    plt.show()
end_time  = 0

if __name__ == '__main__':
    start_time = time.time()
    deployment = ConstrainDeployment()
    visualize(deployment)
    taken_time = end_time - start_time
    print(f"Total time taken {taken_time:.2f} seconds")
