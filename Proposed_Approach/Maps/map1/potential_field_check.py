import numpy as np
import matplotlib.pyplot as plt
from grid_generator_map1 import grid  # Assuming this is available
import sys
import matplotlib.animation as animation
import time

overlap = 0
count = 0
total_grid_without_obstacle = 0
global_sensor_range = 0
coverage_list = []
overlap_list = []


# Simulation settings
NUM_NODES = 137
SENSOR_RANGE = 15.0
COMM_RANGE = 40.0
AREA_SIZE = (300, 300)
K_NEIGHBORS = 6
TIME_STEPS = 10000
DELTA_T = 0.05

# Force parameters
K_COVER = 20.0
K_DEGREE = 40.0
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

positions = []
with open("position.txt", "r") as f:
    for line in f:
        if ',' in line:
            x_str, y_str = line.strip().split(',')
            try:
                x = float(x_str)
                y = float(y_str)
                positions.append(np.array([x, y]))
            except ValueError:
                continue

positions = np.array(positions)
if len(positions) != NUM_NODES:
    print(f"[Warning] Loaded {len(positions)} positions but expected {NUM_NODES}. Check NUM_NODES or file.")
velocities = np.zeros_like(positions)

final_position = []

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


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
    pass

class Robot:
    def __init__(self, id, position, sensor_range):
        self.id = id
        self.position = position
        self.sensor_range = sensor_range

    def sense(self, occupancy_grid):
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

    def run_step(self):
        neighbors = get_neighbors(self.positions)
        self.update_occupancy()
        forces = compute_forces(self.positions, neighbors)
        accels = forces / MASS
        self.velocities = (1 - V) * self.velocities + accels * DELTA_T
        
        # Step clipping
        step = self.velocities * DELTA_T
        max_step = 3.0  
        step_norms = np.linalg.norm(step, axis=1)
        scaling_factors = np.minimum(1, max_step / (step_norms + 1e-8))
        step = step * scaling_factors[:, np.newaxis]
        
        new_positions = self.positions + step
        new_positions = np.clip(new_positions, [0, 0], AREA_SIZE)
        self.positions = new_positions
        for i, robot in enumerate(self.robots):
          robot.position = self.positions[i]
        self.update_occupancy()
    
    def count_coverage(self):
        return np.sum(self.grid == 0)



def visualize(deployment):
    deployment.count_desired_area()
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
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
    
    ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, interval=150)
    plt.show()
    pass

if __name__ == '__main__':
    start_time = time.time()
    deployment = ConstrainDeployment()
    visualize(deployment)
    end_time = time.time()
    taken_time = end_time - start_time
    print(f"Total time taken {taken_time:.2f} seconds")