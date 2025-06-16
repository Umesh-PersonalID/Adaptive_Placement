import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# Define grid size and sensor range
GRID_SIZE = (300, 300)
SENSOR_RANGE = 20
NUM_ROBOTS = 100

# Create a sample grid with -1 for unknown, 0 for free, 1 for obstacle
grid = np.full(GRID_SIZE, -1)
grid[100:120, 150:170] = 1  # Add a block of obstacles for testing

# Global Variables
overlap = 0
total_grid_without_obstacle = 0
global_sensor_range = 0
coverage_list = []
overlap_list = []

class Robot:
    def __init__(self, id, position, sensor_range, deployment=None):
        self.id = id
        self.position = position
        self.sensor_range = sensor_range
        self.deployed = False
        self.deployment = deployment

    def sense(self, occupancy_grid):
        global global_sensor_range
        global_sensor_range = self.sensor_range
        x, y = self.position
        sensor_range_sq = self.sensor_range ** 2
        min_x = max(0, x - self.sensor_range)
        max_x = min(occupancy_grid.shape[0] - 1, x + self.sensor_range)
        min_y = max(0, y - self.sensor_range)
        max_y = min(occupancy_grid.shape[1] - 1, y + self.sensor_range)

        for nx in range(int(min_x), int(max_x) + 1):
            for ny in range(int(min_y), int(max_y) + 1):
                dx = x - nx
                dy = y - ny
                distance_sq = dx**2 + dy**2
                if distance_sq <= sensor_range_sq:
                    if occupancy_grid[nx][ny] == -1:
                        occupancy_grid[nx][ny] = 0
                    if hasattr(self, 'deployment'):
                        self.deployment.sensing_map[nx][ny] += 1

class IncrementalDeployment:
    def __init__(self, grid_size=(300, 300), num_robots=150, sensor_range=20):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.sensor_range = sensor_range
        self.current_step = 0
        self.sensing_map = np.zeros(grid_size)
        self.occupancy_grid = grid
        self.reachability_grid = np.zeros(grid_size)
        self.robots = []
        self.initialize_robots()

        if self.robots:
            self.robots[0].position = (self.sensor_range, self.sensor_range)
            self.robots[0].deployed = True
            self.update_grids()

    def initialize_robots(self):
        for i in range(self.num_robots):
            self.robots.append(Robot(i, (self.sensor_range, self.sensor_range), self.sensor_range, deployment=self))

    def update_grids(self):
        for robot in self.robots:
            if robot.deployed:
                robot.sense(self.occupancy_grid)
        self.update_reachability()

    def update_reachability(self):
        self.reachability_grid.fill(0)
        radius = self.sensor_range
        for robot in self.robots:
            if robot.deployed:
                rx, ry = robot.position
                min_x = max(0, rx - radius)
                max_x = min(self.grid_size[0] - 1, rx + radius)
                min_y = max(0, ry - radius)
                max_y = min(self.grid_size[1] - 1, ry + radius)
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        self.reachability_grid[x, y] = 1

    def find_frontier_directions(self):
        frontier_directions = []
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.reachability_grid[x, y] == 1:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and
                            self.occupancy_grid[nx, ny] == -1):
                            frontier_directions.append(((x, y), (dx, dy)))
                            break
        return frontier_directions

    def calculate_coverage_gain(self, position):
        x, y = position
        gain = 0
        overlap_penalty = 0
        for dx in range(-self.sensor_range, self.sensor_range + 1):
            for dy in range(-self.sensor_range, self.sensor_range + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]):
                    if self.occupancy_grid[nx][ny] == -1:
                        gain += 1
        
        return gain

    def policy4_selection(self):
        frontier_directions = self.find_frontier_directions()
        if not frontier_directions:
            return None
        best_score = -np.inf
        best_position = None
        for (fx, fy), (dx, dy) in frontier_directions:
            tx, ty = fx + dx, fy + dy
            if (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1] and
                self.occupancy_grid[tx][ty] == -1):
                gain = self.calculate_coverage_gain((tx, ty))
                distance_from_others = sum(
                    math.hypot(robot.position[0] - tx, robot.position[1] - ty)
                    for robot in self.robots if robot.deployed
                )
                print(gain, tx,ty)
                avg_distance = distance_from_others / max(1, sum(robot.deployed for robot in self.robots))
                score = gain + 0.1 * avg_distance
                if score > best_score:
                    best_score = score
                    best_position = (tx, ty)
        return best_position

    def deploy_next_robot(self):
        target_pos = self.policy4_selection()
        if target_pos is None:
            return False
        for robot in self.robots:
            if not robot.deployed:
                robot.position = target_pos
                robot.deployed = True
                return True
        return False

    def run_step(self):
        self.update_grids()
        success = self.deploy_next_robot()
        if success:
            self.current_step += 1
        return success

# Animate the deployment
def visualize(deployment):
    fig, ax = plt.subplots(figsize=(8, 8))
    def update(frame):
        ax.clear()
        if frame > 0:
            deployment.run_step()
        img = np.zeros((*deployment.grid_size, 3))
        unknown_mask = deployment.occupancy_grid == -1
        img[unknown_mask] = [0.7, 0.7, 0.7]
        free_mask = deployment.occupancy_grid == 0
        img[free_mask] = [1, 1, 1]
        occupied_mask = deployment.occupancy_grid == 1
        img[occupied_mask] = [0, 0, 0]
        ax.imshow(img, origin='lower')
        for robot in deployment.robots:
            if robot.deployed:
                ax.plot(robot.position[1], robot.position[0], 'ro', markersize=3)
                circle = plt.Circle((robot.position[1], robot.position[0]), SENSOR_RANGE, color='r', fill=False, alpha=0.2)
                ax.add_patch(circle)
        ax.set_title(f"Step {frame}")
        ax.set_xlim(-0.5, deployment.grid_size[1]-0.5)
        ax.set_ylim(-0.5, deployment.grid_size[0]-0.5)
        ax.grid(False)
    ani = animation.FuncAnimation(fig, update, frames=deployment.num_robots + 5,
                                  interval=100, repeat=False)
    plt.show()

# Run deployment and visualize
deployment = IncrementalDeployment(grid_size=GRID_SIZE, num_robots=NUM_ROBOTS, sensor_range=SENSOR_RANGE)
visualize(deployment)
