import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import math
from matplotlib.colors import ListedColormap
from Grid_generator_map6 import grid
import time

overlap = 0
total_grid_without_obstacle = 0
coverage_list = []
overlap_list = []
start_time = 0
end_time = 0

class Robot:
    def __init__(self, id, position, sensor_range, deployment=None):
        self.id = id
        self.position = position
        self.sensor_range = sensor_range
        self.deployed = False
        self.deployment = deployment

    def sense(self, occupancy_grid, reachability_check):
        global global_sensor_range
        global_sensor_range = self.sensor_range
        x, y = self.position
        sensor_range_sq = self.sensor_range ** 2
        min_x = max(0, x - 2*self.sensor_range)
        max_x = min(occupancy_grid.shape[0] - 1, x + 2*self.sensor_range)
        min_y = max(0, y - 2*self.sensor_range)
        max_y = min(occupancy_grid.shape[1] - 1, y + 2*self.sensor_range)

        for nx in range(int(min_x), int(max_x) + 1):
            for ny in range(int(min_y), int(max_y) + 1):
                dx = x - nx
                dy = y - ny
                distance_sq = dx**2 + dy**2
                if distance_sq <= 3 * sensor_range_sq:
                    if occupancy_grid[nx][ny] == -1:
                        reachability_check[nx][ny] = 0

                if distance_sq <= sensor_range_sq:
                    if occupancy_grid[nx][ny] == -1:
                        occupancy_grid[nx][ny] = 0
                    if hasattr(self, 'deployment'):
                        self.deployment.sensing_map[nx][ny] += 1
                


class GreedyDeployment:
    def __init__(self, grid_size=(300, 300), num_robots=150, sensor_range=15):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.sensor_range = sensor_range
        self.occupancy_grid = grid
        self.reachability_grid = np.zeros(grid_size)
        self.reachability_check = grid.copy()
        self.sensing_map = np.zeros(grid_size)
        self.robots = [Robot(i, (0, 0), sensor_range, self) for i in range(num_robots)]
        
       
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.suptitle('Greedy Robot Deployment Algorithm')
        
        self.cmap = ListedColormap(['white', 'blue', 'red'])
        self.norm = plt.Normalize(-1, 1)
        
        self.im = None
        self.robot_scatter = None
        self.iteration = 0
        self.animation_frames = []

        self.robots[0].position = (self.sensor_range,self.sensor_range)
        self.robots[0].deployed = True
        self.robots[0].sense(self.occupancy_grid, self.reachability_check)
        self.update_reachability()
        
        self.init_visualization()

    def init_visualization(self):
        self.im = self.ax1.imshow(self.occupancy_grid, cmap=self.cmap, norm=self.norm, 
                                origin='lower', interpolation='none')
        self.ax1.set_title('Coverage Map')
        self.ax1.set_xlabel('X position')
        self.ax1.set_ylabel('Y position')
        
        deployed_positions = [robot.position for robot in self.robots if robot.deployed]
        if deployed_positions:
            x, y = zip(*deployed_positions)
            self.robot_scatter = self.ax1.scatter(x, y, c='green', s=10, label='Robots')
        
        self.ax2.set_title('Performance Metrics')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Percentage')
        self.ax2.set_xlim(0, self.num_robots)
        self.ax2.set_ylim(0, 100)
        self.coverage_line, = self.ax2.plot([], [], 'b-', label='Coverage %')
        self.overlap_line, = self.ax2.plot([], [], 'r-', label='Overlap %')
        self.ax2.legend()
        
        plt.tight_layout()

    def update_visualization(self):
        vis_grid = np.copy(self.occupancy_grid)
        # Mark overlap areas (where sensing_map > 1)
        overlap_mask = self.sensing_map > 1
        vis_grid[overlap_mask] = 1  # Will be shown as red
        
        self.im.set_array(vis_grid)
        
        deployed_positions = [robot.position for robot in self.robots if robot.deployed]
        if deployed_positions:
            x, y = zip(*deployed_positions)
            if self.robot_scatter:
                self.robot_scatter.set_offsets(np.c_[y, x])
            else:
                self.robot_scatter = self.ax1.scatter(y, x, c='green', s=10, label='Robots')
        
        iterations = range(1, len(coverage_list)+1)
        self.coverage_line.set_data(iterations, coverage_list)
        self.overlap_line.set_data(iterations, overlap_list)
        
        self.animation_frames.append([self.im, self.robot_scatter, self.coverage_line, self.overlap_line])
        
        plt.pause(0.01)  # Pause to allow the plot to update

    def update_reachability(self):
        self.reachability_grid.fill(0)
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

        queue = deque()
        for robot in self.robots:
            if robot.deployed:
                x, y = robot.position
                self.reachability_grid[x, y] = 1
                queue.append((x, y))

        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if self.reachability_check[nx, ny] == 0 and self.reachability_grid[nx, ny] == 0:
                        self.reachability_grid[nx, ny] = 1
                        queue.append((nx, ny))


    def frontier_cells(self):
        frontier = []
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.reachability_grid[x, y] == 1:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and
                            self.occupancy_grid[nx, ny] == -1):
                            frontier.append((nx, ny))
                            break
        return frontier

    def coverage_gain(self, x, y):
        gain = 0
        for dx in range(-self.sensor_range, self.sensor_range + 1):
            for dy in range(-self.sensor_range, self.sensor_range + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if self.occupancy_grid[nx, ny] == -1:
                        gain += 1
        return gain

    def deploy_next(self):
        frontier = self.frontier_cells()
        if not frontier:
            return False

        best_gain = -1
        best_pos = None

        for (x, y) in frontier:
            gain = self.coverage_gain(x, y)
            if gain > best_gain:
                best_gain = gain
                best_pos = (x, y)

        if best_pos:
            for robot in self.robots:
                if not robot.deployed:
                    robot.position = best_pos
                    robot.deployed = True
                    robot.sense(self.occupancy_grid, self.reachability_check)
                    self.update_reachability()
                    self.calculate_overlap(best_pos)
                    self.coverage()
                    self.update_visualization()  # Update visualization after each deployment
                    return True
        return False

    def calculate_overlap(self, pos):
        global overlap, overlap_list
        x, y = pos
        r_sq = self.sensor_range ** 2
        for dx in range(-self.sensor_range, self.sensor_range + 1):
            for dy in range(-self.sensor_range, self.sensor_range + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if dx*dx + dy*dy <= r_sq:
                        if self.occupancy_grid[nx, ny] == 0 and self.sensing_map[nx, ny] > 1:
                            overlap += 1
        overlap_list.append(overlap / total_grid_without_obstacle * 100)

    def count_desired_area(self):
        global total_grid_without_obstacle
        total_grid_without_obstacle = np.sum(self.occupancy_grid == -1)

    def coverage(self):
        covered = np.sum(self.occupancy_grid == 0)
        coverage_list.append(covered / total_grid_without_obstacle * 100)
        return covered

    def create_animation(self):
        ani = animation.ArtistAnimation(self.fig, self.animation_frames, interval=150, 
                                       blit=True, repeat_delay=1000)
        return ani


# Run the deployment
start_time  = time.time()
deployment = GreedyDeployment(grid_size=(300, 300), num_robots=84, sensor_range=15)
deployment.count_desired_area()

for _ in range(deployment.num_robots):
    if not deployment.deploy_next():
        break


coverage = coverage_list[-1]
overlap_pct = overlap_list[-1]
deployed_nodes = sum(robot.deployed for robot in deployment.robots)

import pandas as pd
df = pd.DataFrame({
    'Metric': ['Coverage', 'Overlap %', 'Deployed Nodes'],
    'Value': [coverage, overlap_pct, deployed_nodes]
})
print(df.to_string(index=False))

end_time = time.time()
print(end_time - start_time)

ani = deployment.create_animation()

plt.show()
