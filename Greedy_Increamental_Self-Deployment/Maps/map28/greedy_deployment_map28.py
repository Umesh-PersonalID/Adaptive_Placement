import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import math
from matplotlib.colors import ListedColormap, Normalize
import time
import pandas as pd
from shapely.geometry import Point

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../Newton-Shape_Placement")
    )
)

from geometry.regions import all_regions


overlap = 0
total_grid_without_obstacle = 0
coverage_list = []
overlap_list = []
start_time = 0
end_time = 0


def polygon_to_grid(region, resolution=1.0):
    min_x, min_y, max_x, max_y = region.bounds
    size_x = int(np.ceil((max_x - min_x) / resolution)) + 1
    size_y = int(np.ceil((max_y - min_y) / resolution)) + 1

    grid = np.ones((size_x, size_y)) * -2  # outside region

    for x in range(size_x):
        for y in range(size_y):
            p = Point(min_x + x * resolution, min_y + y * resolution)

            if region.covers(p):
                grid[x, y] = -1   # unknown free space
            else:
                grid[x, y] = -2   # outside polygon

    return grid, (min_x, min_y)

class Robot:
    def __init__(self, id, position, sensor_range, deployment=None):
        self.id = id
        self.position = position
        self.sensor_range = sensor_range
        self.deployed = False
        self.deployment = deployment

    def sense(self, occupancy_grid, reachability_check):
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
                if distance_sq <= sensor_range_sq:
                    if occupancy_grid[nx][ny] == -1:
                        reachability_check[nx][ny] = 0

                if distance_sq <= sensor_range_sq:
                    if occupancy_grid[nx][ny] == -1:
                        occupancy_grid[nx][ny] = 0
                    if hasattr(self, 'deployment'):
                        self.deployment.sensing_map[nx][ny] += 1


class GreedyDeployment:
    def __init__(
        self,
        occupancy_grid,
        num_robots=150,
        sensor_range=15,
        resolution=1.0,
        origin=(0.0, 0.0),
        map_name='Map 25',
        region=None,
    ):
        self.grid_size = occupancy_grid.shape
        self.num_robots = num_robots
        self.sensor_range = sensor_range
        self.resolution = resolution
        self.origin = origin
        self.region = region
        self.occupancy_grid = occupancy_grid
        self.reachability_grid = np.zeros(self.grid_size)
        self.reachability_check = occupancy_grid.copy()
        self.sensing_map = np.zeros(self.grid_size)
        self.robots = [Robot(i, (0, 0), sensor_range, self) for i in range(num_robots)]
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.suptitle(f'Greedy Robot Deployment Algorithm - {map_name}')
        
        self.cmap = ListedColormap(['black','white','blue','red'])
        self.norm = Normalize(-1, 1)
        
        self.im = None
        self.robot_scatter = None
        self.iteration = 0
        self.animation_frames = []

        self.robots[0].position = (14,14)
        self.robots[0].deployed = True
        self.robots[0].sense(self.occupancy_grid, self.reachability_check)
        self.update_reachability()
        
        self.init_visualization()
      
    def init_visualization(self):
        min_x, min_y = self.origin
        max_x = min_x + (self.grid_size[0] - 1) * self.resolution
        max_y = min_y + (self.grid_size[1] - 1) * self.resolution
        self.im = self.ax1.imshow(
            self.occupancy_grid.T,
            cmap=self.cmap,
            norm=self.norm,
            origin='lower',
            interpolation='none',
            extent=[min_x, max_x, min_y, max_y],
        )
        self.ax1.set_title('Coverage Map')
        self.ax1.set_xlabel('X position (units)')
        self.ax1.set_ylabel('Y position (units)')
        self.ax1.set_aspect('equal', adjustable='box')

        if self.region is not None:
            exterior_x, exterior_y = self.region.exterior.xy
            self.ax1.plot(exterior_x, exterior_y, color='yellow', linewidth=1.0, alpha=0.9, label='Polygon')
            for interior in self.region.interiors:
                int_x, int_y = interior.xy
                self.ax1.plot(int_x, int_y, color='yellow', linewidth=0.8, alpha=0.8)
        
        deployed_positions = [robot.position for robot in self.robots if robot.deployed]
        if deployed_positions:
            x, y = zip(*deployed_positions)
            x_plot = [self.origin[0] + xi * self.resolution for xi in x]
            y_plot = [self.origin[1] + yi * self.resolution for yi in y]
            self.robot_scatter = self.ax1.scatter(x_plot, y_plot, c='green', s=10, label='Robots')
        
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
        overlap_mask = self.sensing_map > 1
        vis_grid[overlap_mask] = 1
        
        self.im.set_array(vis_grid.T)
        
        deployed_positions = [robot.position for robot in self.robots if robot.deployed]
        if deployed_positions:
            x, y = zip(*deployed_positions)
            x_plot = [self.origin[0] + xi * self.resolution for xi in x]
            y_plot = [self.origin[1] + yi * self.resolution for yi in y]
            if self.robot_scatter:
                self.robot_scatter.set_offsets(np.c_[x_plot, y_plot])
            else:
                self.robot_scatter = self.ax1.scatter(x_plot, y_plot, c='green', s=10, label='Robots')
        
        iterations = range(1, len(coverage_list)+1)
        self.coverage_line.set_data(iterations, coverage_list)
        self.overlap_line.set_data(iterations, overlap_list)
        
        self.animation_frames.append([self.im, self.robot_scatter, self.coverage_line, self.overlap_line])
        plt.pause(0.01)
    
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

      frontier = set()
      directions = [(1,0), (-1,0), (0,1), (0,-1),
                    (1,1), (1,-1), (-1,1), (-1,-1)]

      for x in range(self.grid_size[0]):
          for y in range(self.grid_size[1]):

              if self.reachability_grid[x, y] != 1:
                  continue

              if self.occupancy_grid[x, y] == -2:
                  continue

              for dx, dy in directions:
                  nx, ny = x + dx, y + dy

                  if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:

                      if self.occupancy_grid[nx, ny] == -1:
                          frontier.add((x, y))
                          break

      return list(frontier)
  
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
                    self.update_visualization()
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
        total_grid_without_obstacle = np.sum(self.occupancy_grid != -2)

    def coverage(self):
        covered = np.sum(self.occupancy_grid == 0)
        coverage_list.append(covered / total_grid_without_obstacle * 100)
        return covered
    

    def create_animation(self):
        ani = animation.ArtistAnimation(self.fig, self.animation_frames, interval=150, 
                                       blit=True, repeat_delay=1000)
        return ani

    def save_positions(self, filename):
        deployed_robots = [robot for robot in self.robots if robot.deployed]
        disk_radius_units = self.sensor_range * self.resolution

        with open(filename, 'w') as f:
            f.write(f"# Greedy Deployment for {self.fig._suptitle.get_text().split(' - ')[-1]}\n")
            f.write(f"# Disk radius: {disk_radius_units:.6f}\n")
            f.write(f"# Number of disks: {len(deployed_robots)}\n")
            f.write("# Format: x,y\n")

            for robot in deployed_robots:
                grid_x, grid_y = robot.position
                world_x = self.origin[0] + grid_x * self.resolution
                world_y = self.origin[1] + grid_y * self.resolution
                f.write(f"{world_x:.6f},{world_y:.6f}\n")


# Run the deployment
start_time = time.time()

TARGET_DISK_RADIUS = 2.0
GRID_RESOLUTION = 0.1
NUM_ROBOTS = 200
MAP_NAME = 'Map 28'
MAP_INDEX = None
TARGET_COVERAGE_PERCENT = 97.0

region = all_regions()
selected_region = region[5]

grid, grid_origin = polygon_to_grid(selected_region, resolution=GRID_RESOLUTION)
sensor_range_cells = max(1, math.ceil(TARGET_DISK_RADIUS / GRID_RESOLUTION))

deployment = GreedyDeployment(
    occupancy_grid=grid,
    num_robots=NUM_ROBOTS,
    sensor_range=sensor_range_cells,
    resolution=GRID_RESOLUTION,
    origin=grid_origin,
    map_name=MAP_NAME,
    region=selected_region,
)
deployment.count_desired_area()

for _ in range(deployment.num_robots):
    if coverage_list and coverage_list[-1] >= TARGET_COVERAGE_PERCENT:
        break
    
    if not deployment.deploy_next():
        break

end_time = time.time()
coverage = coverage_list[-1] if coverage_list else 0.0
overlap_pct = overlap_list[-1] if overlap_list else 0.0
deployed_nodes = sum(robot.deployed for robot in deployment.robots)

deployment.fig.text(0.02, 0.95, f'Coverage: {coverage:.2f}%', fontsize=14, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

deployment.fig.text(0.02, 0.85, f'Nodes: {deployed_nodes}', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
deployment.fig.text(0.02, 0.80, f'Time: {end_time - start_time:.2f}s', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
deployment.fig.text(0.02, 0.75, f'Disk radius: {TARGET_DISK_RADIUS} units ({sensor_range_cells} cells)', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
deployment.fig.savefig('Coverage_map28.png', dpi=300, bbox_inches='tight')
deployment.save_positions('position_map28.txt')


df = pd.DataFrame({
    'Metric': ['Coverage', 'Overlap %', 'Deployed Nodes', 'Time (s)'],
    'Value': [coverage, overlap_pct, deployed_nodes, end_time - start_time]
})
print(df.to_string(index=False))

ani = deployment.create_animation()
plt.show()