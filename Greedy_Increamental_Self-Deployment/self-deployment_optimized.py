import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import math


#Global Variable
overlap = 0
total_grid_without_obstacle = 0
global_sensor_range = 0
coverage_list = []
overlap_list = []

class IncrementalDeployment:
    def __init__(self, grid_size=(90, 70), num_robots=30, sensor_range=5):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.sensor_range = sensor_range
        self.current_step = 0
        self.sensing_map = np.zeros(grid_size)  # Tracks how many robots have sensed each cell
        
        self.occupancy_grid = np.full(grid_size, -1) 
        self.reachability_grid = np.zeros(grid_size)
        
        # for x in range(20, 55):
        #     for y in range(70, 120):
        #         self.occupancy_grid[x, y] = 1  
                
        # for x in range(120, 125):
        #     for y in range(110, 160):
        #         self.occupancy_grid[x, y] = 1 

        # for y in range(40, 65):
        #     for x in range(40, 150):
        #         self.occupancy_grid[x, y] = 1
                
        # for y in range(130, 165):
        #     for x in range(120, 200):
        #         self.occupancy_grid[x, y] = 1
        
        self.robots = []
        self.initialize_robots()
        
        if self.robots:
            self.robots[0].position = (self.sensor_range, self.sensor_range)
            self.robots[0].deployed = True
            self.update_grids()
    
    def initialize_robots(self):
        for i in range(self.num_robots):
            self.robots.append(Robot(i, (self.sensor_range, self.sensor_range), self.sensor_range,deployment=self))
    
    def count_desired_area(self):
        global total_grid_without_obstacle
        for y in range(len(self.occupancy_grid)):
            for x in range(len(self.occupancy_grid[0])):
                if self.occupancy_grid[y][x] == -1:
                    total_grid_without_obstacle += 1
    
    
    def update_grids(self):
        for robot in self.robots:
            if robot.deployed:
                robot.sense(self.occupancy_grid)
        
        self.update_reachability()
    
    
    def save_occupancy_grid_to_file(self, filename="occupancy_log.txt"):
        """Append current occupancy grid state to a text file."""
        with open(filename, "w") as f: 
            f.write(f"current overlap {overlap_list}\n")
            f.write(f"current coverage {coverage_list}\n")
            # for row in self.occupancy_grid:
            #     f.write(" ".join(map(str, row)) + "\n")
            # f.write("\n")  # Blank line between steps
        
        
    def update_reachability(self):
        self.reachability_grid.fill(0)  # Reset reachability grid
        
        robot_range = self.sensor_range
        radius = int(np.sqrt(0.5) * robot_range)

        for robot in self.robots:
            if robot.deployed:
                rx, ry = robot.position
                
                # Define square bounds around robot position
                min_x = max(0, rx - radius)
                max_x = min(self.grid_size[0] - 1, rx + radius)
                min_y = max(0, ry - radius)
                max_y = min(self.grid_size[1] - 1, ry + radius)
                
                # Loop over nearby cells within the bounding box
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        # if self.occupancy_grid[x, y] == 0:  # Only consider free space
                        #     dx = x - rx
                        #     dy = y - ry
                        #     distance_sq = dx*dx + dy*dy
                        #     if distance_sq <= 3* robot_range * robot_range:
                        self.reachability_grid[x, y] = 1
    
    
    def find_frontier_directions(self):
        frontier_directions = []
        directions = [(1,0), (-1,0), (0,1), (0,-1), 
                          (1,1), (1,-1), (-1,1), (-1,-1)]

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.reachability_grid[x, y] == 1: 
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and
                            self.occupancy_grid[nx, ny] == -1 and self.occupancy_grid[nx, ny] != 1):
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
                    elif self.occupancy_grid[nx][ny] == 0:
                        overlap_penalty += max(0, self.sensing_map[nx][ny])  # Already sensed once or more
        return gain - 0.4 * overlap_penalty  # Tune weight based on experiments
    
    def policy4_selection(self):
        """Select deployment location using improved frontier selection"""
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
                avg_distance = distance_from_others / max(1, sum(robot.deployed for robot in self.robots))

                score = gain + 0.1 * avg_distance  # Encourage spacing out

                if score > best_score:
                    best_score = score
                    best_position = (tx, ty)

        return best_position
    
    def calculate_overlap(self,target_pos):
        global overlap, overlap_list
        x, y = target_pos
        sensor_range_sq = global_sensor_range ** 2
        overlap_increased = 0
        min_x = max(0, x - global_sensor_range)
        max_x = min(self.occupancy_grid.shape[0] - 1, x + global_sensor_range)
        min_y = max(0, y - global_sensor_range)
        max_y = min(self.occupancy_grid.shape[1] - 1, y + global_sensor_range)

        for nx in range(int(min_x), int(max_x) + 1):
            for ny in range(int(min_y), int(max_y) + 1):
                dx = x - nx
                dy = y - ny
                distance_sq = dx**2 + dy**2
                if distance_sq <= sensor_range_sq:
                    if self.occupancy_grid[nx,ny] == 0:
                        overlap += 1
        overlap_list.append(overlap/total_grid_without_obstacle*100)
        pass
        
    def deploy_next_robot(self):
        """Deploy the next robot to optimal position in unknown space"""
        target_pos = self.policy4_selection()
        
        if target_pos is None:
            return False  
        
        for robot in self.robots:
            if not robot.deployed:
                robot.position = target_pos
                robot.deployed = True
                self.calculate_overlap(target_pos)
                return True
        return False
    
    def run_step(self):
        self.update_grids()
        success = self.deploy_next_robot()
        
        if success:
            self.current_step += 1
        return success
    
    def count_coverage(self):
        sume = 0
        global coverage_list
        for y in range(len(self.occupancy_grid)):
            for x in range(len(self.occupancy_grid[0])):
                if self.occupancy_grid[y][x] == 0:
                    sume += 1
        coverage_list.append(sume/total_grid_without_obstacle*100)
        return sume



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
                        self.deployment.sensing_map[nx][ny] += 1  # Track overlap

# Visualization
def visualize(deployment):
    global total_grid_without_obstacle, overlap
    deployment.count_desired_area()
    fig, ax = plt.subplots(figsize=(12, 8))
    def update(frame):
        ax.clear()
    
        if frame > 0:
            deployment.run_step()
            deployment.save_occupancy_grid_to_file("occupancy_log.txt")
                
        img = np.zeros((*deployment.grid_size, 3))
   
        unknown_mask = deployment.occupancy_grid == -1
        img[unknown_mask] = [0.7, 0.7, 0.7]
        no_of_zeros = deployment.count_coverage()
        
        free_mask = deployment.occupancy_grid == 0
        img[free_mask] = [1, 1, 1]
        
        occupied_mask = deployment.occupancy_grid == 1
        img[occupied_mask] = [0, 0, 0]
        
        ax.imshow(img, origin='lower')
  
        deployed_count = 0
        for robot in deployment.robots:
            if robot.deployed:
                ax.plot(robot.position[1], robot.position[0], 'ro', markersize=5)
                deployed_count += 1
            else:
                ax.plot(robot.position[0], robot.position[1], 'bo', markersize=5, alpha=0.3)
        
        for robot in deployment.robots:
            if robot.deployed:
                circle = plt.Circle((robot.position[1], robot.position[0]), 15, 
                                  color='r', fill=False, alpha=0.2)
                img[robot.position[1], robot.position[0]] = [1, 0, 0]
                ax.add_patch(circle)
        
        ax.set_title(f"Step {frame} - {deployed_count}/{deployment.num_robots} deployed\n coverage = {no_of_zeros/total_grid_without_obstacle*100} \n"
                     f"overlap area(%) =  {overlap/total_grid_without_obstacle*100}")
        ax.set_xlim(-0.5, deployment.grid_size[0]-0.5)
        ax.set_ylim(-0.5, deployment.grid_size[1]-0.5)
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    
    ani = animation.FuncAnimation(fig, update, frames=deployment.num_robots+5, 
                                interval=500, repeat=False)
    # ani.save('my_animation4.mp4', writer='ffmpeg', fps=3)
    plt.show()

# Create and run simulation
deployment = IncrementalDeployment(grid_size=(300, 300), num_robots=150, sensor_range=15)
visualize(deployment)