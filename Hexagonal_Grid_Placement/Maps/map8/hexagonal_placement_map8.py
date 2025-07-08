import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
import matplotlib.animation as animation
from collections import deque
import math
from generate_grid_map8 import grid
import time
#Global Variable
overlap = 0
count = 0
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
        self.occupancy_grid = grid
        self.reachability_grid = np.zeros(grid_size)
        
        self.robots = []
        self.initialize_robots()
        if self.robots:
            self.robots[0].position = (15,15)
            self.robots[0].deployed = True
            self.update_grids()
    
    def initialize_robots(self):
        for i in range(self.num_robots):
            self.robots.append(Robot(i, (self.sensor_range,self.sensor_range), self.sensor_range))
    
    
    
    
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
        radius = int(np.sqrt(1.73) * robot_range)

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
        
        
    def next_center(self):
        global count
        centers = []
        radius = self.sensor_range
        rows = int(((2*(self.grid_size[1] - 2*radius))//(3*radius) +  1))  
        cols =  int(self.grid_size[0] // (1.73 * radius)) 
 
        dx = np.sqrt(3)  * radius
        dy = 1.5 * radius
        for row in range(rows):
            for col in range(cols):
                y = radius + int(col * dx + (radius if row % 2 else 0))
                x = radius + int(row * dy)
                if (x <= len(self.occupancy_grid) and y <= len(self.occupancy_grid[0]) and self.occupancy_grid[x][y] != 1):
                    centers.append((x, y))
        count+=1
        print(centers)
        return centers[count % len(centers)]


    def deploy_next_robot(self):
        global count
        """Deploy the next robot to optimal position in unknown space"""
        target_pos = self.next_center()
        
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
    def __init__(self, id, position, sensor_range):
        self.id = id
        self.position = position
        self.sensor_range = sensor_range
        self.deployed = False
    
    def sense(self, occupancy_grid):
        global global_sensor_range
        
        global_sensor_range = self.sensor_range
        x, y = self.position
        sensor_range_sq = self.sensor_range ** 2
        overlap_increased = 0
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
                    if occupancy_grid[nx,ny] == 0:
                        overlap_increased += 1
                    if occupancy_grid[nx, ny] == -1:
                        occupancy_grid[nx, ny] = 0


def count_desired_area():
        global total_grid_without_obstacle
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x] == -1:
                    total_grid_without_obstacle += 1
# Visualization
def visualize(deployment):
    global total_grid_without_obstacle, overlap
    
    fig, ax = plt.subplots(figsize=(12, 8))
    def update(frame):
        ax.clear()
    
        # if frame == deployment.num_robots:
        #     end_time = time.time()
        #     print(f"Total Time : {end_time - start_time}")
        #     ani.event_source.stop() 
        #     plt.close(fig)           
        #     return
        
        if frame > 0:
            deployment.run_step()
            # deployment.save_occupancy_grid_to_file("occupancy_hexagonal.txt")
                
        img = np.zeros((grid.shape[0],grid.shape[1], 3))

        unknown_mask = deployment.occupancy_grid == -1
        img[unknown_mask] = [0.7, 0.7, 0.7]
        no_of_zeros = deployment.count_coverage()
        
        free_mask = deployment.occupancy_grid == 0
        img[free_mask] = [1, 1, 1]
        
        occupied_mask = deployment.occupancy_grid == 1
        img[occupied_mask] = [0, 0, 0]
        
        
        ax.imshow(img)  # Swap rows and columns

        
        deployed_count = 0
        for robot in deployment.robots:
            if robot.deployed:
                ax.plot(robot.position[1], robot.position[0], 'ro', markersize=5)
                deployed_count += 1
            else:
                ax.plot(robot.position[1], robot.position[0], 'bo', markersize=5, alpha=0.3)
        
        for robot in deployment.robots:
            if robot.deployed:
                circle = plt.Circle((robot.position[1], robot.position[0]), 15, 
                                  color='r', fill=False, alpha=0.2)
                img[robot.position[0], robot.position[1]] = [1, 0, 0]
                ax.add_patch(circle)
        
        ax.set_title(f"Step {frame} - {deployed_count}/{deployment.num_robots} deployed\n coverage = {no_of_zeros/total_grid_without_obstacle*100} \n"
                     f"overlap area(%) =  {overlap/total_grid_without_obstacle*100}")
        ax.set_xlim(-0.5, deployment.grid_size[0]-0.5)
        ax.set_ylim(-0.5, deployment.grid_size[1]-0.5)
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    
    ani = animation.FuncAnimation(fig, update, frames=deployment.num_robots+1, 
                                interval=150, repeat=False)
    # ani.save('my_animation4.mp4', writer='ffmpeg', fps=3)
    plt.show()

# Create and run simulation
start_time = time.time()
count_desired_area()
end_time = time.time()
deployment = IncrementalDeployment(grid_size=(300,300), num_robots=98, sensor_range=15)
visualize(deployment)
