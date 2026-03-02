# combined_deployment_constrained.py

import numpy as np
import matplotlib.pyplot as plt
from grid_generator_map9 import grid
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import deque
import time
import math
GRID_SIZE = (147,300)
NUM_ROBOTS = 200
SENSOR_RANGE = 15
COMM_RANGE = math.sqrt(3) * SENSOR_RANGE
K_NEIGHBORS = 6
Q = 0.92  # q_attract safety factor
V = 0.1
MASS = 1.0
DELTA_T = 0.05
MAX_STEP = 0.05
K_COVER = 13.5
K_DEGREE = 2 * K_COVER
RELAX_STEPS = 6500
COVERAGE_CHECK_INTERVAL = 500
CONVERGENCE_TOL = 0.1
EPSILON = 1e-5  # avoid division by zero
coverage_list = []
overlap_list = []
total_grid_without_obstacle = 0

# Stopping condition parameters
COST_THRESHOLD = 10.0  # Stop if cost exceeds this value
cost_epsilon = 0.01   # Small epsilon for cost function
cost_list = []
previous_coverage = 0.0
previous_overlap = 0.0

hex_placement = []
with open("position.txt", "r") as f:
    for num,line in enumerate(f):
        line = line.strip().strip("()")
        if not line:
            continue
        x, y = map(float, line.split(","))
        if num%2 == 0:
            hex_placement.append((int(x), int(y)))


class Robot:
    def __init__(self, id, position, sensor_range, deployment=None):
        self.id = id
        self.position = np.array(position, dtype=float)
        self.sensor_range = sensor_range
        self.deployed = False
        self.deployment = deployment

    def sense(self, occupancy_grid, reachability_check):
        x, y = int(self.position[0]), int(self.position[1])
        r_sq = self.sensor_range ** 2
        for dx in range(-self.sensor_range, self.sensor_range + 1):
            for dy in range(-self.sensor_range, self.sensor_range + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < occupancy_grid.shape[0] and 0 <= ny < occupancy_grid.shape[1]:
                    # Check if cell is within sensor range: (x−xi)²+(y−yi)² ≤ R_s²
                    if dx*dx + dy*dy <= r_sq:
                        if occupancy_grid[nx][ny] == -1:
                            occupancy_grid[nx][ny] = 0  # newly covered
                            reachability_check[nx][ny] = 0
                        
                        if self.deployment is not None:
                            self.deployment.sensing_map[nx][ny] += 1  # measure overlap


class HybridDeployment:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.num_robots = NUM_ROBOTS
        self.sensor_range = SENSOR_RANGE
        self.occupancy_grid = grid.copy()
        self.reachability_check = grid.copy()
        self.sensing_map = np.zeros(GRID_SIZE)
        self.robots = [Robot(i, (0, 0), SENSOR_RANGE, self) for i in range(NUM_ROBOTS)]
        self.positions = np.zeros((NUM_ROBOTS, 2))
        self.velocities = np.zeros((NUM_ROBOTS, 2))
        self.deployed_mask = np.zeros(NUM_ROBOTS, dtype=bool)

        # Vis
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.cmap = ListedColormap(["gray", "white", "black", "red"])
        self.norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], self.cmap.N)

        self.im = None
        self.robot_scatter = None

        self.count_desired_area()
        self.init_visualization()
        self.initialize_hexagonal_deployment()

    def count_desired_area(self):
        global total_grid_without_obstacle
        total_grid_without_obstacle = np.sum(self.occupancy_grid == -1)

    def init_visualization(self):
        self.im = self.ax1.imshow(self.occupancy_grid, cmap=self.cmap, norm=self.norm, origin='lower', interpolation='none')
        self.ax1.set_title('Coverage Map')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')

        self.ax2.set_title('Performance Metrics')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Percentage (%)')
        self.ax2.set_xlim(0, NUM_ROBOTS)
        self.ax2.set_ylim(0, 100)
        self.coverage_line, = self.ax2.plot([], [], 'b-', label='Coverage %')
        self.overlap_line, = self.ax2.plot([], [], 'r-', label='Overlap %')
        self.ax2.legend()
        plt.tight_layout()

    def initialize_hexagonal_deployment(self):
        for i, pos in enumerate(hex_placement):
            if i >= NUM_ROBOTS:
                break
            robot = self.robots[i]
            robot.position = np.array(pos, dtype=float)
            robot.deployed = True
            self.deployed_mask[i] = True
            self.positions[i] = robot.position
            robot.sense(self.occupancy_grid, self.reachability_check)
        self.update_reachability()
        self.update_metrics()
        self.update_visualization()

    def update_reachability(self):
        self.reachability_grid = np.zeros(self.grid_size)
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        queue = deque()
        for r in self.robots:
            if r.deployed:
                x, y = r.position.astype(int)
                self.reachability_grid[x, y] = 1
                queue.append((x, y))
        while queue:
            x, y = queue.popleft()
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and
                    self.reachability_check[nx, ny] == 0 and self.reachability_grid[nx, ny] == 0):
                    self.reachability_grid[nx, ny] = 1
                    queue.append((nx, ny))

    def frontier_cells(self):
        """Find frontier cells adjacent to unknown areas"""
        frontier = []
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.reachability_grid[x, y] == 1 and self.occupancy_grid[x, y] == 0:
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and
                            self.occupancy_grid[nx, ny] == -1):
                            frontier.append((x, y))
                            break
        return frontier

    def coverage_gain(self, x, y):
        """Estimate new coverage if robot is placed at (x, y)"""
        gain = 0
        r = self.sensor_range
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = int(x + dx), int(y + dy)
                if (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]):
                    if self.occupancy_grid[nx, ny] == -1 and dx*dx + dy*dy <= r*r:
                        gain += 1
        return gain

    def save_final_results(self):
        with open("Final_Position.txt", "w") as f:
            for r in self.robots:
                if r.deployed:
                    f.write(f"{r.position[0]:.2f}, {r.position[1]:.2f}\n")

        fig, ax = plt.subplots(figsize=(8, 8))
        vis_grid = self.occupancy_grid.copy()
        vis_grid[(self.occupancy_grid == 0) & (self.sensing_map > 1)] = 2
        cmap = ListedColormap(["gray", "white", "black", "red"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

        ax.imshow(vis_grid, cmap=cmap, norm=norm, origin='lower')

        deployed_pos = [r.position for r in self.robots if r.deployed]
        if deployed_pos:
            x, y = zip(*deployed_pos)
            ax.scatter(y, x, c='blue', s=10, label='Robots')

        final_overlap = overlap_list[-1] if overlap_list else 0
        final_coverage = self.count_coverage()
        ax.set_title(f"Final Deployment\nCoverage: {final_coverage:.2f}% | Overlap: {final_overlap:.2f}%")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.tight_layout()
        plt.savefig("Final_Deployment.png", dpi=300)
        print("Final_Deployment.png saved.")
        plt.close('all')

    def save_initial_results(self):
        """Save the initial deployment visualization (after hexagonal placement)"""
        fig, ax = plt.subplots(figsize=(8, 8))
        vis_grid = self.occupancy_grid.copy()
        vis_grid[(self.occupancy_grid == 0) & (self.sensing_map > 1)] = 2
        cmap = ListedColormap(["gray", "white", "black", "red"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

        ax.imshow(vis_grid, cmap=cmap, norm=norm, origin='lower')

        deployed_pos = [r.position for r in self.robots if r.deployed]
        if deployed_pos:
            x, y = zip(*deployed_pos)
            ax.scatter(y, x, c='blue', s=10, label='Robots')

        initial_coverage = self.count_coverage()
        initial_overlap = (np.sum(self.sensing_map > 1) / total_grid_without_obstacle * 100) if total_grid_without_obstacle > 0 else 0
        ax.set_title(f"Initial Hexagonal Deployment\nCoverage: {initial_coverage:.2f}% | Overlap: {initial_overlap:.2f}%")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.tight_layout()
        plt.savefig("Initial_Deployment.png", dpi=300)
        print("Initial_Deployment.png saved.")
        plt.close('all')

    def deploy_next(self):
        global previous_coverage, previous_overlap

        """Greedy deployment: place next robot at best frontier location"""
        frontier = self.frontier_cells()
        if not frontier:
            return False

        # Find position with maximum coverage gain
        best_pos = max(frontier, key=lambda p: self.coverage_gain(p[0], p[1]))

        # Find next undeployed robot
        robot = None
        idx = -1
        for i, r in enumerate(self.robots):
            if not r.deployed:
                robot = r
                idx = i
                break
        if not robot:
            return False

        # Deploy robot
        robot.position = np.array(best_pos, dtype=float)
        robot.deployed = True
        self.deployed_mask[idx] = True
        self.positions[idx] = robot.position
        robot.sense(self.occupancy_grid, self.reachability_check)
        self.update_reachability()
        # Relax ALL currently deployed robots

        self.relax_all_robots(steps=RELAX_STEPS)

        # Recompute sensing and metrics
        self.re_sense_all()
        self.update_metrics()
        
        # Update visualization (show animation)
        self.update_visualization()
        
        # Calculate cost and check stopping condition
        cost = self.calculate_cost()
        deployed_count = sum(1 for r in self.robots if r.deployed)
        
        print(f"Node {deployed_count}: Cost = {cost:.4f}, Coverage = {previous_coverage:.2f}%, Overlap = {previous_overlap:.2f}%")
        
        # Check stopping condition
        if cost > COST_THRESHOLD and deployed_count > len(hex_placement):  # Only check after hex placement
            print(f"\nCost {cost:.4f} exceeds threshold {COST_THRESHOLD}")
            print(f"Removing last deployed node (Node {deployed_count})...")
            
            # Find and remove the last deployed robot
            last_deployed_robot = None
            for r in reversed(self.robots):
                if r.deployed:
                    last_deployed_robot = r
                    break
            
            if last_deployed_robot:
                # Remove the robot
                last_deployed_robot.deployed = False
                last_deployed_robot.position = np.array([0, 0], dtype=float)
                
                # Update deployed mask
                robot_idx = last_deployed_robot.id
                if robot_idx < len(self.deployed_mask):
                    self.deployed_mask[robot_idx] = False
                    self.positions[robot_idx] = np.array([0, 0], dtype=float)
                
                print(f"Removed robot {last_deployed_robot.id} from position")
                
                # Recompute everything without the removed robot
                self.re_sense_all()
                self.update_metrics()
                
                # Recalculate cost after removal (should be better now)
                final_coverage = self.count_coverage()
                final_overlap = (np.sum(self.sensing_map > 1) / total_grid_without_obstacle * 100) if total_grid_without_obstacle > 0 else 0
                final_deployed = sum(1 for r in self.robots if r.deployed)
                
                print(f"After removal - Nodes: {final_deployed}, Coverage: {final_coverage:.2f}%, Overlap: {final_overlap:.2f}%")
                
                # Remove the last cost entry since we rejected that node
                if cost_list:
                    cost_list.pop()
                
                # Update previous values to current state
                previous_coverage = final_coverage
                previous_overlap = final_overlap
            
            print(f"Deployment completed with optimal configuration.")
            self.save_final_results()
            return False

        final_coverage = self.count_coverage()
        if final_coverage >= 99.9:
            print(f"\nCoverage threshold reached: {final_coverage:.2f}%")
            self.save_final_results()
            return False
        return True

    def get_neighbors(self):
        neighbors = []
        for i in range(NUM_ROBOTS):
            if not self.deployed_mask[i]:
                neighbors.append([])
                continue
            n = []
            for j in range(NUM_ROBOTS):
                if i != j and self.deployed_mask[j]:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    if d <= COMM_RANGE:
                        n.append(j)
            neighbors.append(n)
        return neighbors

    def compute_wall_force(self, pos):
        """
        Compute a repulsive force from environment boundaries.
        Stronger when close to wall, zero when far.
        """
        x, y = pos

        force = np.array([0.0, 0.0])
        margin = 1
        
        
        if x < margin:
            force[0] += K_COVER / (x + EPSILON)
            

        if x > self.grid_size[0] - margin:
            dx = self.grid_size[0] - x
            force[0] -= K_COVER / (dx + EPSILON) 
            
        if y < margin:
            force[1] += K_COVER / (y + EPSILON)
            print(x,y)
            

        if y > self.grid_size[1] - margin:
            dy = self.grid_size[1] - y
            force[1] -= K_COVER / (dy + EPSILON)

        return force


    def compute_forces(self, neighbors):
        forces = np.zeros_like(self.positions)
        # --- Step 1: Robot-Robot Forces ---
        
        for i in range(NUM_ROBOTS):
            if not self.deployed_mask[i]: 
                continue
            nbrs = neighbors[i]
            if len(nbrs) == 0:
                continue

            dists = [(j, float(self.distance(self.positions[i], self.positions[j]))) for j in nbrs]
    
            if len(nbrs) >= K_NEIGHBORS:
                sorted_nbrs = sorted(dists, key=lambda x: x[1])
                critical = [j for j,_ in sorted_nbrs[:K_NEIGHBORS]]
            else:
                critical = []  # No critical connections if under-connected

            for j, d_ij in dists:
                u_ij = (self.positions[j] - self.positions[i]) / (d_ij + EPSILON)

                # Always repel
                f_repel = -K_COVER / (d_ij**2 + EPSILON)
                forces[i] += f_repel * u_ij

                
                if j in critical and d_ij > Q * COMM_RANGE:
                    f_attract = -K_DEGREE / (d_ij**2 + EPSILON)
                    forces[i] -= f_attract * u_ij  

        for i in range(NUM_ROBOTS):
            if self.deployed_mask[i]:
                wall_force = self.compute_wall_force(self.positions[i])
                forces[i] += wall_force

        return forces

    def is_velocity_stable(self):
        return np.all(np.abs(self.velocities[self.deployed_mask]) < 0.04)

    def all_have_k_neighbors(self, neighbors):
        return all(len(nbrs) >= K_NEIGHBORS for nbrs in neighbors if len(nbrs) > 0)

    def relax_all_robots(self, steps=RELAX_STEPS):
      last_cov = coverage_list[-1] if coverage_list else 0
      no_improve_steps = 0

      for step in range(steps):
          neighbors = self.get_neighbors()
          forces = self.compute_forces(neighbors)
          accels = forces / MASS

          # Update velocity
          self.velocities[self.deployed_mask] = \
              (1 - V) * self.velocities[self.deployed_mask] + accels[self.deployed_mask] * DELTA_T

          # Compute displacement
          disp = self.velocities * DELTA_T
          norms = np.linalg.norm(disp, axis=1, keepdims=True)
          scale = np.minimum(MAX_STEP / np.maximum(norms, 1e-8), 1.0)
          clipped = disp * scale
          new_positions = self.positions + clipped
          new_positions = np.clip(new_positions, [0, 0], [GRID_SIZE[0]-1, GRID_SIZE[1]-1])
          # Save original positions to revert if blocked
          original_positions = self.positions.copy()

          # Check for obstacles and revert if needed
          for i in range(NUM_ROBOTS):
              if self.deployed_mask[i]:
                  x, y = map(int, new_positions[i])
                  if self.occupancy_grid[x][y] == 1:
                      # Revert to old position and stop movement
                      new_positions[i] = original_positions[i]
                      self.velocities[i] = 0
                  else:
                      # Valid move
                      self.robots[i].position = new_positions[i]

          # Update robot positions after all checks
          self.positions = new_positions

          if step % 10 == 0:  
              self.update_visualization()


    def distance(self,p1, p2):
      return np.linalg.norm(p1 - p2)
    
    def re_sense_all(self):
        self.sensing_map.fill(0)
        self.occupancy_grid = grid.copy()
        for r in self.robots:
            if r.deployed:
                r.sense(self.occupancy_grid, self.reachability_check)
        self.update_reachability()

    def count_coverage(self):
        covered = np.sum(self.occupancy_grid == 0)
        return (covered / total_grid_without_obstacle * 100) if total_grid_without_obstacle > 0 else 0

    def calculate_cost(self):
        """Calculate cost function: (change in overlap)/(change in coverage) + epsilon"""
        global previous_coverage, previous_overlap, cost_list
        
        current_coverage = self.count_coverage()
        current_overlap = (np.sum(self.sensing_map > 1) / total_grid_without_obstacle * 100) if total_grid_without_obstacle > 0 else 0
        
        # Calculate changes
        delta_coverage = current_coverage - previous_coverage
        delta_overlap = current_overlap - previous_overlap
        
        # Calculate cost function
        if abs(delta_coverage) < 1e-6:  # Avoid division by very small numbers
            cost = float('inf')  # If no coverage improvement, cost is infinite
        else:
            cost = (delta_overlap / delta_coverage) + cost_epsilon
        
        # Update previous values
        previous_coverage = current_coverage
        previous_overlap = current_overlap
        
        # Store cost
        cost_list.append(cost)
        
        return cost

    def update_metrics(self):
        covered = np.sum(self.occupancy_grid == 0)
        total = total_grid_without_obstacle
        cov_pct = (covered / total * 100) if total > 0 else 0
        ov_pct = (np.sum(self.sensing_map > 1) / total * 100) if total > 0 else 0
        coverage_list.append(cov_pct)
        overlap_list.append(ov_pct)

    def update_visualization(self):
        vis_grid = self.occupancy_grid.copy()
        vis_grid[(self.occupancy_grid == 0) & (self.sensing_map > 1)] = 2
        if self.im is not None:
            self.im.set_array(vis_grid)
        final_coverage = coverage_list[-1] if coverage_list else self.count_coverage()
        final_overlap = overlap_list[-1] if overlap_list else 0
        
        # Update the title with coverage and overlap values
        self.ax1.set_title(f"Coverage: {final_coverage:.2f}% | Overlap: {final_overlap:.2f}%")
        final_coverage = coverage_list[-1] if coverage_list else self.count_coverage()

        deployed_pos = [r.position for r in self.robots if r.deployed]
        if deployed_pos:
            x, y = zip(*deployed_pos)
            if self.robot_scatter:
                self.robot_scatter.set_offsets(np.c_[y, x])
            else:
                self.robot_scatter = self.ax1.scatter(y, x, c='blue', s=10, zorder=5)

        iterations = list(range(1, len(coverage_list) + 1))
        self.coverage_line.set_data(iterations, coverage_list)
        self.overlap_line.set_data(iterations, overlap_list)

        if len(iterations) > 0:
            self.ax2.relim()
            self.ax2.autoscale_view()
        plt.pause(0.0001) 


if __name__ == "__main__":
    start = time.time()
    deployment = HybridDeployment()
    
    # Initialize baseline metrics after hexagonal placement
    deployment.re_sense_all()
    previous_coverage = deployment.count_coverage()
    previous_overlap = (np.sum(deployment.sensing_map > 1) / total_grid_without_obstacle * 100) if total_grid_without_obstacle > 0 else 0
    
    # Save initial deployment image
    deployment.save_initial_results()
    
    print(f"Initial after hex placement - Coverage: {previous_coverage:.2f}%, Overlap: {previous_overlap:.2f}%")
    print(f"Starting cost-based deployment with threshold: {COST_THRESHOLD}")
    print("=" * 60)

    # Deploy remaining robots greedily with cost-based stopping
    remaining = NUM_ROBOTS - len(hex_placement)
    for i in range(remaining):
        if not deployment.deploy_next():
            print("Deployment stopped.")
            break

    # Save final positions
    with open("Final_Position.txt", "w") as f:
        for r in deployment.robots:
            if r.deployed:
                f.write(f"{r.position[0]:.2f}, {r.position[1]:.2f}\n")

    end = time.time()
    final_coverage = coverage_list[-1] if coverage_list else deployment.count_coverage()
    final_overlap = overlap_list[-1] if overlap_list else 0
    final_cost = cost_list[-1] if cost_list else 0
    deployed_nodes = sum(1 for r in deployment.robots if r.deployed)
    
    print("=" * 60)
    print(f"FINAL RESULTS:")
    print(f"Time: {end - start:.2f}s")
    print(f"Deployed Nodes: {deployed_nodes}")
    print(f"Coverage: {final_coverage:.2f}%")
    print(f"Overlap: {final_overlap:.2f}%")
    print(f"Final Cost: {final_cost:.4f}")
    print(f"Cost Threshold: {COST_THRESHOLD}")
    
    if cost_list:
        print(f"Cost evolution: {[f'{c:.3f}' for c in cost_list[-5:]]}")  # Show last 5 costs
    
    plt.show()