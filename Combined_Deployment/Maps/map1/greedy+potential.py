# combined_deployment_constrained.py

import numpy as np
import matplotlib.pyplot as plt
from grid_generator_map1 import grid
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import deque
import time

GRID_SIZE = (300, 300)
NUM_ROBOTS = 158
SENSOR_RANGE = 15
COMM_RANGE = 28
K_NEIGHBORS = 6
Q = 1.026
V = 0.1
MASS = 1.0
DELTA_T = 0.05
MAX_STEP = 1.0
K_COVER = 12.5
K_DEGREE = 0.067 * K_COVER
RELAX_STEPS = 1500
COVERAGE_CHECK_INTERVAL = 500
CONVERGENCE_TOL = 0.1

coverage_list = []
overlap_list = []
total_grid_without_obstacle = 0

hex_placement = []
with open("position.txt", "r") as f:
    for line in f:
        line = line.strip().strip("()")
        if not line:
            continue
        x, y = map(float, line.split(","))
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
                    if dx*dx + dy*dy <= r_sq + 4:
                        if occupancy_grid[nx][ny] == -1:
                            occupancy_grid[nx][ny] = 0
                            reachability_check[nx][ny] = 0
                        self.deployment.sensing_map[nx][ny] += 1


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
        self.cmap = ListedColormap(["gray", "white", "red"])
        self.norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], self.cmap.N)
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

    def deploy_next(self):
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
        self.update_visualization()
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

    def compute_forces(self, neighbors):
        
        forces = np.zeros_like(self.positions)
        for i in range(NUM_ROBOTS):
            if not self.deployed_mask[i]:
                continue
            nbrs = neighbors[i]
            dists = [(j, self.distance(self.positions[i], self.positions[j])) for j in nbrs]
            sorted_nbrs = sorted(dists, key=lambda x: x[1])
            critical = [j for j, _ in sorted_nbrs[:K_NEIGHBORS]]
            if len(nbrs) > K_NEIGHBORS:
                for j, d in dists:
                    if d == 0:
                        continue
                    if d < COMM_RANGE:
                        f = (K_COVER) / (d ** 2 + 1e-5)
                        dir = (self.positions[i] - self.positions[j]) / d
                        forces[i] += f * dir
            else:
                for j, d in dists:
                    if d == 0:
                        continue
                    dir = (self.positions[j] - self.positions[i]) / d
                    if d <= COMM_RANGE:
                        f = (K_COVER + 2 * d) / (d ** 2 + 1e-5)
                        forces[i] -= f * dir
                    if j in critical and d >= (Q - 0.1) * COMM_RANGE:
                        f = K_DEGREE / ((COMM_RANGE - d) ** 2 + 1e-5)
                        forces[i] += f * dir
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

          # Re-sense and check coverage improvement
          if step % COVERAGE_CHECK_INTERVAL == 0:
              self.re_sense_all()
              curr_cov = self.count_coverage()
              if abs(curr_cov - last_cov) < CONVERGENCE_TOL:
                  no_improve_steps += 1
                  if no_improve_steps > 2:
                      break
              else:
                  no_improve_steps = 0
              last_cov = curr_cov


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

    def update_metrics(self):
        covered = np.sum(self.occupancy_grid == 0)
        total = total_grid_without_obstacle
        cov_pct = (covered / total * 100) if total > 0 else 0
        ov_pct = (np.sum(self.sensing_map > 1) / total * 100) if total > 0 else 0
        coverage_list.append(cov_pct)
        overlap_list.append(ov_pct)

    def update_visualization(self):
        vis_grid = self.occupancy_grid.copy()
        vis_grid[(self.occupancy_grid == 0) & (self.sensing_map > 1)] = 1
        self.im.set_array(vis_grid)

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
        plt.pause(0.01)


if __name__ == "__main__":
    start = time.time()
    deployment = HybridDeployment()

    # Deploy remaining robots greedily
    remaining = NUM_ROBOTS - len(hex_placement)
    for _ in range(remaining):
        current_coverage = deployment.count_coverage()
        if current_coverage >= 99.5:
            print(f"Coverage reached {current_coverage:.2f}%. Stopping deployment.")
            break

        if not deployment.deploy_next():
            print("No more frontiers. Stopping early.")
            break

    # Save final positions
    with open("Final_Position.txt", "w") as f:
        for r in deployment.robots:
            if r.deployed:
                f.write(f"{r.position[0]:.2f}, {r.position[1]:.2f}\n")

    end = time.time()
    print(f"Time: {end - start:.2f}s | Coverage: {coverage_list[-1]:.2f}% | Overlap: {overlap_list[-1]:.2f}%")
    plt.show()
