# Re-run the simulation after reset
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from copy import deepcopy
from collections import deque

# Simulated Annealing for Sensor Placement

class SensorGrid:
    def __init__(self, grid_size=(50, 50), sensor_range=5, budget=150):
        self.grid_size = grid_size
        self.sensor_range = sensor_range
        self.budget = budget
        self.grid = np.full(grid_size, -1)  # -1 = unknown, 0 = free, 1 = sensor
        self.coverage_map = np.zeros(grid_size)
        self.sensors = []

    def add_sensor(self, position):
        if position not in self.sensors and len(self.sensors) < self.budget:
            self.sensors.append(position)

    def remove_sensor(self, position):
        if position in self.sensors:
            self.sensors.remove(position)

    def move_sensor(self, old_pos, new_pos):
        if old_pos in self.sensors and new_pos not in self.sensors:
            self.sensors.remove(old_pos)
            self.sensors.append(new_pos)

    def compute_coverage(self):
        self.coverage_map = np.zeros(self.grid_size)
        for x, y in self.sensors:
            for dx in range(-self.sensor_range, self.sensor_range + 1):
                for dy in range(-self.sensor_range, self.sensor_range + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                        if dx*dx + dy*dy <= self.sensor_range ** 2:
                            self.coverage_map[nx, ny] = 1
        return np.sum(self.coverage_map)

    def energy(self):
        # Lower energy = better coverage (we negate it)
        return -self.compute_coverage()

    def random_position(self):
        return (random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1))


def simulated_annealing(grid_size=(50, 50), sensor_range=5, budget=150, t0=300, tf=10, alpha=0.5, r=10):
    current = SensorGrid(grid_size, sensor_range, budget)
    for _ in range(budget):
        current.add_sensor(current.random_position())

    best = deepcopy(current)
    E_min = current.energy()

    t = t0

    while t > tf:
        for _ in range(r):
            candidate = deepcopy(current)
            if random.random() < 0.5:
                # Remove a sensor
                if candidate.sensors:
                    candidate.remove_sensor(random.choice(candidate.sensors))
            else:
                # Move or add a sensor
                if candidate.sensors and random.random() < 0.5:
                    candidate.move_sensor(random.choice(candidate.sensors), candidate.random_position())
                else:
                    candidate.add_sensor(candidate.random_position())

            if len(candidate.sensors) > budget:
                continue

            delta_E = candidate.energy() - current.energy()
            if delta_E < 0 or random.random() < math.exp(-delta_E / t):
                current = candidate

            if current.energy() < E_min:
                best = deepcopy(current)
                E_min = current.energy()

        t *= alpha

    return best


# Run SA
final_solution = simulated_annealing(grid_size=(300, 300), sensor_range=15, budget= 150)

# Visualize
fig, ax = plt.subplots(figsize=(8, 8))
coverage_img = np.zeros((*final_solution.grid_size, 3))
coverage_img[final_solution.coverage_map == 1] = [1, 1, 1]
coverage_img[final_solution.coverage_map == 0] = [0.7, 0.7, 0.7]

ax.imshow(coverage_img, origin='lower')
for x, y in final_solution.sensors:
    ax.plot(y, x, 'ro', markersize=4)
    circle = plt.Circle((y, x), final_solution.sensor_range, color='r', fill=False, alpha=0.2)
    ax.add_patch(circle)

ax.set_title(f"Simulated Annealing Final Deployment\nCoverage: {np.sum(final_solution.coverage_map)} cells | Sensors: {len(final_solution.sensors)}")
ax.set_xlim(0, final_solution.grid_size[1])
ax.set_ylim(0, final_solution.grid_size[0])
plt.grid(True)
plt.show()
