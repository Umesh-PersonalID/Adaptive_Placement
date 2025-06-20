# Re-run after reset

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from simanneal import Annealer

# Sensor coverage problem using Simulated Annealing (SA)

GRID_SIZE = (50, 50)
SENSOR_RADIUS = 4
NUM_SENSORS = 70

def get_covered_cells(x, y, radius, grid_size):
    """Return all cells covered by a sensor at (x, y)."""
    covered = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                if dx ** 2 + dy ** 2 <= radius ** 2:
                    covered.append((nx, ny))
    return covered


class SensorPlacement(Annealer):
    def __init__(self, state, grid_size, radius):
        self.grid_size = grid_size
        self.radius = radius
        super(SensorPlacement, self).__init__(state)

    def move(self):
        """Randomly move one sensor to a new position."""
        idx = random.randint(0, len(self.state) - 1)
        new_pos = (random.randint(0, self.grid_size[0] - 1),
                   random.randint(0, self.grid_size[1] - 1))
        old_pos = self.state[idx]
        self.state[idx] = new_pos
        new_coverage = self.energy()
        self.state[idx] = old_pos
        old_coverage = self.energy()
        self.state[idx] = new_pos  # Accept for real after comparison
        return new_coverage - old_coverage

    def energy(self):
        """Return negative of total coverage (maximize coverage)."""
        covered = set()
        for x, y in self.state:
            covered.update(get_covered_cells(x, y, self.radius, self.grid_size))
        return -len(covered)


# Initial random sensor placement
initial_state = [(random.randint(0, GRID_SIZE[0] - 1), random.randint(0, GRID_SIZE[1] - 1)) for _ in range(NUM_SENSORS)]
sa = SensorPlacement(initial_state, GRID_SIZE, SENSOR_RADIUS)
sa.set_schedule(sa.auto(minutes=0.2))
sa.copy_strategy = "slice"

best_state, best_energy = sa.anneal()

# Visualization
coverage_map = np.zeros(GRID_SIZE)
for x, y in best_state:
    for nx, ny in get_covered_cells(x, y, SENSOR_RADIUS, GRID_SIZE):
        coverage_map[nx, ny] = 1

fig, ax = plt.subplots(figsize=(10, 10))
img = np.zeros((*GRID_SIZE, 3))
img[coverage_map == 1] = [1, 1, 1]  # covered = white
img[coverage_map == 0] = [0.7, 0.7, 0.7]  # uncovered = gray
ax.imshow(img, origin='lower')

for x, y in best_state:
    ax.plot(y, x, 'ro', markersize=3)
    circle = plt.Circle((y, x), SENSOR_RADIUS, color='r', fill=False, alpha=0.4)
    ax.add_patch(circle)

ax.set_title(f"Simulated Annealing Sensor Placement\nCoverage: {int(-best_energy)} cells | Sensors: {NUM_SENSORS}")
ax.set_xlim(0, GRID_SIZE[1])
ax.set_ylim(0, GRID_SIZE[0])
plt.grid(True)
plt.show()
