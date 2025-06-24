import matplotlib.pyplot as plt
import numpy as np

# Map labels
maps = ['map1', 'map2', 'map3', 'map4', 'map5', 'map6', 'map7', 'map8', 'map9']

# Coverage per node for each strategy
greedy = [0.576, 1.196, 0.789, 1.336, 1.063, 0.982, 0.883, 0.726, 1.481]
square = [0.531, 1.083, 0.768, 1.252, 1.079, 0.939, 0.851, 0.738, 1.561]
constrained = [0.643, 1.238, 0.662, np.nan, 0.633, 0.567, 0.272, 0.476, 0.968]
hexagonal = [0.712, 1.480, 0.987, 1.612, 1.293, 1.143, 1.045, 0.893, 1.953]

# Setup
x = np.arange(len(maps))  # Maps as x-axis
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5 * width, greedy, width, label='Greedy')
ax.bar(x - 0.5 * width, square, width, label='Square Grid')
ax.bar(x + 0.5 * width, constrained, width, label='Constrained')
ax.bar(x + 1.5 * width, hexagonal, width, label='Hexagonal')

ax.set_ylabel('Coverage per Node')
ax.set_xlabel('Maps')
ax.set_title('Coverage per Node Across Deployment Strategies')
ax.set_xticks(x)
ax.set_xticklabels(maps)
ax.legend(loc='upper left')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
