import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

# Map labels
maps = ['map1', 'map2', 'map3', 'map4', 'map5', 'map6', 'map7', 'map8', 'map9']

# Coverage per node for each strategy
greedy = [0.682, 1.474, 0.976, 1.645, 1.247, 1.097, 1.029, 0.918, 2.147]
square = [0.531, 1.083, 0.768, 1.252, 1.079, 0.939, 0.851, 0.738, 1.561]
constrained = [0.643, 1.238, 0.662, np.nan, 0.633, 0.567, 0.272, 0.476, 0.968]
hexagonal = [0.712, 1.480, 0.987, 1.612, 1.293, 1.143, 1.045, 0.893, 1.953]
hex_plus_constrained = [0.7166, 1.5143, 1.0036, 0.0000, 1.2915, 1.1561, 1.0810, 0.9028, 2.0098]

# Setup
x = np.arange(len(maps))  # Maps as x-axis
width = 0.15

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - 2 * width, greedy, width, label='Greedy')
ax.bar(x - width, square, width, label='Square Grid')
ax.bar(x, constrained, width, label='Constrained')
ax.bar(x + width, hexagonal, width, label='Hexagonal')
ax.bar(x + 2 * width, hex_plus_constrained, width, label='Hex + Constrained')

ax.set_ylabel('Coverage per Node')
ax.set_xlabel('Maps')
ax.set_title('Coverage per Node Across Deployment Strategies')
ax.set_xticks(x)
ax.set_xticklabels(maps)
ax.legend(loc='upper left')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

