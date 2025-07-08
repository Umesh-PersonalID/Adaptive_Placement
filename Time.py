import matplotlib.pyplot as plt
import numpy as np

# Map labels
maps = ['map1', 'map2', 'map3', 'map4', 'map5', 'map6', 'map7', 'map8', 'map9']

# Time (in seconds) for each strategy
greedy_time = [350.03, 76.75, 178.03, 54.30, 107.95, 124.29, 149.32, 136.04, 25.22]
square_time = [64.51, 22.60, 35.92, 18.72, 23.11, 26.50, 31.02, 36.34, 13.68]
constrained_time = [2091.01, 842.62, 767.30, np.nan, 545.68, 1131.00, 257.71, 645.96, 502.58]
hexagonal_time = [41.14, 15.01, 25.28, 12.19, 17.76, 21.21, 22.43, 26.95, 8.84]

# Setup
x = np.arange(len(maps))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - 1.5 * width, greedy_time, width, label='Greedy')
bars2 = ax.bar(x - 0.5 * width, square_time, width, label='Square Grid')
bars3 = ax.bar(x + 0.5 * width, constrained_time, width, label='Constrained')
bars4 = ax.bar(x + 1.5 * width, hexagonal_time, width, label='Hexagonal')

# Add value labels on top of each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

for bar_group in [bars1, bars2, bars3, bars4]:
    add_labels(bar_group)

ax.set_ylabel('Time (s)')
ax.set_xlabel('Maps')
ax.set_title('Execution Time Across Deployment Strategies')
ax.set_xticks(x)
ax.set_xticklabels(maps)
ax.legend(loc='upper right')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
