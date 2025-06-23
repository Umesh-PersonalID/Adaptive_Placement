import numpy as np
import matplotlib.pyplot as plt

# Simulation settings
NUM_NODES = 60
SENSOR_RANGE = 4.0
COMM_RANGE = 8.0
AREA_SIZE = (25, 25)
K_NEIGHBORS = 4
TIME_STEPS = 10000
DELTA_T = 0.006

# Force parameters
K_COVER = 13.0
K_DEGREE = 60.0
Q = 1
V = 0.1
MASS = 1.0

# Initialize positions in center
np.random.seed(42)
center_x, center_y = AREA_SIZE[0] / 2, AREA_SIZE[1] / 2
positions = np.random.rand(NUM_NODES, 2) * 2 + [center_x - 1, center_y - 1]
velocities = np.zeros_like(positions)

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def get_neighbors(pos):
    neighbors = []
    for i in range(len(pos)):
        n = []
        for j in range(len(pos)):
            if i != j and distance(pos[i], pos[j]) <= COMM_RANGE:
                n.append(j)
        neighbors.append(n)
    return neighbors

def compute_forces(pos, neighbors):
    forces = np.zeros_like(pos)
    for i in range(len(pos)):
        nbrs = neighbors[i]
        dists = [(j, distance(pos[i], pos[j])) for j in nbrs]
        sorted_nbrs = sorted(dists, key=lambda x: x[1])
        critical = [j for j, _ in sorted_nbrs[:K_NEIGHBORS]]

        if len(nbrs) > K_NEIGHBORS:
            for j, d in dists:
                if d == 0: continue
                if d < 2 * SENSOR_RANGE:
                    f = K_COVER / (d ** 2 + 1e-5)
                    dir = (pos[i] - pos[j]) / d
                    forces[i] += f * dir
        else:
            for j, d in dists:
                if d == 0: continue
                dir = (pos[j] - pos[i]) / d
                if d < 2 * SENSOR_RANGE:
                    f = K_COVER / (d ** 2 + 1e-5)
                    forces[i] -= f * dir
                if j in critical and (d >= Q * COMM_RANGE or d >= (Q-0.1) * COMM_RANGE):
                    f = K_DEGREE / ((COMM_RANGE - d) ** 2 + 1e-5)
                    forces[i] += f * dir
    return forces

def is_velocity_stable(velocity):
    for i in range(len(velocity)):
        for j in range(len(velocity[0])):
            if abs(velocity[i][j]) > 0.04:
                return False
    return True

def all_have_k_neighbors(neighbors):
    return all(len(nbrs) == K_NEIGHBORS for nbrs in neighbors)

positions_over_time = [positions.copy()]
for step in range(TIME_STEPS):
    nbrs = get_neighbors(positions)
    forces = compute_forces(positions, nbrs)
    accels = forces / MASS
    velocities = (1 - V) * velocities + accels * DELTA_T
    if all_have_k_neighbors(nbrs) or is_velocity_stable(velocities):
        print(f"Converged at step {step}")
        break
    positions += velocities * DELTA_T
    positions = np.clip(positions, [0, 0], AREA_SIZE)
    positions_over_time.append(positions.copy())

# Plot final positions
plt.figure(figsize=(6, 6))
plt.xlim(0, AREA_SIZE[0])
plt.ylim(0, AREA_SIZE[1])
plt.title("Final Sensor Node Placement")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect('equal')
plt.grid(True)

for pos in positions:
    circle = plt.Circle((pos[0], pos[1]), SENSOR_RANGE, color='blue', alpha=0.3)
    plt.gca().add_patch(circle)
plt.scatter(positions[:, 0], positions[:, 1], color='red')
plt.show()
