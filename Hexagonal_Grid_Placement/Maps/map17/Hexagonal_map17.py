
import math, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

# grid:  1 = obstacle (dark pixels),  -1 = unknown/free-to-cover (light pixels)
from grid_generator_map17 import grid

# -------------------- globals just for logging/plots --------------------
coverage_list, overlap_list = [], []
total_grid_without_obstacle = 0

def count_desired_area():
    """How many cells need to be covered (unknown = -1)."""
    global total_grid_without_obstacle
    total_grid_without_obstacle = int((grid == -1).sum())

# --------------------------- core classes ------------------------------
class Robot:
    def __init__(self, rid, pos, sensor_range):
        self.id = rid
        self.position = pos  # (row, col)
        self.sensor_range = sensor_range
        self.deployed = False

    def sense(self, occ):
        """Mark cells in sensor disk: -1 -> 0 (new coverage); 0 -> overlap."""
        r = self.sensor_range
        r2 = r*r
        cx, cy = self.position  # row, col

        rmin = max(0, cx - r)
        rmax = min(occ.shape[0] - 1, cx + r)
        cmin = max(0, cy - r)
        cmax = min(occ.shape[1] - 1, cy + r)

        new_covered = 0
        overlapped = 0

        for x in range(rmin, rmax + 1):
            dx = x - cx
            dx2 = dx*dx
            for y in range(cmin, cmax + 1):
                dy = y - cy
                if dx2 + dy*dy <= r2:
                    if occ[x, y] == -1:
                        occ[x, y] = 0       # newly covered
                        new_covered += 1
                    elif occ[x, y] == 0:
                        overlapped += 1      # already covered -> overlap
        return new_covered, overlapped


class IncrementalDeployment:
    """
    Places circles (robots) on a hexagonal *covering* lattice so there are no gaps.
    Spacing comes from the hex covering condition: a = sqrt(3)*R, vertical step = 1.5*R.
    """
    def __init__(self, grid_size=(300, 300), num_robots=200, sensor_range=15, overlap_alpha=1.0):
        """
        overlap_alpha <= 1.0. Use <1.0 to add extra overlap beyond the minimal hex covering.
        """
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.sensor_range = int(sensor_range)
        self.alpha = float(overlap_alpha)

        self.occupancy_grid = grid.copy()
        self.robots = []
        self.centers = self._build_hex_cover_centers()  # precompute centers
        self._init_robots()

        # metrics
        self.current_step = 0
        self.total_overlap = 0
        self.total_covered = 0

        # deploy the first one immediately if we have any
        if self.robots:
            self.robots[0].deployed = True
            new_cov, ov = self.robots[0].sense(self.occupancy_grid)
            self.total_covered += new_cov
            self.total_overlap += ov

    # ---------- placement: hexagonal covering over the free rectangle ----------
    def _build_hex_cover_centers(self):
        """
        Generate centers using a hexagonal *covering* lattice.
        Centers are kept inside the map and not inside obstacles.
        """
        H, W = self.grid_size
        r = self.sensor_range

        # Hex covering spacing (no holes):
        # Equilateral triangle side 'a' = sqrt(3)*r; vertical step = 1.5*r; row offset = a/2.
        a = math.sqrt(3) * r * self.alpha
        dx = a
        dy = 1.5 * r * self.alpha
        row_offset = dx / 2.0

        centers = []
        # Rows from r up to H - r (keep centers inside; circle may touch boundary)
        n_rows = int(math.floor((H - 2*r) / dy)) + 2
        for i in range(n_rows):
            cx = int(round(0.48*r + i * dy))
            y0 = 0 + (row_offset if (i % 2 == 1) else 0.0)  # staggered
            n_cols = int(math.floor((W - 2*r - (y0 - r)) / dx)) + 2
            for j in range(n_cols):
                cy = int(round(y0 + j * dx))
                if 0 <= cx < H and 0 <= cy < W:
                    # don’t place the center inside an obstacle
                    if self.occupancy_grid[cx, cy] != 1:
                        centers.append((cx, cy))
        return centers

    def _init_robots(self):
        # Create robots up to min(num_robots, len(centers))
        k = min(self.num_robots, len(self.centers))
        for i in range(k):
            self.robots.append(Robot(i, self.centers[i], self.sensor_range))

    def deploy_next_robot(self):
        """Deploy the next not-yet-deployed robot (greedy order = precomputed centers)."""
        for rb in self.robots:
            if not rb.deployed:
                rb.deployed = True
                new_cov, ov = rb.sense(self.occupancy_grid)
                self.total_covered += new_cov
                self.total_overlap += ov
                return True
        return False

    def run_step(self):
        ok = self.deploy_next_robot()
        if ok:
            self.current_step += 1
        return ok


# --------------------------- visualization ------------------------------
def visualize(deployment):
    fig, ax = plt.subplots(figsize=(12, 8))
    H, W = deployment.grid_size
    r = deployment.sensor_range

    def draw_frame(frame):
        # run deployment except on frame 0
        if frame > 0:
            deployment.run_step()

        # ---- draw (always) ----
        ax.clear()
        img = np.zeros((H, W, 3), dtype=float)

        # color map
        unknown_mask = deployment.occupancy_grid == -1
        free_mask    = deployment.occupancy_grid == 0
        obs_mask     = deployment.occupancy_grid == 1
        img[unknown_mask] = [0.75, 0.75, 0.75]
        img[free_mask]    = [1.0, 1.0, 1.0]
        img[obs_mask]     = [0.0, 0.0, 0.0]

        ax.imshow(img, origin="lower", interpolation="nearest")

        deployed = 0
        for rb in deployment.robots:
            if rb.deployed:
                deployed += 1
                ax.plot(rb.position[1], rb.position[0], 'o', ms=3)
                ax.add_patch(Circle((rb.position[1], rb.position[0]), r,
                                    fill=False, alpha=0.25))
            else:
                # faint center for not-yet-deployed (optional)
                ax.plot(rb.position[1], rb.position[0], '.', alpha=0.2)

        cov_pct = (deployment.total_covered / max(1, total_grid_without_obstacle)) * 100.0
        ovl_pct = (deployment.total_overlap / max(1, total_grid_without_obstacle)) * 100.0
        coverage_list.append(cov_pct); overlap_list.append(ovl_pct)

        ax.set_title(
            f"Step {frame}  deployed {deployed}/{len(deployment.robots)}\n"
            f"coverage = {cov_pct:.2f}%   overlap area = {ovl_pct:.2f}%"
        )
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.grid(True, ls='-', lw=0.4, alpha=0.4)

        # On the final frame, save after drawing
        if frame == len(deployment.robots):
            fig.savefig("final_deployment.png", dpi=300, bbox_inches="tight")
            ani.event_source.stop()
            plt.close(fig)
        return [ax]

    ani = animation.FuncAnimation(
        fig, draw_frame, frames=len(deployment.robots) + 1, interval=120, repeat=False
    )
    plt.show()


# ------------------------------- run ------------------------------------
start_time = time.time()
count_desired_area()
deployment = IncrementalDeployment(
    grid_size=(285, 300),    # H, W
    num_robots=9999,         # will stop when centers are exhausted
    sensor_range=15,         # circle radius in cells
    overlap_alpha=1.0        # <1.0 => more overlap, >1.0 not allowed for covering
)
visualize(deployment)

# dump deployed centers (rows, cols)
with open("position.txt", "w") as f:
    for rb in deployment.robots:
        if rb.deployed:
            x, y = rb.position
            f.write(f"{x},{y}\n")

print(f"Saved deployed robot positions to position.txt in {time.time()-start_time:.2f}s")