from geometry.disk_reg import create_region
from optimization.outer_min_m import minimize_number_of_disks
from utils.plotting import plot_sol



if __name__ == "__main__":
    for i in range(4):
        region = create_region(i)
        fixed_radius = 2
        centers, m_opt, best_g = minimize_number_of_disks(region, fixed_radius, region_idx=i)
        print(f"Optimal m for region {i} is {m_opt} with uncovered area {best_g:.6f}")
