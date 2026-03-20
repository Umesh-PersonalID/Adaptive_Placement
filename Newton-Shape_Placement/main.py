from geometry.disk_reg import create_region
from optimization.outer_min_m import minimize_number_of_disks
from utils.plotting import plot_sol
import numpy as np
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    total_runs = 10
    i = 0
    with tqdm(total=total_runs, desc="Overall Progress", unit="run") as pbar:
        for trial in range(10): 
            pbar.set_description(f"Region {i}, Trial {trial}")
            region = create_region(i)
            fixed_radius = 2
            centers, m_opt, best_g = minimize_number_of_disks(region, fixed_radius, region_idx=i,trial=trial)            
            #save final centers
            results_root = Path(__file__).resolve().parents[0] / "Results"
            region_output_dir = results_root / f"region_{i}"
            region_output_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(region_output_dir / f"final_centers_trial_{trial}.txt", centers, delimiter=',', fmt='%.6f')
            pbar.update(1) 