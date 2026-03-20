from objective.gradient import build_union
import math
import numpy as np
from pathlib import Path
from shapely.geometry import Point
from optimization.inner_solver import solve_inner_fixed_r
from utils.plotting import plot_sol
from utils.sampling import sample_points_in_polygon, project_points_to_polygon


def lower_bound_m(region, r):
    return int(np.ceil(region.area / (math.pi * r**2)))


def hex_lattice_points(region, r):
    minx, miny, maxx, maxy = region.bounds

    dx = 2 * r
    dy = np.sqrt(3) * r

    centers = []

    y = miny
    row = 0

    while y <= maxy:
        x_offset = 0 if row % 2 == 0 else r
        x = minx

        while x <= maxx:
            pt = np.array([x + x_offset, y])
            centers.append(pt)
            x += dx

        y += dy
        row += 1

    return np.array(centers)


def clip_to_region(centers, region):
    filtered = []
    for c in centers:
        if region.buffer(1e-8).contains(Point(c)):
            filtered.append(c)
    return np.array(filtered)

def uncovered_region(centers, r, region):
    union = build_union(centers, r)
    return region.difference(union)


def largest_uncovered_component(uncovered):
    if uncovered.is_empty:
        return None

    if uncovered.geom_type == "Polygon":
        return uncovered

    return max(uncovered.geoms, key=lambda g: g.area)


def farthest_point_in_uncovered(uncovered, centers, samples=300):
    pts = sample_points_in_polygon(uncovered, samples)

    max_dist = -1
    best_pt = None

    for p in pts:
        d = min(np.linalg.norm(p - c) for c in centers)
        if d > max_dist:
            max_dist = d
            best_pt = p

    return best_pt


def add_disk_largest_gap(centers, r, region):
    U = uncovered_region(centers, r, region)
    comp = largest_uncovered_component(U)

    if comp is None:
        return centers

    new_center = farthest_point_in_uncovered(comp, centers)

    if new_center is None:
        return centers

    return np.vstack([centers, new_center])


def minimize_number_of_disks(
    region,
    r,
    region_idx=0,
    tol=1,
    max_m=1600,
    local_refinements=3,
    trial=0
):
    target_area = region.area
    results_root = Path(__file__).resolve().parents[1] / "Results"
    region_output_dir = results_root / f"region_{region_idx}"

    m = lower_bound_m(region, r)

    lattice = hex_lattice_points(region, r)
    lattice = clip_to_region(lattice, region)

    if len(lattice) < m:
        m = len(lattice)

    centers = lattice[:m]

    while m <= max_m:
        x0 = centers.reshape(-1)
        x_opt, g_val = solve_inner_fixed_r(x0, region, r, m)
        centers = x_opt.reshape(m, 2)

        best_g = g_val
        best_centers = centers.copy()

        for _ in range(local_refinements):
            perturb = centers + np.random.normal(scale=0.3*r, size=centers.shape)
            perturb = project_points_to_polygon(perturb, region)
            x_opt, g_val = solve_inner_fixed_r(perturb.reshape(-1), region, r, m)

            if g_val < best_g:
                best_g = g_val
                best_centers = x_opt.reshape(m, 2)

        centers = best_centers

        #calculate overlap area
        union = build_union(centers, r)
        percentage_overlap = ((m * math.pi * (r**2)) - union.area) / (m * math.pi * (r**2)) * 100

        plot_sol(
            region,
            centers,
            r,
            m,
            best_g,
            save_dir=region_output_dir/f"trial_{trial}",
            overlap_area=percentage_overlap,
            save_name=f"m_{m}_trial_{trial}.png",
            show_plot=False
        )
        

        if best_g / target_area <= 0.01:
            return centers, m, best_g
    
 
        new_centers = add_disk_largest_gap(centers, r, region)
        if len(new_centers) == len(centers):
            print("No valid new center found. Stopping.")
            return centers, m, best_g

        centers = new_centers
        m = centers.shape[0]
    return None, None, float("inf")