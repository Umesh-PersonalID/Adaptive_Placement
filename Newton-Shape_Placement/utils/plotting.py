import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from shapely.geometry import Point
from shapely.ops import unary_union
from pathlib import Path


def plot_sol(region, centers, r, m, uncovered_area, save_dir=None, save_name=None, show_plot=True):
    fig, ax = plt.subplots(figsize=(6, 6))

    # --- Plot region ---
    x, y = region.exterior.xy
    ax.fill(x, y, color="orange", alpha=0.4)

    for hole in region.interiors:
        hx, hy = hole.xy
        ax.fill(hx, hy, color="white")

    # --- Disk union ---
    disks = [Point(c).buffer(r, resolution=128) for c in centers]
    union_disks = unary_union(disks)

    # --- Uncovered region ---
    uncovered = region.difference(union_disks)

    if not uncovered.is_empty:
        if uncovered.geom_type == "Polygon":
            ux, uy = uncovered.exterior.xy
            ax.fill(ux, uy, color="red", alpha=0.6)
        elif uncovered.geom_type == "MultiPolygon":
            for poly in uncovered.geoms:
                ux, uy = poly.exterior.xy
                ax.fill(ux, uy, color="red", alpha=0.6)

    # --- Plot disks ---
    for i, c in enumerate(centers):
        circle = Circle(c, r, fill=False, linewidth=1)
        ax.add_patch(circle)
        ax.plot(c[0], c[1], "ko", markersize=3)

    ax.set_aspect("equal")
    ax.set_title(f"m = {m}, Uncovered = {uncovered_area:.6f}")
    ax.grid(True)

    if save_dir is not None:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = save_name if save_name is not None else f"m_{m}.png"
        fig.savefig(output_dir / file_name, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)