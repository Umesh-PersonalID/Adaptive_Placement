from geometry.disk_reg import create_region
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
import numpy as np


def plot(region, show_vertices=False, show_centroid=True, show_bbox=True):
    fig, ax = plt.subplots(figsize=(7, 7))

    def draw_polygon(poly):
        # Exterior
        x, y = poly.exterior.xy
        ax.fill(x, y, facecolor="#ff914d", edgecolor="black", linewidth=2, alpha=0.6)

        # Holes
        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, facecolor="black", edgecolor="black", linewidth=1.5)

        # Vertices
        if show_vertices:
            ax.scatter(x, y, color="red", s=15, zorder=5)

    # Handle MultiPolygon safely
    if isinstance(region, MultiPolygon):
        for poly in region.geoms:
            draw_polygon(poly)
    else:
        draw_polygon(region)

    # Bounding Box
    if show_bbox:
        minx, miny, maxx, maxy = region.bounds
        ax.plot(
            [minx, maxx, maxx, minx, minx],
            [miny, miny, maxy, maxy, miny],
            linestyle="--",
            color="gray",
            linewidth=1
        )

    # Styling
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f9f9f9")

    plt.title("Region Visualization", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()
    
for i in range(0,6):
    region = create_region(i)
    plot(region)
