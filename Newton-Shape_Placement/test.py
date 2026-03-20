import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.collections import PatchCollection
import numpy as np
from geometry.regions import all_regions

def plot_region(region, ax, color='lightblue', edgecolor='black', alpha=0.7, linewidth=1):
    """Plot a Shapely region (polygon or multipolygon) on a matplotlib axis."""
    if region.geom_type == 'Polygon':
        # Plot exterior
        exterior_coords = list(region.exterior.coords)
        polygon = MPLPolygon(exterior_coords, closed=True, facecolor=color, 
                           edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
        ax.add_patch(polygon)
        
        # Plot holes (interiors)
        for interior in region.interiors:
            hole_coords = list(interior.coords)
            hole = MPLPolygon(hole_coords, closed=True, facecolor='white', 
                            edgecolor=edgecolor, alpha=1.0, linewidth=linewidth)
            ax.add_patch(hole)
            
    elif region.geom_type == 'MultiPolygon':
        # Handle MultiPolygon
        for geom in region.geoms:
            plot_region(geom, ax, color, edgecolor, alpha, linewidth)

def plot_all_regions():
    """Plot all regions in a grid layout."""
    regions = all_regions()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('All Regions from regions.py', fontsize=16, fontweight='bold')
    
    # Colors for different regions
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    
    # Flatten axes array for easy indexing
    axes_flat = axes.flatten()
    
    for i, (region, ax) in enumerate(zip(regions, axes_flat)):
        # Plot the region
        plot_region(region, ax, color=colors[i % len(colors)])
        
        # Set axis properties
        ax.set_title(f'Region {i+1}', fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits based on region bounds
        minx, miny, maxx, maxy = region.bounds
        padding = max((maxx - minx), (maxy - miny)) * 0.1
        ax.set_xlim(minx - padding, maxx + padding)
        ax.set_ylim(miny - padding, maxy + padding)
        
        # Add coordinate labels
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def plot_single_region(region_index=0):
    """Plot a single region with more detail."""
    regions = all_regions()
    
    if region_index >= len(regions):
        print(f"Invalid region index. Available regions: 0-{len(regions)-1}")
        return
    
    region = regions[region_index]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f'Region {region_index + 1} - Detailed View', fontsize=14, fontweight='bold')
    
    # Plot the region
    plot_region(region, ax, color='lightblue', edgecolor='navy', linewidth=2)
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with padding
    minx, miny, maxx, maxy = region.bounds
    padding = max((maxx - minx), (maxy - miny)) * 0.05
    ax.set_xlim(minx - padding, maxx + padding)
    ax.set_ylim(miny - padding, maxy + padding)
    
    # Add labels and info
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    
    # Add region info as text
    area = region.area
    perimeter = region.length
    ax.text(0.02, 0.98, f'Area: {area:.2f}\nPerimeter: {perimeter:.2f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_region_analysis():
    """Plot regions with analysis information."""
    regions = all_regions()
    
    # Create figure for analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Area comparison
    areas = [region.area for region in regions]
    region_names = [f'Region {i+1}' for i in range(len(regions))]
    
    bars = ax1.bar(region_names, areas, color=['lightblue', 'lightgreen', 'lightcoral', 
                                              'lightyellow', 'lightpink', 'lightgray'])
    ax1.set_title('Region Areas Comparison', fontweight='bold')
    ax1.set_ylabel('Area')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, area in zip(bars, areas):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{area:.1f}', ha='center', va='bottom')
    
    # Plot 2: Perimeter comparison
    perimeters = [region.length for region in regions]
    bars2 = ax2.bar(region_names, perimeters, color=['lightblue', 'lightgreen', 'lightcoral', 
                                                    'lightyellow', 'lightpink', 'lightgray'])
    ax2.set_title('Region Perimeters Comparison', fontweight='bold')
    ax2.set_ylabel('Perimeter')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, perimeter in zip(bars2, perimeters):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{perimeter:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Plot all regions
    print("Plotting all regions...")
    plot_all_regions()
    
    # Plot detailed view of first region
    print("\nPlotting detailed view of Region 1...")
    plot_single_region(0)
    
    # Plot analysis
    print("\nPlotting region analysis...")
    plot_region_analysis()
    
    # Interactive mode - ask user which region to plot in detail
    try:
        while True:
            user_choice = input("\nEnter region number (1-6) for detailed view, or 'q' to quit: ").strip()
            if user_choice.lower() == 'q':
                break
            try:
                region_num = int(user_choice) - 1
                if 0 <= region_num < 6:
                    plot_single_region(region_num)
                else:
                    print("Please enter a number between 1 and 6.")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
    except KeyboardInterrupt:
        print("\nExiting...")
