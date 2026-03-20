import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import time
import sys
import os
from matplotlib.patches import Circle

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'Newton-Shape_Placement'))
from geometry.regions import all_regions
from shapely.geometry import Point

REGION_INDEX = 0 
regions = all_regions()
region = regions[REGION_INDEX]

disk_radius = 2 

region_bounds = region.bounds
minx, miny, maxx, maxy = region_bounds


def is_point_in_region(x, y):
    """Check if a point (in region coordinates) is inside the region"""
    point = Point(x, y)
    return region.contains(point) or region.intersects(point.buffer(disk_radius * grid_resolution))



class HexagonalGridPlacement:
    def __init__(self, region_index=2, disk_radius=2):
        """
        Initialize hexagonal grid placement for a specific region
        Args:
            region_index: Index of the region (0-5 for regions 1-6)
            disk_radius: Radius of each disk/sensor
        """
        self.regions = all_regions()
        self.region = self.regions[region_index]
        self.region_index = region_index
        self.disk_radius = disk_radius
        self.grid_spacing = disk_radius * math.sqrt(3)  # Hexagonal grid spacing
        self.disk_positions = []
        self.valid_positions = []
        self.overlap_areas = []
        
        # Get region bounds
        self.minx, self.miny, self.maxx, self.maxy = self.region.bounds
         
        # Add padding
        padding = 0.5 * self.disk_radius
        self.minx -= padding*0
        self.miny += padding
        self.maxx += padding*2
        self.maxy += padding*2
    
    def generate_hexagonal_grid(self):
        grid_points = []

        x_spacing = math.sqrt(3) * self.disk_radius
        y_spacing = 1.5 * self.disk_radius

        y = self.miny
        row = 0

        while y <= self.maxy:
            if row % 2 == 0:
                x = self.minx
            else:
                x = self.minx + x_spacing / 2

            while x <= self.maxx:
                grid_points.append((x, y))
                x += x_spacing

            y += y_spacing
            row += 1

        return grid_points
    
    def remove_redundant_outside_disks(self):
        inside_disks = []
        outside_disks = []

        for x, y in self.valid_positions:
            if self.region.contains(Point(x, y)):
                inside_disks.append((x, y))
            else:
                outside_disks.append((x, y))

        from shapely.ops import unary_union

        # coverage from inside disks
        inside_circles = [Point(x, y).buffer(self.disk_radius) for x, y in inside_disks]
        inside_union = unary_union(inside_circles).intersection(self.region)

        kept_outside = []

        for x, y in outside_disks:

            disk = Point(x, y).buffer(self.disk_radius)
            disk_region = disk.intersection(self.region)

            # new coverage added
            new_area = disk_region.difference(inside_union).area

            if new_area > 1e-3:  # threshold
                kept_outside.append((x, y))
                inside_union = inside_union.union(disk_region)

        self.valid_positions = inside_disks + kept_outside
    
    def is_disk_valid(self, center_x, center_y):
        """
        Check if a disk at given center is valid:
        2. Disk should not be completely inside a hole
        """
        disk_center = Point(center_x, center_y)
        disk_circle = disk_center.buffer(self.disk_radius)
        

        # Check if disk intersects with the region (not completely outside)
        if not disk_circle.intersects(self.region):
            return False
        
        return True
    
    def calculate_coverage_and_overlap(self):
        """Calculate coverage percentage and overlap within valid disks"""
        if not self.valid_positions:
            return 0, 0
        
        # Create circles for all valid positions
        circles = []
        for x, y in self.valid_positions:
            circle = Point(x, y).buffer(self.disk_radius)
            circles.append(circle)
        
        # Calculate total coverage area within the region
        from shapely.ops import unary_union
        total_coverage = unary_union(circles)
        coverage_within_region = total_coverage.intersection(self.region)
        
        # Calculate region area
        region_area = self.region.area
        coverage_area = coverage_within_region.area
        
        # Calculate coverage percentage
        coverage_percentage = min((coverage_area / region_area) * 100, 100.0)
        
        # Calculate overlap
        total_disk_area = len(self.valid_positions) * math.pi * self.disk_radius ** 2
        overlap_area = total_disk_area - coverage_area
        overlap_percentage = (overlap_area / coverage_area) * 100 if coverage_area > 0 else 0
        
        return coverage_percentage, overlap_percentage
    
    def run_placement(self):
        """Run the complete hexagonal grid placement algorithm"""
        print(f"Running hexagonal grid placement for Region {self.region_index + 1}...")
        
        # Generate all possible grid points
        grid_points = self.generate_hexagonal_grid()
        print(f"Generated {len(grid_points)} potential grid points")
        
        # Filter valid positions
        for x, y in grid_points:
            if self.is_disk_valid(x, y):
                self.valid_positions.append((x, y))
        
        self.remove_redundant_outside_disks()
        
        print(f"Found {len(self.valid_positions)} valid disk positions")
        
        # Calculate metrics
        coverage_pct, overlap_pct = self.calculate_coverage_and_overlap()
        
        print(f"Coverage: {coverage_pct:.2f}%")
        print(f"Overlap: {overlap_pct:.2f}%")
        
        return self.valid_positions, coverage_pct, overlap_pct
    
    def save_positions_to_file(self, filename="position.txt"):
        """Save disk positions to text file"""
        with open(filename, 'w') as f:
            f.write(f"# Hexagonal Grid Placement for Region {self.region_index + 1}\n")
            f.write(f"# Disk radius: {self.disk_radius}\n")
            f.write(f"# Number of disks: {len(self.valid_positions)}\n")
            f.write(f"# Format: x,y\n")
            for x, y in self.valid_positions:
                f.write(f"{x:.6f},{y:.6f}\n")
        
        print(f"Positions saved to {filename}")
    
    def generate_visualization(self, filename="hexagonal_grid_placement.png"):
        """Generate and save visualization of the placement"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot the region
        if self.region.geom_type == 'Polygon':
            # Plot exterior
            exterior_coords = list(self.region.exterior.coords)
            exterior_x, exterior_y = zip(*exterior_coords)
            ax.fill(exterior_x, exterior_y, alpha=0.3, color='lightblue', 
                   edgecolor='blue', linewidth=2, label='Region')
            
            # Plot holes
            for interior in self.region.interiors:
                hole_coords = list(interior.coords)
                hole_x, hole_y = zip(*hole_coords)
                ax.fill(hole_x, hole_y, alpha=1.0, color='white', 
                       edgecolor='red', linewidth=1.5, label='Obstacle')
        
        elif self.region.geom_type == 'MultiPolygon':
            for geom in self.region.geoms:
                if geom.geom_type == 'Polygon':
                    exterior_coords = list(geom.exterior.coords)
                    exterior_x, exterior_y = zip(*exterior_coords)
                    ax.fill(exterior_x, exterior_y, alpha=0.3, color='lightblue', 
                           edgecolor='blue', linewidth=2)
                    
                    # Plot holes for this polygon
                    for interior in geom.interiors:
                        hole_coords = list(interior.coords)
                        hole_x, hole_y = zip(*hole_coords)
                        ax.fill(hole_x, hole_y, alpha=1.0, color='white', 
                               edgecolor='red', linewidth=1.5)
        
        # Plot valid disk positions
        for x, y in self.valid_positions:
            circle = plt.Circle((x, y), self.disk_radius, color='green', 
                              alpha=0.6, edgecolor='darkgreen', linewidth=1)
            ax.add_patch(circle)
            ax.plot(x, y, 'ko', markersize=2)
        
        # Set axis properties
        ax.set_xlim(self.minx - self.disk_radius, self.maxx + self.disk_radius)
        ax.set_ylim(self.miny - self.disk_radius, self.maxy + self.disk_radius)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X coordinates')
        ax.set_ylabel('Y coordinates')
        
        # Calculate and display metrics
        coverage_pct, overlap_pct = self.calculate_coverage_and_overlap()
        
        ax.set_title(f'Hexagonal Grid Placement - Region {self.region_index + 1}\n'
                    f'Disks: {len(self.valid_positions)}, '
                    f'Coverage: {coverage_pct:.1f}%', 
                    fontsize=14, fontweight='bold')
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.3, edgecolor='blue', label='Region'),
                   plt.Circle((0,0), 0.1, facecolor='green', alpha=0.6, edgecolor='darkgreen', label='Disks')]
        ax.legend(handles=handles, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to {filename}")
        
        return coverage_pct, overlap_pct




def main():
    """Main function to run hexagonal grid placement for Region 4 """
    # Create placement instance for Region 1 (index 0)
    placement = HexagonalGridPlacement(region_index=3, disk_radius=2)
    
    # Run placement algorithm
    positions, coverage, overlap = placement.run_placement()
    
    # Save results
    placement.save_positions_to_file("position.txt")
    
    # Generate visualization
    coverage_pct, overlap_pct = placement.generate_visualization("Coverage_map26.png")
    
    # Save summary
    with open("results_summary.txt", 'w') as f:
        f.write(f"Hexagonal Grid Placement Results - Region 4\n")
        f.write(f"=========================================\n")
        f.write(f"Disk radius: {placement.disk_radius}\n")  
        f.write(f"Grid spacing: {placement.grid_spacing}\n")
        f.write(f"Number of disks placed: {len(positions)}\n")
        f.write(f"Coverage percentage: {coverage_pct:.2f}%\n")
        f.write(f"Overlap percentage: {overlap_pct:.2f}%\n")
        f.write(f"Region area: {placement.region.area:.2f}\n")
    
    print(f"\nResults Summary:")
    print(f"Disks placed: {len(positions)}")
    print(f"Coverage: {coverage_pct:.2f}%") 
    print(f"Overlap: {overlap_pct:.2f}%")

if __name__ == "__main__":
    main()