import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'Newton-Shape_Placement'))

from geometry.regions import all_regions
from shapely.geometry import Point
import math

# Global variables for tracking metrics
overlap_percentage = 0
total_area_covered = 0
total_region_area = 0
disk_positions = []

class SquareGridPlacement:
    def __init__(self, region_index=3, disk_radius=2):
        """
        Initialize square grid placement for a specific region
        Args:
            region_index: Index of the region (0-5 for regions 1-6)
            disk_radius: Radius of each disk/sensor
        """
        self.regions = all_regions()
        self.region = self.regions[region_index]
        self.region_index = region_index
        self.disk_radius = disk_radius
        self.grid_spacing = disk_radius * math.sqrt(2)  
        self.disk_positions = []
        self.valid_positions = []
        self.overlap_areas = []
        
        # Get region bounds
        self.minx, self.miny, self.maxx, self.maxy = self.region.bounds
        
        # Add padding
        padding = 0.299*self.disk_radius
        self.minx -= padding
        self.miny -= padding 
        self.maxx += padding*2
        self.maxy += padding*2
    
    def generate_square_grid(self):
        grid_points = []
        x_points = np.arange(self.minx + self.disk_radius, self.maxx, self.grid_spacing)
        y_points = np.arange(self.miny + self.disk_radius, self.maxy, self.grid_spacing)
        
        for x in x_points:
            for y in y_points:
                grid_points.append((x, y))
        
        return grid_points
    
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
        
        
        return coverage_percentage, 0

    def run_placement(self):
        """Run the complete square grid placement algorithm"""
        print(f"Running square grid placement for Region {self.region_index + 1}...")
        
        # Generate all possible grid points
        grid_points = self.generate_square_grid()
        print(f"Generated {len(grid_points)} potential grid points")
        
        # Filter valid positions
        for x, y in grid_points:
            if self.is_disk_valid(x, y):
                self.valid_positions.append((x, y))
        
        print(f"Found {len(self.valid_positions)} valid disk positions")
        
        # Calculate metrics
        coverage_pct, overlap_pct = self.calculate_coverage_and_overlap()
        
        print(f"Coverage: {coverage_pct:.2f}%")
        print(f"Overlap: {overlap_pct:.2f}%")
        
        return self.valid_positions, coverage_pct, overlap_pct
    
    def save_positions_to_file(self, filename="position.txt"):
        """Save disk positions to text file"""
        with open(filename, 'w') as f:
            f.write(f"# Square Grid Placement for Region {self.region_index + 1}\n")
            f.write(f"# Disk radius: {self.disk_radius}\n")
            f.write(f"# Number of disks: {len(self.valid_positions)}\n")
            f.write(f"# Format: x,y\n")
            for x, y in self.valid_positions:
                f.write(f"{x:.6f},{y:.6f}\n")
        
        print(f"Positions saved to {filename}")
    
    def generate_visualization(self, filename="square_grid_placement.png"):
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
        
        ax.set_title(f'Square Grid Placement - Region {self.region_index + 1}\n'
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
    """Main function to run square grid placement for Region 4"""
    # Create placement instance for Region 4 (index 4)
    placement = SquareGridPlacement(region_index=4, disk_radius=2)
    
    # Run placement algorithm
    positions, coverage, overlap = placement.run_placement()
    
    # Save results
    placement.save_positions_to_file("position.txt")
    
    # Generate visualization
    coverage_pct, overlap_pct = placement.generate_visualization("Coverage_map27.png")
    
    # Save summary
    with open("results_summary.txt", 'w') as f:
        f.write(f"Square Grid Placement Results - Region 5\n")
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