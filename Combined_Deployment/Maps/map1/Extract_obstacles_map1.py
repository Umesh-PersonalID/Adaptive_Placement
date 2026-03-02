from PIL import Image
import numpy as np

def load_grid(image_path, threshold=128, size=(300, 300)):
    image = Image.open(image_path).convert("L").resize(size)
    image_array = np.array(image)
    grid = np.where(image_array < threshold, 1, -1)  # 1: obstacle, -1: free/unknown
    return grid

def extract_obstacle_coords(grid):
    obstacle_coords = []
    rows, cols = grid.shape
    for y in range(rows):
        for x in range(cols):
            if grid[y, x] == 1:
                obstacle_coords.append((x, y, 1))
    return obstacle_coords

def save_obstacles_to_file(coords, output_file):
    with open(output_file, 'w') as f:
        for x, y, t in coords:
            f.write(f"{x}, {y}, {t}\n")

if __name__ == "__main__":
    grid = load_grid("M_0036.png")
    obstacle_coords = extract_obstacle_coords(grid)
    save_obstacles_to_file(obstacle_coords, "obstacles_from_grid.txt")
    print(f"Extracted {len(obstacle_coords)} obstacle coordinates.")
