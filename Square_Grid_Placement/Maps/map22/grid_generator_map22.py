from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img_path = "M_0022.png"
image = Image.open(img_path).convert("L").resize((300, 285))  # Convert to grayscale and resize

image_array = np.array(image)

grid = np.where(image_array < 128, 1, -1)


unknown_mask = grid == -1

with open("obstacle_map22.txt", "w") as f:
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
          if grid[i, j] == -1:
            f.write(str('0' + " "))
          else:
            f.write(str("1" + " " ))
        f.write("\n")