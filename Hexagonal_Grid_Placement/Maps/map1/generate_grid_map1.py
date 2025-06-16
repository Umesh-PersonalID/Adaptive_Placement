from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load and resize the image to 300x300
img_path = "M_0036.png"
image = Image.open(img_path).convert("L").resize((300, 300))  # Convert to grayscale and resize

# Convert to numpy array
image_array = np.array(image)

# Thresholding: consider pixel value < 128 as black (1), else white (-1)
grid = np.where(image_array < 128, 1, -1)

# Create an RGB image for visualization
img = np.zeros((grid.shape[0], grid.shape[1], 3))

unknown_mask = grid == -1
print(unknown_mask)
img[unknown_mask] = [0.7, 0.7, 0.7] 

occupied_mask = grid == 1
img[occupied_mask] = [0, 0, 0]       


plt.imshow(img)
plt.title("Grid Visualization")
plt.show()
