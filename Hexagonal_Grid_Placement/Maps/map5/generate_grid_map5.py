from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img_path = "M_0064.png"
image = Image.open(img_path).convert("L").resize((300, 300))  # Convert to grayscale and resize

image_array = np.array(image)

grid = np.where(image_array < 128, 1, -1)

img = np.zeros((grid.shape[0], grid.shape[1], 3))

unknown_mask = grid == -1
img[unknown_mask] = [0.7, 0.7, 0.7]  # Light gray

occupied_mask = grid == 1
img[occupied_mask] = [0, 0, 0]       # Black


# Show the image
plt.imshow(img)
plt.axis('off')  # Optional: turn off axis
plt.title("Grid Visualization")
plt.show()
