from PIL import Image
import numpy as np

# Load and resize the image to 300x300
img_path = "M_0023.png"
image = Image.open(img_path).convert("L").resize((157, 300))  # Convert to grayscale and resize

# Convert to numpy array
image_array = np.array(image)

# Thresholding: consider pixel value < 128 as black (1), else white (-1)
grid = np.where(image_array < 128, 1, -1)