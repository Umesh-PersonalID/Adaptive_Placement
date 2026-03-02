from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img_path = "M_0017.png"
image = Image.open(img_path).convert("L").resize((300, 285))  # Convert to grayscale and resize

image_array = np.array(image)

grid = np.where(image_array < 128, 1, -1)

unknown_mask = grid == -1