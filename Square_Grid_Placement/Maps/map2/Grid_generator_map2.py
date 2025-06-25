from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img_path = "M_0023.png"
image = Image.open(img_path).convert("L").resize((300,157)) 

image_array = np.array(image)

grid = np.where(image_array < 128, 1, -1)


