import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from PIL import Image

# Load the image
image_path = 'path_to_your_grayscale_image.png'
img = Image.open(image_path)

# Convert PIL image to NumPy array
img_array = np.array(img)

# Get original image dimensions
img_width, img_height = img.size

# Plot with aspect ratio preservation and no ticks/values
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(img_array, cmap='hsv', extent=[0, img_width, 0, img_height])  # Use 'gray' colormap

# Remove ticks and values from the image
ax.set_xticks([])
ax.set_yticks([])

# Add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Intensity')

plt.savefig('image_with_colorbar.png', bbox_inches='tight', dpi=600)  # Save the plot with tight bounding box
plt.close()

print('Plot saved successfully.')
