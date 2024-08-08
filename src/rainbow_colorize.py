import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def grayscale_to_rainbow(image_path):
    # Load the grayscale image
    img = Image.open(image_path).convert('L')  # 'L' mode is for grayscale

    # Convert the grayscale image to a numpy array
    grayscale_array = np.array(img)

    # Normalize the grayscale values to the range [0, 1]
    normalized_array = grayscale_array / 255.0

    # Get the rainbow colormap from matplotlib
    cmap = plt.get_cmap('hsv')

    # Apply the colormap to the normalized grayscale values
    colorized_array = cmap(normalized_array)

    # Convert the colormap output to 8-bit RGB values
    colorized_array = (colorized_array[:, :, :3] * 255).astype(np.uint8)

    # Convert the numpy array back to an image
    colorized_image = Image.fromarray(colorized_array)

    # Save the colorized image with "_rainbow" appended to the original file name
    base, ext = os.path.splitext(image_path)
    new_image_path = f"{base}_rainbow{ext}"
    colorized_image.save(new_image_path)

    print(f"Saved rainbow-colorized image as: {new_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Convert a grayscale image to a rainbow colorized image.")
    parser.add_argument("image_path", type=str, help="Path to the grayscale image.")
    
    args = parser.parse_args()
    
    grayscale_to_rainbow(args.image_path)
