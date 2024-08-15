import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import help_messages_wfc


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Add a colormap to a grayscale image. Optionally, convert the image to rainbow colors.",
    )
    parser.add_argument("image_path", type=str, help="Path to the grayscale image.")
    parser.add_argument(
        "-p",
        "--palette",
        type=str,
        default="greyscale",
        help="Color palette to apply ('greyscale' or 'hue'). Optional, default is 'greyscale'.",
    )
    parser.add_argument(
        "-ct2pi", type=float, required=True, help=help_messages_wfc.ct2pi
    )
    return parser.parse_args()


def read_image(image_path):
    """Read and convert the image to grayscale."""
    grayscale_image = Image.open(image_path).convert("L")
    return np.array(grayscale_image)


def convert_to_phase(image_array, ct2pi):
    """Convert grayscale image values to phase values linearly."""
    return image_array / ct2pi * 2 * np.pi


def save_image_with_colorbar(img_array, original_image_path, palette):
    """Save the colored image with a colorbar to the directory of its origin."""
    base, ext = os.path.splitext(original_image_path)
    output_path = f"{base}_{palette}_colorbar{ext}"

    # Get original image dimensions
    # img_width, img_height = img.size

    # Plot with aspect ratio preservation and no ticks/values
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        img_array, cmap=palette, aspect="equal", vmin=0, vmax=2 * np.pi
    )  # extent=[0, img_width, 0, img_height])

    # Remove ticks and values from the image
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # cbar = fig.colorbar(im, cax=cax)
    cbar = fig.colorbar(
        im, cax=cax, ticks=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    )
    cbar.set_label("Fázový posun [rad]")
    cbar.ax.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    plt.savefig(
        output_path, bbox_inches="tight", dpi=600
    )  # Save the plot with tight bounding box
    plt.close()

    print(f"Image saved to {output_path}")


def main(args):
    # Read, normalize, apply colormap, and save the image
    image_array = read_image(args.image_path)
    phase_array = convert_to_phase(image_array, args.ct2pi)
    save_image_with_colorbar(phase_array, args.image_path, args.palette)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
