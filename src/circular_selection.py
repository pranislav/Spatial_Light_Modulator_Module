'''opens specified phase mask and applies
a circular selection to it, then saves the result.
motivation: examine the effect of a mask
without the edges which tends to be noisy.
'''


import argparse
from PIL import Image as im
import numpy as np



def circular_selection(args):
    dest_dir = "lc-slm/holograms/wavefront_correction_phase_masks"
    img = np.array(im.open(f"{dest_dir}/{args.file_name}"))
    mask = circular_selection_mask(img.shape, args.circle_size_fraction)
    img = img * mask
    img = im.fromarray(img)
    img.convert("L").save(f"{dest_dir}/{args.file_name[:-4]}_circular_selection.png")


def circular_selection_mask(shape, circle_size_fraction):
    h, w = shape
    R = h * circle_size_fraction // 2
    i0, j0 = h // 2, w // 2
    mask = np.array([[1 if (i - i0)**2 + (j - j0)**2 < R**2 else 0 for j in range(w)] for i in range(h)])
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    parser.add_argument("-cs", "--circle_size_fraction", type=float, default=0.9)
    
    args = parser.parse_args()
    circular_selection(args)
