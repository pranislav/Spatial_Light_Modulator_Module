from skimage.restoration import unwrap_phase
import numpy.ma as ma
import numpy as np
from PIL import Image as im
import argparse
import constants as c
import time
import copy
import cv2
import wavefront_correction_lib as cl


def main(args):
    phase_mask = read_phase_mask(args.phase_mask_name, args.source_dir)
    small_phase_mask = shrink_phase_mask(phase_mask, args.subdomain_size)
    unwrapped_mask = unwrap_phase_picture(small_phase_mask, args.correspond_to_2pi)
    resample = im.BICUBIC if args.resample == "bicubic" else im.BILINEAR
    upscaled_unwrapped_mask = im.fromarray(unwrapped_mask).resize((c.slm_width, c.slm_height), resample=resample)
    upscaled_unwrapped_mask = np.array(upscaled_unwrapped_mask)
    time_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_smoothed_mask_unwrapped(copy.deepcopy(upscaled_unwrapped_mask), args.phase_mask_name[:-4], time_name, args.source_dir)
    save_smoothed_mask(upscaled_unwrapped_mask, args.phase_mask_name[:-4], time_name, args.correspond_to_2pi, args.source_dir)

def main_blur(args):
    phase_mask = read_phase_mask(args.phase_mask_name, args.source_dir)
    small_phase_mask = shrink_phase_mask(phase_mask, args.subdomain_size)
    unwrapped_mask = unwrap_phase_picture(small_phase_mask, args.correspond_to_2pi)
    original_size_unwrapped_mask = cl.expand_phase_mask(unwrapped_mask, args.subdomain_size)
    blurred_unwrapped_mask = circular_box_blur(original_size_unwrapped_mask, args.subdomain_size // 2)
    time_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_smoothed_mask_unwrapped(copy.deepcopy(blurred_unwrapped_mask), args.phase_mask_name, time_name)
    save_smoothed_mask(blurred_unwrapped_mask, args.phase_mask_name, time_name, args.correspond_to_2pi)

def original_frame(phase_mask, unwrapped_mask):
    mask = circular_hole_inclusive(phase_mask.shape)
    unwrapped_mask_without_frame = unwrapped_mask * (np.ones(phase_mask.shape) - mask)
    frame = phase_mask * mask
    return unwrapped_mask_without_frame #+ frame

def save_smoothed_mask_unwrapped(smoothed_mask, name, time_name, source_dir):
    smoothed_mask = smoothed_mask / max(smoothed_mask.flatten()) * 255
    smoothed_mask_unwrapped = im.fromarray(smoothed_mask)
    smoothed_mask_unwrapped.convert("L").save(f"{source_dir}/{name}_smoothed_unwrapped_{time_name}.png")

def save_smoothed_mask(smoothed_mask, name, time_name, correspond_to_2pi, source_dir):
    smoothed_mask = smoothed_mask % correspond_to_2pi
    smoothed_mask = im.fromarray(smoothed_mask)
    print(f"{name}_smoothed_{time_name}.png")
    smoothed_mask.convert("L").save(f"{source_dir}/{name}_smoothed_{time_name}.png")


def unwrap_phase_picture(phase_mask, correspond_to_2pi):
    phase_mask = transform_to_phase_values(phase_mask, correspond_to_2pi)
    # phase_mask = mask_mask(phase_mask)
    unwrapped_phase_mask = unwrap_phase(phase_mask)
    unwrapped_phase_mask = transform_to_color_values(unwrapped_phase_mask, correspond_to_2pi)
    return unwrapped_phase_mask

def preview_img(arr):
    im.fromarray((arr / arr.max()) * 255).show()


def read_phase_mask(phase_mask_name, source_dir):
    phase_mask = im.open(f"{source_dir}/{phase_mask_name}")
    phase_mask_arr = np.array(phase_mask).astype(float)
    return phase_mask_arr

def shrink_phase_mask(phase_mask: np.array, subdomain_size: int):
    h, w = phase_mask.shape
    H, W = h // subdomain_size, w // subdomain_size
    small_phase_mask = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            small_phase_mask[i, j] = phase_mask[i * subdomain_size, j * subdomain_size]
    return small_phase_mask

            

def transform_to_phase_values(phase_mask, correspond_to_2pi):
    return phase_mask / correspond_to_2pi * 2 * np.pi - np.pi

def mask_mask(phase_mask):
    mask = circular_hole_inclusive(phase_mask.shape)
    return ma.masked_array(phase_mask, mask)


def circular_hole_inclusive(shape):
    h, w = shape
    R = h // 2 + 1
    i0, j0 = h // 2, w // 2
    mask = np.array([[0 if (i - i0)**2 + (j - j0)**2 < R**2 else 1 for j in range(w)] for i in range(h)])
    return mask

def transform_to_color_values(phase_mask, correspond_to_2pi):
    offset = determine_offset(phase_mask.min())
    positive_phase_mask = (phase_mask + offset) * correspond_to_2pi / (2 * np.pi)
    return positive_phase_mask

def determine_offset(min_val):
    offset = 0
    while offset + min_val < 0:
        offset += 2 * np.pi
    return offset

def create_circular_kernel(radius):
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    kernel /= kernel.sum()
    return kernel

def circular_box_blur(image, radius):
    kernel = create_circular_kernel(radius)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def show_negative(arr):
    binary = np.array([[0 if a > 0 else 255 for a in arr[i]]for i in range(arr.shape[0])])
    im.fromarray(binary).convert("L").show()


if __name__ == "__main__":
    source_dir = "holograms/wavefront_correction_phase_masks/"
    parser = argparse.ArgumentParser()
    parser.add_argument("phase_mask_name", type=str, help=f'name of a phase mask in directory {source_dir}')
    parser.add_argument("-ct2pi", "--correspond_to_2pi", metavar="INT", type=int, required=True, help="value of pixel corresponding to 2pi phase shift")
    parser.add_argument("-ss", "--subdomain_size", metavar="INT", type=int, default=32, help="subdomain size used to create the phase mask")
    parser.add_argument("-resample", type=str, choices=["bilinear", "bicubic"], default="bilinear", help="smoothing method used to upscale the unwrapped phase mask")
    args = parser.parse_args()
    args.source_dir = source_dir
    main(args)
