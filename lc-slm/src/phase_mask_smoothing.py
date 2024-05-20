from skimage.restoration import unwrap_phase
import numpy.ma as ma
import numpy as np
from PIL import Image as im
import argparse
import os
import constants as c
from numpy.polynomial.polynomial import polyfit, polyval
import time
import copy
from polyfit2d import polyfit2d
import cv2


# def main(phase_mask_name, correspond_to_2pi, subdomain_size):
#     phase_mask = read_phase_mask(phase_mask_name)
#     small_phase_mask = subdomain_to_pixel(phase_mask, subdomain_size)
#     unwrapped_mask = unwrap_phase_picture(small_phase_mask, correspond_to_2pi, subdomain_size)
#     unwrapped_mask_original_frame = original_frame(small_phase_mask, unwrapped_mask.data)
#     original_size_unwrapped_mask = pixel_to_subdomain(unwrapped_mask_original_frame, subdomain_size)
#     blurred_unwrapped_mask = circular_box_blur(original_size_unwrapped_mask, subdomain_size // 2)
#     time_name = time.strftime("%Y-%m-%d_%H-%M-%S")
#     save_blurred_mask_unwrapped(copy.deepcopy(blurred_unwrapped_mask), phase_mask_name, time_name)
#     save_blurred_mask(blurred_unwrapped_mask, phase_mask_name, time_name, correspond_to_2pi)
def main(args):
    phase_mask = read_phase_mask(args.phase_mask_name)
    small_phase_mask = subdomain_to_pixel(phase_mask, args.subdomain_size)
    unwrapped_mask = unwrap_phase_picture(small_phase_mask, args.correspond_to_2pi)
    # unwrapped_mask_original_frame = original_frame(small_phase_mask, unwrapped_mask.data)
    original_size_unwrapped_mask = pixel_to_subdomain(unwrapped_mask, args.subdomain_size)
    # original_size_unwrapped_mask_img = im.fromarray(original_size_unwrapped_mask, mode='L')
    blurred_unwrapped_mask = circular_box_blur(original_size_unwrapped_mask, args.subdomain_size // 2)
    time_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_blurred_mask_unwrapped(copy.deepcopy(blurred_unwrapped_mask), args.phase_mask_name, time_name)
    save_blurred_mask(blurred_unwrapped_mask, args.phase_mask_name, time_name, args.correspond_to_2pi)


def original_frame(phase_mask, unwrapped_mask):
    mask = circular_hole(phase_mask.shape)
    unwrapped_mask_without_frame = unwrapped_mask * (np.ones(phase_mask.shape) - mask)
    frame = phase_mask * mask
    return unwrapped_mask_without_frame #+ frame

def save_blurred_mask_unwrapped(blurred_mask, name, time_name):
    blurred_mask = blurred_mask / max(blurred_mask.flatten()) * 255
    blurred_mask_unwrapped = im.fromarray(blurred_mask)
    blurred_mask_unwrapped.convert("L").save(f"{name}_blurred_unwrapped_{time_name}.png")

def save_blurred_mask(blurred_mask, name, time_name, correspond_to_2pi):
    blurred_mask = blurred_mask % correspond_to_2pi
    blurred_mask = im.fromarray(blurred_mask)
    blurred_mask.convert("L").save(f"{name}_blurred_{time_name}.png")


# def fit_and_eval(unwrapped_mask, polynom_degree):
#     h, w = unwrapped_mask.shape
#     x = np.linspace(0, w, w)
#     y = np.linspace(0, h, h)
#     p = polyfit2d(x, y, unwrapped_mask, polynom_degree)
#     fitted_mask = polyval(x, p)
#     return fitted_mask.reshape(h, w)

def unwrap_phase_picture(phase_mask, correspond_to_2pi):
    phase_mask = transform_to_phase_values(phase_mask, correspond_to_2pi)
    phase_mask = mask_mask(phase_mask)
    unwrapped_phase_mask = unwrap_phase(phase_mask)
    unwrapped_phase_mask = transform_to_color_values(unwrapped_phase_mask, correspond_to_2pi)
    # preview_img(unwrapped_phase_mask)
    return unwrapped_phase_mask

def preview_img(arr):
    im.fromarray((arr / arr.max()) * 255).show()


def read_phase_mask(phase_mask_path):
    phase_mask = im.open(phase_mask_path)
    phase_mask_arr = np.array(phase_mask).astype(float)
    return phase_mask_arr

def subdomain_to_pixel(phase_mask, subdomain_size):
    h, w = phase_mask.shape
    H, W = h // subdomain_size, w // subdomain_size
    small_phase_mask = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            small_phase_mask[i, j] = phase_mask[i * subdomain_size, j * subdomain_size]
    return small_phase_mask

def pixel_to_subdomain(phase_mask, subdomain_size):
    h, w = phase_mask.shape
    ss = subdomain_size
    big_phase_mask = np.zeros((h * ss, w * ss))
    for i in range(h):
        for k in range(ss):
            for j in range(w):
                for p in range(ss):
                    try:
                        big_phase_mask[ss * i + k, ss * j + p] = int(phase_mask[i, j])
                    except:
                        big_phase_mask[ss * i + k, ss * j + p] = 0
    return big_phase_mask
            

def transform_to_phase_values(phase_mask, correspond_to_2pi):
    return phase_mask / correspond_to_2pi * 2 * np.pi - np.pi

def mask_mask(phase_mask):
    mask = circular_hole(phase_mask.shape)
    return ma.masked_array(phase_mask, mask)

def circular_hole(shape):
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
    # im.fromarray(image).convert("L").show()
    blurred = cv2.filter2D(image, -1, kernel)
    show_negative(blurred)
    # im.fromarray(blurred).convert("L").show()
    return blurred

def show_negative(arr):
    binary = np.array([[0 if a > 0 else 255 for a in arr[i]]for i in range(arr.shape[0])])
    im.fromarray(binary).convert("L").show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase_mask_name", type=str, help='full path to the phase mask')
    parser.add_argument("-ct2pi", "--correspond_to_2pi", type=int, default=256, help="value of pixel corresponding to 2pi phase shift")
    parser.add_argument("-ss", "--subdomain_size", type=int, default=32, help="subdomain size used to create the phase mask")
    args = parser.parse_args()
    main(args)
    # main("lc-slm/holograms/fit_maps/phase_shift_size_32_precision_8_x1_y__ref16_12_avg1_try_lab_may10.png", 256, 32)