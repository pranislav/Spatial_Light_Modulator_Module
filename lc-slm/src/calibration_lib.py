import constants as c
import tkinter as tk
from PIL import Image as im, ImageTk
from screeninfo import get_monitors
import numpy as np
import os
import sys
import explore_calibration as e
import phase_mask_smoothing as pms
import time


def produce_phase_mask(phase_mask, args):
    specification = "phase_mask_" + make_specification(args)
    dest_dir = "lc-slm/holograms/calibration_phase_masks"
    big_phase_mask = expand_phase_mask(phase_mask, args.subdomain_size)
    save_phase_mask(big_phase_mask, dest_dir, specification)
    phase_mask_unwrapped = pms.unwrap_phase_picture(phase_mask, args.correspond_to2pi)
    if args.smooth_phase_mask:
        # big_phase_mask = pms.circular_box_blur(phase_mask_unwrapped, args.subdomain_size // 2)
        big_phase_mask = im.fromarray(phase_mask_unwrapped).resize((c.slm_width, c.slm_height), im.BICUBIC)
        save_phase_mask(np.array(big_phase_mask) % args.correspond_to2pi, dest_dir, "smoothed_"+specification)

def make_specification(args):
    return f"{args.calibration_name}_ss{args.subdomain_size}_ct2pi_{args.correspond_to2pi}_precision_{args.precision}_x{args.angle[0]}_y{args.angle[1]}_ref_{args.reference_coordinates}_intensity_coords_{args.intensity_coordinates}"


# ----------- best phase() -------------- #

def naive(phase_list):
    opt_index = phase_list[1].index(max(phase_list[1]))
    return phase_list[0][opt_index]


def trick(phase_intensity_list, ct2pi):
    imaginary_part = trick_function(phase_intensity_list, np.sin, ct2pi)
    real_part = trick_function(phase_intensity_list, np.cos, ct2pi)
    return np.angle(real_part + 1j * imaginary_part) * ct2pi / (2 * np.pi)

def trick_function(phase_intensity_list, fun, ct2pi):
    phase_list = phase_intensity_list[0]
    intensity_list = phase_intensity_list[1]
    return sum([intensity_list[i] * fun(2 * np.pi / ct2pi * phase_list[i]) for i in range(len(phase_list))])



# ---------- sample holograms ----------- #

def make_sample_holograms(angle, precision, ct2pi):
    sample = []
    sample.append(decline(angle, ct2pi))
    for i in range(1, precision):
        offset = i * 256 // precision
        sample.append((sample[0] + offset) % ct2pi)
    return sample

def decline(angle, ct2pi):
    x_angle, y_angle = angle
    hologram = np.zeros((c.slm_height, c.slm_width))
    const = ct2pi * c.px_distance / c.wavelength
    for i in range(c.slm_height):
        for j in range(c.slm_width):
            new_phase = const * (np.sin(float(y_angle) * c.u) * i + np.sin(float(x_angle) * c.u) * j)
            hologram[i, j] = int(new_phase % ct2pi)
    return hologram


# ----------- subdomain manipulation ------------ #

def add_subdomain(hologram: im, sample: np.array, img_coordinates, subdomain_size):
    i0, j0 = img_coordinates
    for i in range(subdomain_size):
        for j in range(subdomain_size):
            hologram.putpixel((i0 + i, j0 + j), int(sample[j0 + j, i0 + i]))
    return hologram

def clear_subdomain(hologram: im, coordinates, subdomain_size):
    # equivalent to add_subdomain(hologram, np.zeros(), coordinates, subdomain_size)
    # but it would be more expensive on time
    i0, j0 = coordinates
    for i in range(subdomain_size):
        for j in range(subdomain_size):
            hologram.putpixel((j0 + j, i0 + i), 0)
    return hologram


# ---------- new getters ----------- #

def extract_reference_coordinates(reference_hologram_coordinates_ratio, subdomain_size, subdomains_number):
    y_numerator, y_denominator, x_numerator, x_denominator = reference_hologram_coordinates_ratio.split("_")
    H, W = subdomains_number
    y_coord = subdomain_size * (int(y_numerator) * H // int(y_denominator))
    x_coord = subdomain_size * (int(x_numerator) * W // int(x_denominator))
    return (y_coord, x_coord)

def read_reference_coordinates(reference_coordinates_str, shape):
    if reference_coordinates_str == "center":
        H, W = shape
        return W // 2, H // 2
    x, y = reference_coordinates_str.split('_')
    return int(x), int(y)


def get_number_of_subdomains(subdomain_size):
    if c.slm_height % subdomain_size != 0 or c.slm_width % subdomain_size != 0:
        print(f"some of the SLM pixels won't be covered, you better choose subdomain size which divides {c.slm_height} and {c.slm_width}")
    return c.slm_height//subdomain_size, c.slm_width//subdomain_size



# --------- a little image processing --------- # - for integral metrics (which probably does not work at all)

def detect_bright_area(picture: np.array):
    coord = mean_position_of_white_pixel(picture)
    length = deviation_of_bright_pixels(picture, coord)
    return (coord, length)

def mean_position_of_white_pixel(picture):
    h, w = picture.shape
    sum = np.array([0, 0])
    norm = 0
    for y in range(h):
        for x in range(w):
            sum += np.array([y, x]) * picture[y, x]
            norm += picture[y, x]
    a, b = sum / norm
    return (int(round(a)), int(round(b)))

def mean_position_of_overflow_pixel(picture):
    h, w = picture.shape
    sum = np.array([0, 0])
    norm = 0
    for y in range(h):
        for x in range(w):
            if picture[y, x] == 255:
                sum += np.array([y, x])
                norm += 1
    a, b = sum / norm
    return (int(round(a)), int(round(b)))

def deviation_of_bright_pixels(picture, mean):
    h, w = picture.shape
    deviation = 0
    sum = 0
    for y in range(h):
        for x in range(w):
            dist = np.array([y, x]) - mean
            deviation += np.dot(dist, dist) * picture[y, x]
            sum += picture[y, x]
    return int(np.sqrt(deviation / sum))


# ------------ The Phase Mask ---------- #

def expand_phase_mask(phase_mask, subdomain_size):
    '''takes a phase mask where one pixel represents a subdomain
    and returns a phase mask where one pixel represents a pixel in SLM
    '''
    h, w = phase_mask.shape
    ss = subdomain_size
    big_phase_mask = np.zeros((h * ss, w * ss))
    for i in range(h):
        for k in range(ss):
            for j in range(w):
                for p in range(ss):
                    big_phase_mask[ss * i + k, ss * j + p] =  int(phase_mask[i, j])
    return big_phase_mask


def save_phase_mask(phase_mask, dest_dir, name):                
    if not os.path.exists(dest_dir): os.makedirs(dest_dir)
    phase_mask_img = im.fromarray(phase_mask)
    phase_mask_img.convert('L').save(f"{dest_dir}/{name}.png")


def mask_hologram(path_to_hologram, path_to_mask):
    hologram_im = im.open(path_to_hologram)
    mask_im = im.open(path_to_mask)
    hologram_arr = np.array(hologram_im)
    mask_arr = np.array(mask_im)
    masked_hologram = hologram_arr + mask_arr
    return im.fromarray(masked_hologram)


# ------------ intensity getters a.k.a. metrics -------------- #

def get_intensity_naive(img_arr: np.array):
    '''returns maximal value on the img'''
    return max(img_arr.flatten())

def get_intensity_on_coordinates(img_arr: np.array, coordinates: tuple):
    return img_arr[coordinates]

def get_intensity_integral(frame, square):
    (x, y), length = square
    intensity_sum = 0
    overflow = 0
    for i in range(2 * length):
        for j in range(2 * length):
            intensity = frame[x - length + i, y - length + j]
            if intensity == 255: overflow += 1
            intensity_sum += intensity
    if overflow: print(f"max intensity reached on {overflow} pixels")
    return intensity_sum



# ---------- functions for work with camera ----------- #

def set_exposure(cam, intensity_range, num_to_avg):
    '''set exposure time such that maximal signal value is
    in interval (max_val_lower, max_val_upper)
    '''

    num_to_avg = max(8, num_to_avg)
    max_val_lower, max_val_upper = intensity_range
    step = 10e-3
    expo = 0
    max_val = max_val_lower - 1
    while max_val < max_val_lower or max_val > max_val_upper:
        if max_val < max_val_lower:
            expo += step
        if max_val > max_val_upper:
            step /= 2
            expo -= step
        if expo < 0.000267:
            print("exposure time reached minimal possible value!")
            cam.set_exposure(0.000267)
            return
        cam.set_exposure(expo)
        print(expo)
        avgd_frame = average_frames(cam, num_to_avg)
        # im.fromarray(frame).show()
        max_val = max(avgd_frame.flatten())
        # print(max_val)
    

def average_frames(cam, num_to_avg):
    frame = cam.snap()
    frame = np.array(frame).astype(float)
    # im.fromarray(frame).show()
    for _ in range(1, num_to_avg):
        new_frame = np.array(cam.snap()).astype(float)
        frame += new_frame
    frame /= num_to_avg
    return frame

def set_exposure_wrt_reference_img(cam, window, intensity_range, hologram, num_to_avg):
    display_image_on_external_screen(window, hologram)
    set_exposure(cam, intensity_range, num_to_avg)

# def set_exposure_wrt_reference_img_path(cam, window, intensity_range, hologram_path, num_to_avg):
#     display_image_on_external_screen(window, hologram_path)
#     set_exposure(cam, intensity_range, num_to_avg)

def get_intensity_coords(cam, window, hologram, args):
    display_image_on_external_screen(window, hologram)
    img = cam.snap()
    if args.intensity_coordinates is None:
        # get_highest_intensity_coordinates_img(cam, window, reference_hologram, args.num_to_avg) 
        coords = mean_position_of_overflow_pixel(img)
        print(f"intensity coordinates: {coords}")
    else:
        x, y = args.intensity_coordinates.split("_")
        coords = (int(x), int(y))
    show_coords_on_img(cam.snap(), coords)
    return coords

def show_coords_on_img(array_img, coords):
    img_img = im.fromarray(array_img, "L").convert("RGB")
    marked_img = e.add_cross(img_img, coords)
    marked_img.show()


def get_highest_intensity_coordinates(cam, window, hologram_path, num_to_avg):
    display_image_on_external_screen(window, hologram_path)
    img = average_frames(cam, max(8, num_to_avg))
    return find_highest_value_coordinates(img)

def get_highest_intensity_coordinates_img(cam, window, hologram, num_to_avg):
    print("getting highest intensity coordinates")
    display_image_on_external_screen(window, hologram)
    img = average_frames(cam, max(8, num_to_avg))
    return find_highest_value_coordinates(img)

def find_highest_value_coordinates(arr):
    max_value = arr[0][0]
    max_i = 0
    max_j = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] > max_value:
                max_value = arr[i][j]
                max_i = i
                max_j = j
    return max_i, max_j


# ----------- displaying on external screen ----------- #
# ----------- credits: chatGPT


def create_tk_window():
    # Determine the external screen dimensions
    for monitor in get_monitors():
        if monitor.x != 0 or monitor.y != 0:
            SCREEN_WIDTH = monitor.width
            SCREEN_HEIGHT = monitor.height
            break

    # Create a Tkinter window
    window = tk.Tk()
    window.overrideredirect(True)
    window.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}+{monitor.x}+{monitor.y}")
    # window.geometry(f"{c.slm_width}x{c.slm_height}+{monitor.x}+{monitor.y}")


    return window


def display_image_on_external_screen(window, image):
    """
    Display an image on an external screen without borders or decorations.

    Parameters:
    - image_path (str): The path to the image file to be displayed.

    Returns:
    None
    """

    # Destroy the existing window if it exists
    for widget in window.winfo_children():
            widget.destroy()

    if not isinstance(image, im.Image):
        image = im.open(image)

    # Create a Tkinter PhotoImage object
    photo = ImageTk.PhotoImage(image)

    # Create a label to display the image
    label = tk.Label(window, image=photo)
    label.pack()
    label.photo = photo # makes the image to persist through a while cycle

    # Update the window to display the new image
    window.update()

    time.sleep(0.017) # cca 1/60 s which is the refresh rate of the SLM


# def display_image_on_external_screen_img(window, image):
#     """
#     Display an image on an external screen without borders or decorations.
#     DIffers from the other function by datatype of the second argument

#     Parameters:
#     - image (Image object): The path to the image file to be displayed.

#     Returns:
#     None
#     """

#     # Destroy the existing window if it exists
#     for widget in window.winfo_children():
#             widget.destroy()

#     # Create a Tkinter PhotoImage object
#     photo = ImageTk.PhotoImage(image)

#     # Create a label to display the image
#     label = tk.Label(window, image=photo)
#     label.pack()
#     label.photo = photo # makes the image to persist through a while cycle

#     # Update the window to display the new image
#     window.update()

#     time.sleep(0.017) # cca 1/60 s which is the refresh rate of the SLM
