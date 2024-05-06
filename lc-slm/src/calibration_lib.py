import constants as c
import tkinter as tk
from PIL import Image as im, ImageTk
from screeninfo import get_monitors
import numpy as np
import os
import sys
import explore_calibration as e



black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))

def make_specification(args):
    return f"size_{args.subdomain_size}_precision_{args.precision}_x{args.angle[0]}_y{args.angle[1]}_ref{args.reference_coordinates}_avg{args.num_to_avg}_{args.calibration_name}"


# ----------- best phase() -------------- #

def naive(phase_list):
    opt_index = phase_list[1].index(max(phase_list[1]))
    return phase_list[0][opt_index]

def fit_phase_shift(phase_list):
    try:
        _, _, _, phase_shift = e.fit_intensity(phase_list)
    except:
        print("fit unsuccessful")
        phase_shift = 0
    return int(phase_shift)

def trick(phase_list):
    pass


# ---------- sample holograms ----------- #
# TODO: nice but could be faster. compare if consistent with same-name function in explore and replace

def make_sample_holograms(angle, precision):
    angle = angle.split("_")
    sample_holograms = []
    for i in range(precision):
        sample_holograms.append(decline(angle, i * 256 // precision))
    return sample_holograms

def decline(angle, offset):
    x_angle, y_angle = angle
    hologram = np.zeros((c.slm_height, c.slm_width))
    const = 256 * c.px_distance / c.wavelength # 256 gives more accurate result
    for i in range(c.slm_height):
        for j in range(c.slm_width):
            new_phase = const * (np.sin(float(y_angle) * c.u) * i + np.sin(float(x_angle) * c.u) * j)
            hologram[i, j] = int((new_phase + offset) % 256)
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

def read_reference_coordinates(reference_coordinates_str):
    x, y = reference_coordinates_str.split('_')
    return int(x), int(y)


def get_number_of_subdomains(subdomain_size):
    if c.slm_height % subdomain_size != 0 or c.slm_width % subdomain_size != 0:
        print(f"some of the SLM pixels won't be covered, you better choose number which divides {c.slm_height} and {c.slm_width}")
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
    return (int(a), int(b))

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

def create_phase_mask(phase_mask, subdomain_size, name, dest_dir):
    '''creates and saves phase mask image based on phase mask array
    '''
    h, w = phase_mask.shape
    ss = subdomain_size
    phase_mask_img = im.new("L", (w * ss, h * ss))
    for i in range(h):
        for k in range(ss):
            for j in range(w):
                for p in range(ss):
                    phase_mask_img.putpixel((ss * j + k, ss * i + p), int(phase_mask[i, j]))
    if not os.path.exists(dest_dir): os.makedirs(dest_dir)
    phase_mask_img.save(f"{dest_dir}/{name}.png")


def mask_hologram(path_to_hologram, path_to_mask):
    hologram_im = im.open(path_to_hologram)
    mask_im = im.open(path_to_mask)
    hologram_arr = np.array(hologram_im)
    mask_arr = np.array(mask_im)
    masked_hologram = hologram_arr + mask_arr
    return im.fromarray(masked_hologram)


# ---------- getters ------------ # - for old implementation

# def get_path_to_reference_hologram(path):
#     file_list = [_ for _ in filter(could_be_file, os.listdir(path))]
#     if len(file_list) > 1:
#         print("found multiple files while looking for reference hologram")
#         sys.exit(1)
#     return f"{path}/{file_list[0]}"

# def could_be_file(basename):
#     _, ext = os.path.splitext(basename)
#     return bool(ext)

# def get_reference_position(path_to_reference_hologram):
#     name = os.path.basename(path_to_reference_hologram)
#     wout_ext, _ = os.path.splitext(name)
#     return eval(wout_ext)
    

# def get_subdomain_size(path):
#     '''finds out subdomain size based on system of saving holograms'''
#     dir_count = 0
#     for item in os.listdir(path):
#         item_path = os.path.join(path, item)
#         if os.path.isdir(item_path):
#             dir_count += 1
#     return c.slm_height // dir_count


# def get_precision(path):
#     '''finds out precision based on system of number of holograms for one subdomain'''
#     file_count = 0
#     for item in os.listdir(path):
#         item_path = os.path.join(path, item)
#         if os.path.isfile(item_path):
#             file_count += 1
#     return file_count


# ------------ intensity getters a.k.a. metrics -------------- #

def get_intensity_naive(img_arr: np.array):
    '''returns maximal value on the img'''
    return max(img_arr.flatten())

def get_intensity_coordinates(img_arr: np.array, coordinates: tuple):
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
    display_image_on_external_screen_img(window, hologram)
    set_exposure(cam, intensity_range, num_to_avg)

def set_exposure_wrt_reference_img_path(cam, window, intensity_range, hologram_path, num_to_avg):
    display_image_on_external_screen(window, hologram_path)
    set_exposure(cam, intensity_range, num_to_avg)


def get_highest_intensity_coordinates(cam, window, hologram_path, num_to_avg):
    display_image_on_external_screen(window, hologram_path)
    img = average_frames(cam, max(8, num_to_avg))
    return find_highest_value_coordinates(img)

def get_highest_intensity_coordinates_img(cam, window, hologram, num_to_avg):
    display_image_on_external_screen_img(window, hologram)
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


def display_image_on_external_screen(window, image_path):
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

    # Load the image
    image = im.open(image_path)

    # Create a Tkinter PhotoImage object
    photo = ImageTk.PhotoImage(image)

    # Create a label to display the image
    label = tk.Label(window, image=photo)
    label.pack()
    label.photo = photo # makes the image to persist through a while cycle

    # Update the window to display the new image
    window.update()


def display_image_on_external_screen_img(window, image):
    """
    Display an image on an external screen without borders or decorations.
    DIffers from the other function by datatype of the second argument

    Parameters:
    - image (Image object): The path to the image file to be displayed.

    Returns:
    None
    """

    # Destroy the existing window if it exists
    for widget in window.winfo_children():
            widget.destroy()

    # Create a Tkinter PhotoImage object
    photo = ImageTk.PhotoImage(image)

    # Create a label to display the image
    label = tk.Label(window, image=photo)
    label.pack()
    label.photo = photo # makes the image to persist through a while cycle

    # Update the window to display the new image
    window.update()
