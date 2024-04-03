import constants as c
import tkinter as tk
from PIL import Image as im, ImageTk
from screeninfo import get_monitors
import numpy as np
import os
import sys


def create_phase_mask(phase_mask, subdomain_size, name):
    '''creates and saves phase mask image based on phase mask array
    '''
    h, w = phase_mask.shape
    ss = subdomain_size
    phase_mask_img = im.new("L", (w * ss, h * ss))
    for i in range(h):
        for k in range(ss):
            for j in range(w):
                for p in range(ss):
                    phase_mask_img.putpixel((ss * i + k, ss* j + p), int(phase_mask[i, j]))
    dest_dir = "lc-slm/holograms_for_calibration/calibration_phase_masks"
    if not os.path.exists(dest_dir): os.makedirs(dest_dir)
    phase_mask_img.save(f"{dest_dir}/{name}_phase_mask.png")


def mask_hologram(path_to_hologram, path_to_mask):
    hologram_im = im.open(path_to_hologram)
    mask_im = im.open(path_to_mask)
    hologram_arr = np.array(hologram_im)
    mask_arr = np.array(mask_im)
    masked_hologram = hologram_arr + mask_arr
    return im.fromarray(masked_hologram)


# ---------- getters ------------ #

def get_path_to_reference_hologram(path):
    file_list = [_ for _ in filter(could_be_file, os.listdir(path))]
    if len(file_list) > 1:
        print("found multiple files while looking for reference hologram")
        sys.exit(1)
    return f"{path}/{file_list[0]}"

def could_be_file(basename):
    _, ext = os.path.splitext(basename)
    return bool(ext)

def get_reference_position(path_to_reference_hologram):
    name = os.path.basename(path_to_reference_hologram)
    wout_ext, _ = os.path.splitext(name)
    return eval(wout_ext)
    

def get_subdomain_size(path):
    '''finds out subdomain size based on system of saving holograms'''
    dir_count = 0
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            dir_count += 1
    return c.slm_height // dir_count


def get_precision(path):
    '''finds out precision based on system of number of holograms for one subdomain'''
    file_count = 0
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            file_count += 1
    return file_count


def get_number_of_subdomains(subdomain_size):
    if c.slm_height % subdomain_size != 0 or c.slm_width % subdomain_size != 0:
        print(f"some of the SLM pixels won't be covered, you better choose number which divides {c.slm_height} and {c.slm_width}")
    return c.slm_height//subdomain_size, c.slm_width//subdomain_size


def get_intensity_naive(img_arr: np.array):
    '''returns maximal value on the img'''
    return max(img_arr.flatten())

def get_intensity_coordinates(img_arr: np.array, coordinates: tuple):
    return img_arr[coordinates]


# ---------- functions for work with camera ----------- #

def set_exposure(cam):
    '''set exposure time such that maximal signal value is
    in interval (max_val_lower, max_val_upper)
    '''
    max_val_upper = 256 // 2 # in fully-constructive interference the value of intensity should be twice as high 
    max_val_lower = max_val_upper - 20
    step = 10e-3
    expo = -10e-3
    max_val = - expo
    while max_val < max_val_lower or max_val > max_val_upper:
        if max_val < max_val_lower:
            expo += step
        if max_val > max_val_upper:
            step /= 2
            expo -= step
        cam.set_exposure(expo)
        print(expo)
        frame = cam.snap()
        max_val = max(frame.flatten())
    

def set_exposure_wrt_reference_img(cam, window, hologram_path):
    display_image_on_external_screen(window, hologram_path)
    set_exposure(cam)


def get_highest_intensity_coordinates(cam, window, hologram_path):
    display_image_on_external_screen(window, hologram_path)
    img = cam.snap()
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
