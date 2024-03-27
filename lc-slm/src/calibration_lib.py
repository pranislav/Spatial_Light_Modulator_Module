import constants as c
import tkinter as tk
from PIL import Image as im, ImageTk
from screeninfo import get_monitors
import numpy as np
import os


def get_subdomain_size(path_to_hologram):
    pass # TODO

def get_precision(path):
    pass # TODO: return number of files in dir at path

def get_number_of_subdomains(subdomain_size):
    if c.slm_height % subdomain_size != 0 or c.slm_width % subdomain_size != 0:
        print(f"some of the SLM pixels won't be covered, you better choose number which divides {c.slm_height} and {c.slm_width}")
    return c.slm_height//subdomain_size, c.slm_width//subdomain_size


def get_intensity_naive(img):
    '''returns maximal value on the img'''
    return max(img.flatten())


def create_phase_mask(phase_mask, subdomain_size, name):
    '''creates and saves phase mask image based on phase mask array
    '''
    h, w = phase_mask.shape
    ss = subdomain_size
    phase_mask_img = im.new("L", (h * ss, w * ss))
    for i in range(h):
        for k in range(ss):
            for j in range(w):
                for p in range(ss):
                    phase_mask_img.putpixel((ss * i + k, ss* j + p), int(phase_mask[i, j]))
    dest_dir = "lc-slm/holograms/calibrtion_phase_masks"
    if not os.path.exists(dest_dir): os.makedirs(dest_dir)
    phase_mask_img.save(f"{dest_dir}/{name}_phase_mask.png")

a = np.random.randint(0, 256, (100, 80))
create_phase_mask(a, 8, "lc-slm/images/create_phase_mask_demo")


def set_exposure(cam):
    '''set exposure time such that maximal signal value is
    in interval (max_val_lower, max_val_upper)
    '''
    max_val_lower = 200
    max_val_upper = 220
    step = 1e-3
    expo = 1e-3
    max_val = - expo
    while max_val < max_val_lower or max_val > max_val_upper:
        if max_val < max_val_lower:
            expo += step
        if max_val > max_val_upper:
            step /= 2
            expo -= step
        cam.set_exposure(expo)
        frame = cam.snap()
        max_val = max(frame.flatten())
    

def set_exposure_wrt_reference_img(cam, window, hologram_path):
    display_image_on_external_screen(window, hologram_path)
    set_exposure(cam)


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

    return window

image_path = "lc-slm/images/example_image.jpg"

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

    # Update the window to display the new image
    window.update()



