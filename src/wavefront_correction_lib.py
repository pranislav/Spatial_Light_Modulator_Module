import constants as c
import tkinter as tk
from PIL import Image as im, ImageTk
from screeninfo import get_monitors
import numpy as np
import os
import explore_wavefront_correction as e
import time
from skimage.restoration import unwrap_phase
from scipy.optimize import leastsq
import numpy.ma as ma
from scipy.ndimage import zoom



def produce_phase_mask_single(phase_mask, args):
    specification = make_specification_phase_mask(args)
    np.save(f"{args.dest_dir}/{specification}.npy", phase_mask)
    # big_phase_mask = expand_phase_mask((phase_mask % (2 * np.pi)) * args.correspond_to2pi / (2 * np.pi), args.subdomain_size)
    # save_phase_mask(big_phase_mask, args.dest_dir, specification)
    big_phase_mask = resize_2d_array(phase_mask, (c.slm_height, c.slm_width))
    name = "smoothed_" + specification
    np.save(f"{args.dest_dir}/{name}.npy", big_phase_mask)
    save_phase_mask(big_phase_mask * args.correspond_to_2pi % (2 * np.pi), args.dest_dir, name)


def combine_phase_masks(phase_masks):
    mean_phase_mask = np.zeros(phase_masks[0].shape)
    for phase_mask in phase_masks:
        phase_mask = unwrap_phase(phase_mask - np.pi)
        phase_mask = fit_and_subtract_masked(phase_mask, linear_func, [0, 0, 0])
        mean_phase_mask += phase_mask
    mean_phase_mask /= len(phase_masks)
    # print(f"mean_phase_mask is {"" if ma.is_masked(mean_phase_mask) else "not"} masked")
    return mean_phase_mask

def produce_phase_mask(phase_masks, args):
    mean_phase_mask = combine_phase_masks(phase_masks)
    if args.remove_defocus:
        mean_phase_mask = fit_and_subtract_masked(mean_phase_mask, quadratic_func, [0, 0])
    produce_phase_mask_single(mean_phase_mask, "phase_mask", args)

def resize_2d_array(array, new_shape):
    """
    Resize a 2D array using bilinear interpolation.

    Parameters:
    - array (numpy.ndarray): The 2D input array to resize.
    - new_shape (tuple): The desired shape for the resized array.

    Returns:
    - numpy.ndarray: The resized 2D array.
    """
    if ma.is_masked(array):
        array = array.filled(fill_value=np.nan)
    zoom_factors = [n / o for n, o in zip(new_shape, array.shape)]
    resized_array = zoom(array, zoom_factors, order=1)
    return np.nan_to_num(resized_array, nan=0)

def linear_func(params, x, y):
    a, b, c = params
    return a * x + b * y + c

def  quadratic_func(params, x, y):
    a, b = params
    return a * (x **2 + y ** 2) + b



def fit_and_subtract_masked(array, fit_func, initial_guess):
    # Determine if the array is masked
    is_masked = ma.is_masked(array)
    
    # If not masked, convert to masked array with no mask
    if not is_masked:
        array = ma.array(array, mask=np.zeros_like(array, dtype=bool))
    
    # Get the shape of the array
    ny, nx = array.shape
    
    # Generate x and y coordinates
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    
    # Flatten the arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    array_flat = array.flatten()
    
    # Get the mask and apply it to the flattened arrays
    mask_flat = array.mask.flatten()
    x_flat_masked = x_flat[~mask_flat]
    y_flat_masked = y_flat[~mask_flat]
    array_flat_masked = array_flat[~mask_flat]
    
    # Define the error function
    def error_func(params, x, y, z):
        return z - fit_func(params, x, y)
    
    # Perform the least squares fitting
    params, _ = leastsq(error_func, initial_guess, args=(x_flat_masked, y_flat_masked, array_flat_masked))
    
    # Compute the fitted values
    fitted_values = fit_func(params, x, y)
    
    # Subtract the fitted values from the original array
    result_array = array - fitted_values
    
    # Return a masked array with the original mask
    result_array = ma.array(result_array, mask=array.mask)
    
    return result_array



def make_specification_phase_mask(args):
    return f"phase_mask_{args.wavefront_correction_name}_ss{args.subdomain_size}_ct2pi_{args.correspond_to2pi}_samples_per_period_{args.samples_per_period}_x{args.decline[0]}_y{args.decline[1]}_ref_{args.reference_coordinates}_intensity_coords_{args.intensity_coordinates[0]}_{args.intensity_coordinates[1]}_source_pxs_{args.sqrted_number_of_source_pixels}"



def get_upper_left_corner_coords(middle_coords, square_size):
    x, y = middle_coords
    half_square = (square_size - 1) // 2
    x_lc = x - half_square
    y_lc = y - half_square
    return x_lc, y_lc

def get_lower_right_corner_coords(middle_coords, square_size):
    x, y = middle_coords
    half_square = square_size - ((square_size - 1) // 2)
    x_lc = x + half_square
    y_lc = y + half_square
    return x_lc, y_lc

def square_selection(frame, upper_left_corner, lower_right_corner):
    i_ul, j_ul = upper_left_corner
    i_lr, j_lr = lower_right_corner
    return frame[i_ul:i_lr, j_ul:j_lr]


def mean_best_phase(intensity_list, best_phase, args):
    mean_best_phase = 0
    h, w = intensity_list[0].shape
    for i in range(h):
        for j in range(w):
            intensity_list_ij = [intensity[i, j] for intensity in intensity_list]
            mean_best_phase += best_phase([args.phase_list, intensity_list_ij])
    return mean_best_phase / (h * w)


def fill_pixel_phase_masks(phase_masks, intensity_list, best_phase, i, j, args):
    for k in range(len(phase_masks)):
        intensity_list_k = [intensity.flatten()[k] for intensity in intensity_list]
        phase_masks[k][i, j] = best_phase([args.phase_list, intensity_list_k])
        # print("*", end="")


# ----------- best phase() -------------- #

def naive(phase_list):
    opt_index = phase_list[1].index(max(phase_list[1]))
    return phase_list[0][opt_index]


def trick(phase_intensity_list):
    imaginary_part = trick_function(phase_intensity_list, np.sin)
    real_part = trick_function(phase_intensity_list, np.cos)
    return (np.angle(real_part + 1j * imaginary_part))

def trick_function(phase_intensity_list, fun):
    phase_list = phase_intensity_list[0]
    intensity_list = phase_intensity_list[1]
    return sum([intensity_list[i] * fun(phase_list[i]) for i in range(len(phase_list))])



# ---------- sample holograms ----------- #

def make_sample_holograms(angle, samples_per_period, ct2pi):
    sample = []
    sample.append(decline(angle, ct2pi))
    for i in range(1, samples_per_period):
        offset = i * ct2pi // samples_per_period
        sample.append((sample[0] + offset) % ct2pi)
    return sample

def decline(angle, ct2pi):
    x_angle, y_angle = angle
    hologram = np.zeros((c.slm_height, c.slm_width))
    const = ct2pi * c.px_distance / c.wavelength
    for i in range(c.slm_height):
        for j in range(c.slm_width):
            new_phase = const * (np.sin(y_angle * c.u) * i + np.sin(x_angle * c.u) * j)
            hologram[i, j] = int(new_phase % ct2pi)
    return hologram


def make_sample_holograms_2pi(angle, phase_list):
    sample = []
    sample.append(decline_2pi(angle))
    for phase in phase_list:
        sample.append((sample[0] + phase) % (2 * np.pi))
    return sample

def decline_2pi(angle):
    x_angle, y_angle = angle
    hologram = np.zeros((c.slm_height, c.slm_width))
    const = 2 * np.pi * c.px_distance / c.wavelength
    for i in range(c.slm_height):
        for j in range(c.slm_width):
            new_phase = const * (np.sin(y_angle * c.u) * i + np.sin(x_angle * c.u) * j)
            hologram[i, j] = new_phase % (2 * np.pi)
    return hologram

def convert_phase_holograms_to_color_holograms(sample, ct2pi):
    return [convert_phase_hologram_to_color_hologram(hologram, ct2pi) for hologram in sample]

def convert_phase_hologram_to_color_hologram(hologram, ct2pi):
    return np.round(hologram * ct2pi / (2 * np.pi)).astype(np.uint8)


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
    if ma.is_masked(phase_mask):
        phase_mask = np.ma.filled(phase_mask, fill_value=0)               
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

def set_exposure_wrt_reference_img(cam, window, intensity_range, hologram, num_to_avg=8):
    display_image_on_external_screen(window, hologram)
    set_exposure(cam, intensity_range, num_to_avg)

# def set_exposure_wrt_reference_img_path(cam, window, intensity_range, hologram_path, num_to_avg):
#     display_image_on_external_screen(window, hologram_path)
#     set_exposure(cam, intensity_range, num_to_avg)

def get_and_show_intensity_coords(cam, window, hologram, args):
    display_image_on_external_screen(window, hologram)
    img = cam.snap()
    if args.intensity_coordinates is None:
        args.intensity_coordinates = mean_position_of_overflow_pixel(img)
        print(f"intensity coordinates: {args.intensity_coordinates}")
    show_coords_on_img(cam.snap(), args.intensity_coordinates)


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
