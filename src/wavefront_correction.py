# ! working in constants.u units

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
from skimage.restoration.inpaint import inpaint_biharmonic
import argparse
from pylablib.devices import uc480
from time import time, sleep
import fit_stuff as f
from queue import Queue
from threading import Thread
import help_messages_wfc
from typing import Union, Callable, Dict


def main(args: argparse.Namespace) -> None:
    initialize(args)
    best_phase = choose_phase(args.choose_phase)
    phase_masks = make_blank_phase_masks(
        args.subdomain_scale_shape,
        args.sqrted_number_of_source_pixels**2,
        args.skip_subdomains_out_of_inscribed_circle,
    )
    coordinates_list = make_coordinates_list(args)
    print("mainloop start.")
    count = 0
    for i, j in coordinates_list:
        print(f"\rcalibrating subdomain {count + 1}/{len(coordinates_list)}", end="")
        intensity_list = wavefront_correction_loop(i, j, args)
        fill_pixel_phase_masks(phase_masks, intensity_list, best_phase, i, j, args)
        count += 1
    produce_phase_mask(phase_masks, args)


def main_parallelized(args: argparse.Namespace) -> None:
    initialize(args)
    best_phase = choose_phase(args)
    phase_masks = [
        np.zeros(args.subdomain_scale_shape)
        for _ in range(args.sqrted_number_of_source_pixels**2)
    ]
    coordinates_list = make_coordinates_list(args)
    print("mainloop start.")

    results_queue = Queue()

    def wavefront_correction_worker(i, j):
        intensity_list = wavefront_correction_loop(i, j, args)
        results_queue.put((i, j, intensity_list))

    def fill_pixel_phase_masks_worker():
        while True:
            i, j, intensity_list = results_queue.get()
            if (i, j, intensity_list) is None:
                break
            fill_pixel_phase_masks(phase_masks, intensity_list, best_phase, i, j, args)
            results_queue.task_done()

    # Start the worker thread for fill_pixel_phase_masks
    fill_thread = Thread(target=fill_pixel_phase_masks_worker)
    fill_thread.start()

    count = 0
    for i, j in coordinates_list:
        print(f"\rcalibrating subdomain {count + 1}/{len(coordinates_list)}", end="")
        wavefront_correction_worker(
            i, j
        )  # Call directly to ensure it's on the main thread
        count += 1

    results_queue.join()
    results_queue.put((None, None, None))  # Signal the fill thread to stop
    fill_thread.join()

    produce_phase_mask(phase_masks, args)


# -----------------


def initialize(args: argparse.Namespace) -> None:
    """Fill args with relevant parameters so they can be easily passed to
    wavefront_correction_loop
    """
    args.cam = uc480.UC480Camera()
    args.window = create_tk_window()
    print("creating sample holograms...")
    args.phase_list = [
        i * 2 * np.pi / args.samples_per_period for i in range(args.samples_per_period)
    ]
    sample_list_2pi = make_sample_holograms_2pi(args.deflect, args.phase_list)
    args.samples_list = convert_2pi_holograms_to_int_holograms(
        sample_list_2pi, args.correspond_to2pi
    )
    args.subdomain_scale_shape = get_number_of_subdomains(args.subdomain_size)
    H, W = args.subdomain_scale_shape
    rx, ry = (
        (H // 2, W // 2)
        if args.reference_subdomain_coordinates is None
        else args.reference_subdomain_coordinates
    )
    args.real_reference_subdomain_coordinates = (
        rx * args.subdomain_size,
        ry * args.subdomain_size,
    )
    black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))
    reference_hologram = add_subdomain(
        black_hologram,
        args.samples_list[0],
        args.real_reference_subdomain_coordinates,
        args.subdomain_size,
    )
    print("adjusting exposure time...")
    set_exposure_wrt_reference_img(
        args.cam, args.window, (256 / 4 - 20, 256 / 4), reference_hologram
    )  # in fully-constructive interference the value of amplitude could be twice as high, therefore intensity four times as high
    get_and_show_intensity_coords(
        args.cam, args.window, im.fromarray(args.samples_list[0]), args
    )
    args.hologram = reference_hologram
    args.upper_left_corner = get_upper_left_corner_coords(
        args.intensity_coordinates, args.sqrted_number_of_source_pixels
    )
    args.lower_right_corner = get_lower_right_corner_coords(
        args.intensity_coordinates, args.sqrted_number_of_source_pixels
    )


def wavefront_correction_loop(i: int, j: int, args: argparse.Namespace) -> list:
    """Display wavefront correction holograms and capture the result on the
    Fourier plane with the camera. Return list of intensity measurements.
    """
    i_real = i * args.subdomain_size
    j_real = j * args.subdomain_size
    k = 0
    intensity_list = []
    while k < len(args.phase_list):
        args.hologram = add_subdomain(
            args.hologram, args.samples_list[k], (j_real, i_real), args.subdomain_size
        )
        display_image_on_external_screen(
            args.window, args.hologram
        )  # displays hologram on an external dispaly (SLM)
        frame = args.cam.snap()
        relevant_pixels = square_selection(
            frame, args.upper_left_corner, args.lower_right_corner
        )
        if relevant_pixels.max() == 255:
            print("maximal intensity was reached, adapting...")
            args.cam.set_exposure(
                args.cam.get_exposure() * 0.9
            )  # 10 % decrease of exposure time
            k = 0
            intensity_list = []
            continue
        intensity_list.append(relevant_pixels)
        k += 1
    clear_subdomain(args.hologram, (i_real, j_real), args.subdomain_size)
    return intensity_list


# -----------


def choose_phase(choose_phase: str) -> Callable:
    """Choose function to determine best phase
    according to relevant program argument
    """
    if choose_phase == "fit":
        fit = lambda lst: f.fit_intensity_general(
            lst, f.positive_cos_fixed_wavelength(2 * np.pi), "2pi"
        )
        best_phase = compose_func(return_phase, fit)
    elif choose_phase == "trick":
        best_phase = trick
    return best_phase


def compose_func(func1: Callable, func2: Callable) -> Callable:
    return lambda x: func1(func2(x))


def return_phase(dict: Dict):
    return dict["phase_shift"]


def make_blank_phase_masks(
    shape: tuple, number_of_masks: int, skip_subdomains_out_of_inscribed_circle: bool
) -> list:
    """Return list with demanded number of blank phase masks of demanded shape"""
    phase_masks = [np.zeros(shape) for _ in range(number_of_masks)]
    if skip_subdomains_out_of_inscribed_circle:
        phase_masks = [
            ma.masked_array(phase_mask, mask=make_circular_mask(shape))
            for phase_mask in phase_masks
        ]
    return phase_masks


def make_circular_mask(shape: tuple) -> list:
    condition = circular_hole_inclusive_condition
    return np.array(
        [
            [(0 if condition(i, j, shape) else 1) for j in range(shape[1])]
            for i in range(shape[0])
        ]
    )


def make_coordinates_list(args: argparse.Namespace) -> list:
    """Return list of subdomain coordinates to iterate through
    in the main process
    """
    H, W = args.subdomain_scale_shape
    j0, i0 = (
        (H // 2, W // 2)
        if args.reference_subdomain_coordinates is None
        else args.reference_subdomain_coordinates
    )
    if args.skip_subdomains_out_of_inscribed_circle:
        coordinates_list = [
            (i, j)
            for i in range(H)
            for j in range(W)
            if circular_hole_inclusive_condition(i, j, (H, W))
            and not (i == i0 and j == j0)
        ]
    else:
        coordinates_list = [
            (i, j) for i in range(H) for j in range(W) if not (i == i0 and j == j0)
        ]
    if args.shuffle:
        np.random.shuffle(coordinates_list)
    return coordinates_list


def circular_hole_inclusive_condition(i: int, j: int, shape: tuple) -> bool:
    h, w = shape
    R = h // 2 + 1
    i0, j0 = h // 2, w // 2
    return (i - i0) ** 2 + (j - j0) ** 2 < R**2


# -------------- Phase mask production -------------- #


def produce_phase_mask(phase_masks: list, args: argparse.Namespace) -> None:
    mean_phase_mask = combine_phase_masks(phase_masks)
    if args.remove_defocus_compensation:
        mean_phase_mask = fit_and_subtract_masked(
            mean_phase_mask, quadratic_func, [0, 0]
        )
    produce_phase_mask_single(mean_phase_mask, args)


def produce_phase_mask_single(phase_mask: np.array, args: argparse.Namespace) -> None:
    specification = make_specification_phase_mask(args)
    if ma.is_masked(phase_mask):
        array_to_save = ma.filled(phase_mask, fill_value=0)
    else:
        array_to_save = phase_mask
    save_path = originalize_name(f"{args.dest_dir}/{specification}.npy")
    np.save(save_path, array_to_save)
    big_phase_mask = resize_2d_array(phase_mask, (c.slm_height, c.slm_width))
    big_phase_mask_mod = big_phase_mask % (2 * np.pi)
    mask_to_save = (
        inpaint_biharmonic(big_phase_mask_mod, np.isnan(big_phase_mask))
        if args.skip_subdomains_out_of_inscribed_circle
        else big_phase_mask_mod
    )
    save_path = originalize_name(f"{args.dest_dir}/smoothed_{specification}.npy")
    np.save(save_path, mask_to_save)
    save_phase_mask(
        mask_to_save * args.correspond_to2pi / (2 * np.pi),
        args.dest_dir,
        "smoothed_" + specification,
    )


def combine_phase_masks(phase_masks: list) -> np.array:
    mean_phase_mask = np.zeros(phase_masks[0].shape)
    if ma.is_masked(phase_masks[0]):
        mean_phase_mask = ma.array(mean_phase_mask, mask=phase_masks[0].mask)
    for phase_mask in phase_masks:
        phase_mask = unwrap_phase(phase_mask - np.pi)
        phase_mask = fit_and_subtract_masked(phase_mask, linear_func, [0, 0, 0])
        mean_phase_mask += phase_mask
    mean_phase_mask /= len(phase_masks)
    return mean_phase_mask


def make_specification_phase_mask(args: argparse.Namespace) -> str:
    """Return core name of the mask to be saved under"""
    return f"phase_mask\
            _{args.wavefront_correction_name}\
            _ss{args.subdomain_size}\
            _ct2pi_{args.correspond_to2pi}\
            _spp_{args.samples_per_period}\
            _defl_{args.deflect[0]}\
            _{args.deflect[1]}\
            _ref_{args.reference_subdomain_coordinates}\
            _ic_{args.intensity_coordinates[0]}\
            _{args.intensity_coordinates[1]}\
            _nsp_{args.sqrted_number_of_source_pixels}"


def save_phase_mask(phase_mask: np.array, dest_dir: str, name: str) -> None:
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    phase_mask_img = im.fromarray(phase_mask)
    original_name = originalize_name(f"{dest_dir}/{name}.png")
    phase_mask_img.convert("L").save(original_name)


def originalize_name(name: str) -> str:
    """If there already exists file `name`
    add there a number so the new name is original
    """
    if not os.path.exists(name):
        return name
    base, ext = os.path.splitext(name)
    i = 1
    while True:
        new_name = f"{base}_{i}{ext}"
        if not os.path.exists(new_name):
            return new_name
        i += 1


def resize_2d_array(array: np.array, new_shape: tuple) -> np.array:
    """Smooth phase mask by resizing with linear interpolation"""
    if ma.is_masked(array):
        array = array.filled(fill_value=np.nan)
    zoom_factors = [n / o for n, o in zip(new_shape, array.shape)]
    resized_array = zoom(array, zoom_factors, order=1)
    return resized_array


def linear_func(params: tuple, x: int, y: int) -> float:
    a, b, c = params
    return a * x + b * y + c


def quadratic_func(params: tuple, x: int, y: int) -> float:
    a, b = params
    return a * (x**2 + y**2) + b


def fit_and_subtract_masked(
    array: np.array, fit_func: Callable, initial_guess: tuple
) -> np.array:
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
    params, _ = leastsq(
        error_func,
        initial_guess,
        args=(x_flat_masked, y_flat_masked, array_flat_masked),
    )

    # Compute the fitted values
    fitted_values = fit_func(params, x, y)

    # Subtract the fitted values from the original array
    result_array = array - fitted_values

    # Return a masked array with the original mask
    result_array = ma.array(result_array, mask=array.mask)

    return result_array


def expand_phase_mask(phase_mask: np.array, subdomain_size: int) -> np.array:
    """takes a phase mask where one pixel represents a subdomain
    and returns a phase mask where one pixel represents a pixel in SLM
    """
    h, w = phase_mask.shape
    if ma.is_masked(phase_mask):
        phase_mask = ma.filled(phase_mask, fill_value=0)
    ss = subdomain_size
    big_phase_mask = np.zeros((h * ss, w * ss))
    for i in range(h):
        for k in range(ss):
            for j in range(w):
                for p in range(ss):
                    big_phase_mask[ss * i + k, ss * j + p] = int(phase_mask[i, j])
    return big_phase_mask


# ---------- sample holograms ----------- #


def make_sample_holograms_2pi(angle: tuple, phase_list: list) -> list:
    """Return list of whole-SLM holograms
    which will be cut to create wavefront correction holograms
    """
    sample = []
    base = deflect_2pi(angle)
    for phase in phase_list:
        sample.append((base + phase) % (2 * np.pi))
    return sample


def deflect_2pi(angle: tuple) -> np.array:
    """Return hologram for deflecting light."""
    x_angle, y_angle = angle
    hologram = np.zeros((c.slm_height, c.slm_width))
    const = 2 * np.pi * c.px_distance / c.wavelength
    for i in range(c.slm_height):
        for j in range(c.slm_width):
            new_phase = const * (np.sin(y_angle * c.u) * i + np.sin(x_angle * c.u) * j)
            hologram[i, j] = new_phase % (2 * np.pi)
    return hologram


def convert_2pi_holograms_to_int_holograms(sample: list, ct2pi: int) -> list:
    return [
        convert_2pi_hologram_to_int_hologram(hologram, ct2pi) for hologram in sample
    ]


def convert_2pi_hologram_to_int_hologram(hologram: np.array, ct2pi: int) -> np.array:
    return np.round(hologram * ct2pi / (2 * np.pi)).astype(np.uint8)


# ----------- subdomain manipulation ------------ #


def add_subdomain(
    hologram: im.Image, sample: np.array, img_coordinates: tuple, subdomain_size: int
) -> im:
    """Paste square section of `sample` to `hologram."""
    i0, j0 = img_coordinates
    for i in range(subdomain_size):
        for j in range(subdomain_size):
            hologram.putpixel((i0 + i, j0 + j), int(sample[j0 + j, i0 + i]))
    return hologram


def clear_subdomain(hologram: im.Image, coordinates: tuple, subdomain_size: int) -> im:
    i0, j0 = coordinates
    for i in range(subdomain_size):
        for j in range(subdomain_size):
            hologram.putpixel((j0 + j, i0 + i), 0)
    return hologram


def get_number_of_subdomains(subdomain_size: int) -> int:
    """Return actual number of subdomains based on `subdomain_size`
    and size of SLM screen.
    """
    if c.slm_height % subdomain_size != 0 or c.slm_width % subdomain_size != 0:
        print(
            f"some of the SLM pixels won't be covered,\
                you better choose subdomain size which divides\
                {c.slm_height} and {c.slm_width}"
        )
    return c.slm_height // subdomain_size, c.slm_width // subdomain_size


# ---------- camera, image processing ----------- #


def set_exposure_wrt_reference_img(
    cam: uc480.UC480Camera,
    window: tk.Tk,
    intensity_range: tuple,
    hologram: Union[im.Image, str],
    num_to_avg: int = 8,
) -> None:
    display_image_on_external_screen(window, hologram)
    set_exposure(cam, intensity_range, num_to_avg)


def set_exposure(
    cam: uc480.UC480Camera, intensity_range: tuple, num_to_avg: int
) -> None:
    """set exposure time such that maximal signal value is
    in interval (max_val_lower, max_val_upper)
    """

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
        max_val = max(avgd_frame.flatten())


def average_frames(cam: uc480.UC480Camera, num_to_avg: int) -> np.array:
    frame = cam.snap()
    frame = np.array(frame).astype(float)
    for _ in range(1, num_to_avg):
        new_frame = np.array(cam.snap()).astype(float)
        frame += new_frame
    frame /= num_to_avg
    return frame


def get_and_show_intensity_coords(
    cam: uc480.UC480Camera,
    window: tk.Tk,
    hologram: Union[im.Image, str],
    args: argparse.Namespace,
) -> None:
    """Take a photo and use it to find suitable coordinates
    for intensity measurement (intensity_coordinates) if not provided.
    Mark the place defined by the coordinates on the photo with a red cross.
    Show photo with marked place and record the coordinates to the args namespace.
    """
    display_image_on_external_screen(window, hologram)
    img = cam.snap()
    if args.intensity_coordinates is None:
        args.intensity_coordinates = mean_position_of_overflow_pixel(img)
        print(f"intensity coordinates: {args.intensity_coordinates}")
    show_coords_on_img(cam.snap(), args.intensity_coordinates)


def mean_position_of_overflow_pixel(picture: np.array) -> tuple:
    """Return coordinates of mean position of pixels with maximal value (255)
    on given picture.
    """
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


def weighted_mean_position_of_grey_pixel(picture: np.array, treshold: int = 200):
    h, w = picture.shape
    sum = np.array([0, 0])
    norm = 0.0
    for y in range(h):
        for x in range(w):
            if picture[y, x] > treshold:
                sum += np.array([y, x]) * picture[y, x]
                norm += picture[y, x]
    a, b = sum / norm
    return (int(round(a)), int(round(b)))


def show_coords_on_img(array_img: np.array, coords: tuple) -> None:
    """Add a red cross on `array_img` at `coords` and show it"""
    img_img = im.fromarray(array_img, "L").convert("RGB")
    marked_img = e.add_cross(img_img, coords)
    marked_img.show()


# ----------- displaying on SLM ----------- #


def create_tk_window() -> tk.Tk:
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


def display_image_on_external_screen(
    window: tk.Tk, image: Union[im.Image, str]
) -> None:
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
    label.photo = photo  # makes the image to persist through a while cycle

    # Update the window to display the new image
    window.update()

    sleep(0.017)  # cca 1/60 s which is the refresh rate of the SLM


# ---------------


def get_upper_left_corner_coords(middle_coords: tuple, square_size: int) -> tuple:
    """Return coordinates of an upper left corner of
    a square of size `square_size`
    with middle on `middle_coords`
    """
    x, y = middle_coords
    half_square = (square_size - 1) // 2
    x_lc = x - half_square
    y_lc = y - half_square
    return x_lc, y_lc


def get_lower_right_corner_coords(middle_coords: tuple, square_size: int) -> tuple:
    """Return coordinates of an lower right corner of
    a square of size `square_size`
    with middle on `middle_coords`
    """
    x, y = middle_coords
    half_square = square_size - ((square_size - 1) // 2)
    x_lc = x + half_square
    y_lc = y + half_square
    return x_lc, y_lc


def square_selection(
    frame: np.array, upper_left_corner: tuple, lower_right_corner: tuple
) -> np.array:
    """Return square section of `frame`
    defined by `upper_left_corner` and `lower_right_corner`.
    """
    i_ul, j_ul = upper_left_corner
    i_lr, j_lr = lower_right_corner
    return frame[i_ul:i_lr, j_ul:j_lr]


def fill_pixel_phase_masks(
    phase_masks: list,
    intensity_list: list,
    best_phase: Callable,
    i: int,
    j: int,
    args: argparse.Namespace,
) -> None:
    """Find the best phase shift for based on given phase list
    and record it to phase mask on coordinates (`i`, `j`)
    """
    for k in range(len(phase_masks)):
        intensity_list_k = [intensity.flatten()[k] for intensity in intensity_list]
        phase_masks[k][i, j] = best_phase([args.phase_list, intensity_list_k])


def trick(phase_intensity_list: list) -> float:
    """function to choose best phase shift
    based on some math magic
    """
    imaginary_part = trick_function(phase_intensity_list, np.sin)
    real_part = trick_function(phase_intensity_list, np.cos)
    return np.angle(real_part + 1j * imaginary_part)


def trick_function(phase_intensity_list: list, fun: Callable) -> float:
    phase_list = phase_intensity_list[0]
    intensity_list = phase_intensity_list[1]
    return sum([intensity_list[i] * fun(phase_list[i]) for i in range(len(phase_list))])


# ---------------

if __name__ == "__main__":
    description = """Create phase mask for transmissive phase-only SLM
    which compensates aberrations in optical path
    and curvature of SLM itself usinig Tomas Cizmar's wave front correction.
    The mask is saved in holograms/wavefront_correction_phase_masks in three formats:
    upscaled & smoothed .npy for use (keyword "smoothed" at the beginning of its name),
    corresponding .png for preview and raw .npy for further processing.
    The mask should be added up with any projected hologram.
    For each optical path there should be generated its own mask.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description
    )

    parser.add_argument(
        "wavefront_correction_name",
        type=str
    )
    parser.add_argument(
        "-ss",
        "--subdomain_size",
        metavar="INT",
        type=int,
        default=32,
        help=help_messages_wfc.subdomain_size,
    )
    parser.add_argument(
        "-spp",
        "--samples_per_period",
        metavar="INT",
        type=int,
        default=4,
        help=help_messages_wfc.samples_per_period,
    )
    parser.add_argument(
        "-d",
        "--deflect",
        metavar=("X_ANGLE", "Y_ANGLE"),
        nargs=2,
        type=float,
        default=(0.5, 0.5),
        help=help_messages_wfc.deflect,
    )
    parser.add_argument(
        "-c",
        "--reference_subdomain_coordinates",
        metavar=("X_COORD", "Y_COORD"),
        nargs=2,
        type=int,
        default=None,
        help=help_messages_wfc.reference_subdomain_coordinates,
    )
    parser.add_argument(
        "-ct2pi",
        "--correspond_to2pi",
        metavar="INT",
        type=int,
        required=True,
        help=help_messages_wfc.ct2pi,
    )
    parser.add_argument(
        "-skip",
        "--skip_subdomains_out_of_inscribed_circle",
        action="store_true",
        help=help_messages_wfc.skip_subdomains_out_of_inscribed_circle,
    )
    parser.add_argument(
        "-shuffle",
        action="store_true",
        help=help_messages_wfc.shuffle
    )
    parser.add_argument(
        "-ic",
        "--intensity_coordinates",
        metavar=("X_COORD", "Y_COORD"),
        nargs=2,
        type=int,
        default=None,
        help=help_messages_wfc.intensity_coordinates,
    )
    parser.add_argument(
        "-cp",
        "--choose_phase",
        type=str,
        choices=["trick", "fit"],
        default="trick",
        help=help_messages_wfc.choose_phase,
    )
    parser.add_argument(
        "-nsp",
        "--sqrted_number_of_source_pixels",
        type=int,
        default=1,
        help=help_messages_wfc.sqrted_number_of_source_pixels,
    )
    parser.add_argument(
        "-parallel",
        action="store_true",
        help="run capturing of photos and computation of optimal phase offset in parallel.\
            It is recommended to use this option when the number of subdomains is high\
            and optimal phase is chosen by fit."
    )
    parser.add_argument(
        "-rd",
        "--remove_defocus_compensation",
        action="store_true",
        help=help_messages_wfc.remove_defocus_compensation,
    )

    args = parser.parse_args()
    args.dest_dir = "holograms/wavefront_correction_phase_masks"
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    start = time()
    if args.parallel:
        main_parallelized(args)
    else:
        main(args)
    print("\nexecution_time: ", round((time() - start) / 60, 1), " min")
