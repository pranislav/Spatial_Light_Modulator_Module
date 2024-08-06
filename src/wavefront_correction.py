'''creates phase mask for LC-SLM which compensates
aberrations caused both by the modulator and whole optical path.
This mask should be added up with any projected hologram.
For each optical path there should be generated its own mask.
There is implemented Tomas Cizmar's approach here.
'''

# ! working in constants.u units

from wavefront_correction_lib import *
import numpy as np
import argparse
from pylablib.devices import uc480
from time import time
import fit_stuff as f
from queue import Queue
from threading import Thread
import help_messages_wfc



def wavefront_correction(args):
    initialize(args)
    best_phase = choose_phase(args)
    phase_masks = [np.zeros(args.subdomain_scale_shape) for _ in range(args.sqrted_number_of_source_pixels ** 2)]
    if args.skip_subdomains_out_of_inscribed_circle:
        phase_masks = [ma.masked_array(phase_mask, mask=make_circular_mask(args.subdomain_scale_shape)) for phase_mask in phase_masks]
    coordinates_list = make_coordinates_list(args)
    print("mainloop start.")
    count = 0
    for i, j in coordinates_list:
        print(f"\rcalibrating subdomain {count + 1}/{len(coordinates_list)}", end="")
        intensity_list = wavefront_correction_loop(i, j, args)
        fill_pixel_phase_masks(phase_masks, intensity_list, best_phase, i, j, args)
        count += 1
    produce_phase_mask(phase_masks, args)

def wavefront_correction_parallelized(args):
    initialize(args)
    best_phase = choose_phase(args)
    phase_masks = [np.zeros(args.subdomain_scale_shape) for _ in range(args.sqrted_number_of_source_pixels ** 2)]
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
        wavefront_correction_worker(i, j)  # Call directly to ensure it's on the main thread
        count += 1

    results_queue.join()
    results_queue.put((None, None, None))  # Signal the fill thread to stop
    fill_thread.join()
    
    produce_phase_mask(phase_masks, args)


def make_circular_mask(shape):
    condition = circular_hole_inclusive_condition
    return np.array([[(0 if condition(i, j, shape) else 1) for j in range(shape[1])] for i in range(shape[0])])


def choose_phase(args):
    if args.choose_phase == "fit":
        fit = lambda lst: f.fit_intensity_general(lst, f.positive_cos_fixed_wavelength(2 * np.pi), "2pi")  # TODO other options?
        best_phase = compose_func(return_phase, fit)
    elif args.choose_phase == "trick":
        best_phase = lambda lst: trick(lst)
    return best_phase

def compose_func(func1, func2):
    return lambda x: func1(func2(x))
    
def return_phase(dict):
    return dict["phase_shift"]


def make_coordinates_list(args):
    H, W = args.subdomain_scale_shape
    j0, i0 = (H // 2, W // 2) if args.reference_coordinates is None else args.reference_coordinates
    if args.skip_subdomains_out_of_inscribed_circle:
        coordinates_list = [(i, j) for i in range(H) for j in range(W) if circular_hole_inclusive_condition(i, j, (H, W)) and not (i == i0 and j == j0)]
    else:
        coordinates_list = [(i, j) for i in range(H) for j in range(W) if not (i == i0 and j == j0)]
    if args.shuffle:
        np.random.shuffle(coordinates_list)
    return coordinates_list

def circular_hole_inclusive_condition(i, j, shape):
    h, w = shape
    R = h // 2 + 1
    i0, j0 = h // 2, w // 2
    return (i - i0)**2 + (j - j0)**2 < R**2

def print_estimate(outer_loops_num, start_loops):
    time_elapsed = time() - start_loops
    time_interval = (outer_loops_num - 1) * time_elapsed
    print(f"estimated remaining time is {round(time_interval / 60, 1)} min")

def initialize(args):
    args.cam = uc480.UC480Camera()
    args.window = create_tk_window()
    print("creating sample holograms...")
    args.phase_list = [i * 2 * np.pi / args.samples_per_period for i in range(args.samples_per_period)]
    sample_list_2pi = make_sample_holograms_2pi(args.deflect, args.phase_list)
    args.samples_list = convert_phase_holograms_to_color_holograms(sample_list_2pi, args.correspond_to2pi)
    args.subdomain_scale_shape = get_number_of_subdomains(args.subdomain_size)
    H, W = args.subdomain_scale_shape
    rx, ry = (H // 2, W // 2) if args.reference_coordinates is None else args.reference_coordinates
    args.real_reference_coordinates = (rx * args.subdomain_size, ry * args.subdomain_size)
    black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))
    reference_hologram = add_subdomain(black_hologram, args.samples_list[0], args.real_reference_coordinates, args.subdomain_size)
    print("adjusting exposure time...")
    set_exposure_wrt_reference_img(args.cam, args.window, (256 / 4 - 20, 256 / 4), reference_hologram) # in fully-constructive interference the value of amplitude could be twice as high, therefore intensity four times as high 
    get_and_show_intensity_coords(args.cam, args.window, im.fromarray(args.samples_list[0]), args)
    args.hologram = reference_hologram
    args.upper_left_corner = get_upper_left_corner_coords(args.intensity_coordinates, args.sqrted_number_of_source_pixels)
    args.lower_right_corner = get_lower_right_corner_coords(args.intensity_coordinates, args.sqrted_number_of_source_pixels)


def wavefront_correction_loop(i, j, args):
    i_real = i * args.subdomain_size
    j_real = j * args.subdomain_size
    k = 0
    nsp = args.sqrted_number_of_source_pixels
    intensity_list = []
    while k < len(args.phase_list):
        args.hologram = add_subdomain(args.hologram, args.samples_list[k], (j_real, i_real), args.subdomain_size)
        display_image_on_external_screen(args.window, args.hologram) # displays hologram on an external dispaly (SLM)
        frame = args.cam.snap()
        relevant_pixels = square_selection(frame, args.upper_left_corner, args.lower_right_corner)
        if relevant_pixels.max() == 255:
            print("maximal intensity was reached, adapting...")
            args.cam.set_exposure(args.cam.get_exposure() * 0.9) # 10 % decrease of exposure time
            k = 0
            intensity_list = []
            continue
        intensity_list.append(relevant_pixels)
        k += 1
    clear_subdomain(args.hologram, (i_real, j_real), args.subdomain_size)
    return intensity_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script creates phase mask which compensates aberrations in optical path and curvature of SLM itself")

    parser.add_argument('wavefront_correction_name', type=str)
    parser.add_argument('-ss', '--subdomain_size', metavar="INT", type=int, default=32, help=help_messages_wfc.subdomain_size)
    parser.add_argument('-spp', '--samples_per_period', metavar="INT", type=int, default=4, help=help_messages_wfc.samples_per_period)
    parser.add_argument('-d', '--deflect', metavar=("X_ANGLE", "Y_ANGLE"), nargs=2, type=float, default=(0.5, 0.5), help=help_messages_wfc.deflect)
    parser.add_argument('-c', '--reference_coordinates', metavar=("X_COORD", "Y_COORD"), nargs=2, type=int, default=None, help=help_messages_wfc.reference_subdomain_coordinates)
    parser.add_argument('-ct2pi', '--correspond_to2pi', metavar="INT", type=int, required=True, help=help_messages_wfc.ct2pi)
    parser.add_argument('-skip', '--skip_subdomains_out_of_inscribed_circle', action="store_true", help=help_messages_wfc.skip_subdomains_out_of_inscribed_circle)
    parser.add_argument("-shuffle", action="store_true", help=help_messages_wfc.shuffle)
    parser.add_argument('-ic', "--intensity_coordinates", metavar=("X_COORD", "Y_COORD"), nargs=2, type=int, default=None, help=help_messages_wfc.intensity_coordinates)
    parser.add_argument('-cp', '--choose_phase', type=str, choices=["trick", "fit"], default="trick", help=help_messages_wfc.choose_phase)
    # parser.add_argument('-resample', type=str, choices=["bilinear", "bicubic"], default="bilinear", help="smoothing method used to upscale the unwrapped phase mask")
    parser.add_argument('-nsp', '--sqrted_number_of_source_pixels', type=int, default=1, help=help_messages_wfc.sqrted_number_of_source_pixels)
    parser.add_argument('-parallel', action="store_true", help="use parallelization")
    parser.add_argument('-rd', '--remove_defocus', action="store_true", help=help_messages_wfc.remove_defocus)

    args = parser.parse_args()
    args.dest_dir = "holograms/wavefront_correction_phase_masks"
    start = time()
    if args.parallel:
        wavefront_correction_parallelized(args)
    else:
        wavefront_correction(args)
    print("\nexecution_time: ", round((time() - start) / 60, 1),  " min")
