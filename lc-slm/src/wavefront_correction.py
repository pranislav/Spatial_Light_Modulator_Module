'''creates phase mask for LC-SLM which compensates
aberrations caused both by the modulator and whole optical path.
This mask should be added up with any projected hologram.
For each optical path there should be generated its own mask.

Principle:
modulator's screen is divided into square subdomains
for each subdomain we are searching for optimal phase offset
optimal phase offset is found as following:
let us have a reference subdomain which deflects light to particular angle
for each subdomain we do following:
the subdomain deflects light to the exact same angle as the reference one
all the others subdomains are "off"
the hologram on the subdomain is shifted by a constant phase several times
the phase shift which causes best constructive interference with the reference subdomain
is chosen and written into the phase mask
the quality of the interference is decided by
measuring intensity at the end of the optical path with a camera
'''

# ! working in constants.u units

from wavefront_correction_lib import *
import numpy as np
import argparse
from pylablib.devices import uc480
from time import time
import fit_stuff as f
import phase_mask_smoothing as pms


def calibrate(args):
    initialize(args)
    best_phase = choose_phase(args)
    phase_mask = np.zeros(get_number_of_subdomains(args.subdomain_size))
    coordinates_list = make_coordinates_list(args)
    print("mainloop start.")
    count = 0
    for i, j in coordinates_list:
        print(f"\rcalibrating subdomain {count + 1}/{len(coordinates_list)}", end="")
        intensity_list = wavefront_correction_loop(i, j, args)
        phase_mask[i, j] = mean_best_phase(intensity_list, best_phase, args)
        count += 1
    produce_phase_mask(phase_mask, args)


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
    H, W = get_number_of_subdomains(args.subdomain_size)
    j0, i0 = read_reference_coordinates(args.reference_coordinates, (H, W))
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
    args.angle = read_angle(args.angle)
    args.phase_list = [i * 2 * np.pi / args.precision for i in range(args.precision)]
    sample_list_2pi = make_sample_holograms_2pi(args.angle, args.phase_list)
    args.samples_list = convert_phase_holograms_to_color_holograms(sample_list_2pi, args.correspond_to2pi)
    rx, ry = read_reference_coordinates(args.reference_coordinates, get_number_of_subdomains(args.subdomain_size))
    args.real_reference_coordinates = (rx * args.subdomain_size, ry * args.subdomain_size)
    black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))
    reference_hologram = add_subdomain(black_hologram, args.samples_list[0], args.real_reference_coordinates, args.subdomain_size)
    print("adjusting exposure time...")
    set_exposure_wrt_reference_img(args.cam, args.window, (256 / 4 - 20, 256 / 4), reference_hologram, args.num_to_avg) # in fully-constructive interference the value of amplitude could be twice as high, therefore intensity four times as high 
    args.intensity_coord = get_intensity_coords(args.cam, args.window, im.fromarray(args.samples_list[0]), args)
    args.hologram = reference_hologram
    args.upper_left_corner = get_corner_coords(args.intensity_coordinates, args.sqrted_number_of_source_pixels, "upper_left")
    args.lower_right_corner = get_corner_coords(args.intensity_coordinates, args.sqrted_number_of_source_pixels)


def wavefront_correction_loop(i, j, args):
    i_real = i * args.subdomain_size
    j_real = j * args.subdomain_size
    k = 0
    nsp = args.sqrted_number_of_source_pixels
    intensity_lists = [[] for _ in range(nsp ** 2)]
    while k < len(args.phase_list):
        args.hologram = add_subdomain(args.hologram, args.samples_list[k], (j_real, i_real), args.subdomain_size)
        display_image_on_external_screen(args.window, args.hologram) # displays hologram on an external dispaly (SLM)
        # intensity = 0
        # for _ in range(args.num_to_avg):
        frame = args.cam.snap()
        relevant_pixels = square_selection(frame, args.upper_left_corner, args.lower_right_corner)
            # intensity = get_intensity_on_coordinates(frame, coords)
        # intensity /= args.num_to_avg
        if max(relevant_pixels) == 255:
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
    parser = argparse.ArgumentParser(description="Script for calibrating an optical path by SLM")

    help_ref_coord = "pseudo coordinates of reference subdomain. use form: x_y, multiply by subdomain_size to find out real coordinates of reference subdomain. maximal allowed coords: (slm_width // ss, slm_height // ss) where ss is subdomain size"

    parser.add_argument('wavefront_correction_name', type=str)
    parser.add_argument('-ss', '--subdomain_size', type=int, default=32)
    parser.add_argument('-p', '--precision', type=int, default=8, help='"color depth" of the phase mask')
    parser.add_argument('-a', '--angle', type=str, default="1_1", help="use form: xdecline_ydecline (angles in constants.u unit)")
    parser.add_argument('-c', '--reference_coordinates', type=str, default="center", help=help_ref_coord)
    # parser.add_argument('-avg', '--num_to_avg', type=int, default=1, help="number of frames to average when measuring intensity")
    parser.add_argument('-ct2pi', '--correspond_to2pi', type=int, default=256, help="value of pixel corresponding to 2pi phase shift")
    parser.add_argument('-skip', '--skip_subdomains_out_of_inscribed_circle', action="store_true", help="subdomains out of the inscribed circle will not be callibrated. use when the SLM is not fully illuminated and the light beam is circular.")
    parser.add_argument('-smooth', '--smooth_phase_mask', action="store_true", help="the phase mask will be smoothed")
    parser.add_argument("-shuffle", action="store_true", help="subdomains will be calibrated in random order")
    parser.add_argument('-ic', "--intensity_coordinates", type=str, default=None, help="coordinates of the point where intensity is measured in form x_y. if not provided, the point will be found automatically.")
    parser.add_argument('-choose_phase', type=str, choices=["trick", "fit"], default="fit", help="method of finding the optimal phase shift")
    parser.add_argument('-resample', type=str, choices=["bilinear", "bicubic"], default="bilinear", help="smoothing method used to upscale the unwrapped phase mask")
    parser.add_argument('-nsp', '--sqrted_number_of_source_pixels', type=int, default=1, help='number of pixel of side of square area on photo from which intensity is taken')

    args = parser.parse_args()
    start = time()
    calibrate(args)
    print("\nexecution_time: ", round((time() - start) / 60, 1),  " min")
