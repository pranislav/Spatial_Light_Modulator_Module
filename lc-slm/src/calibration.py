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

from calibration_lib import *
import numpy as np
import argparse
from pylablib.devices import uc480
from time import time
import fit_stuff as f
import phase_mask_smoothing as pms


def calibrate(args):
    loop_args = make_loop_args(args) # & set exposure
    fit = lambda x: f.fit_intensity_general(x, f.positive_cos_fixed_wavelength(args.correspond_to2pi))  # TODO other options?
    best_phase = compose_func(return_phase, fit)
    H, W = get_number_of_subdomains(args.subdomain_size)
    j0, i0 = read_reference_coordinates(args.reference_coordinates)
    phase_mask = np.zeros((H, W))
    if args.skip_subdomains_out_of_inscribed_circle:
        skip_subdomain = pms.circular_hole_inclusive((H, W))
    start_loops = time()
    print("mainloop start. estimate of remaining time comes after first row. actual row:")
    for i in range(H):
        if i == 1: print_estimate(H, start_loops)
        print(f"{i + 1}/{H}")
        for j in range(W):
            if i == i0 and j == j0:
                continue
            if skip_subdomain[i, j]:
                continue
            intensity_list = calibration_loop(i, j, loop_args)
            phase_mask[i, j] = best_phase(intensity_list)
    specification = "phase_mask_" + make_specification(args)
    dest_dir = "lc-slm/holograms/holograms_for_calibration/calibration_phase_masks"
    big_phase_mask = expand_phase_mask(phase_mask, args.subdomain_size)
    if args.smooth_phase_mask:
        big_phase_mask = pms.circular_box_blur(big_phase_mask, args.subdomain_size // 2)
    save_phase_mask(big_phase_mask, dest_dir, specification)


def compose_func(func1, func2):
    return lambda x: func1(func2(x))
    
def return_phase(dict):
    return dict["phase_shift"]

def print_estimate(outer_loops_num, start_loops):
    time_elapsed = time() - start_loops
    time_interval = (outer_loops_num - 1) * time_elapsed
    print(f"estimated remaining time is {round(time_interval / 60, 1)} min")

def make_loop_args(args):
    loop_args = {}
    subdomain_size = args.subdomain_size
    cam = uc480.UC480Camera()
    window = create_tk_window()
    print("creating sample holograms...")
    samples_list = make_sample_holograms(args.angle, args.precision, args.correspond_to2pi)
    rx, ry = read_reference_coordinates(args.reference_coordinates)
    real_reference_coordinates = (rx * subdomain_size, ry * subdomain_size)
    black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))
    reference_hologram = add_subdomain(black_hologram, samples_list[0], real_reference_coordinates, subdomain_size)
    print("adjusting exposure time...")
    set_exposure_wrt_reference_img(cam, window, (256 / 4 - 20, 256 / 4), reference_hologram, args.num_to_avg) # in fully-constructive interference the value of amplitude could be twice as high, therefore intensity four times as high 
    # cam.set_exposure(0.001)
    loop_args["intensity_coord"] = get_highest_intensity_coordinates_img(cam, window, reference_hologram, args.num_to_avg)
    loop_args["precision"] = args.precision
    loop_args["subdomain_size"] = subdomain_size
    loop_args["samples_list"] = samples_list
    loop_args["cam"] = cam
    loop_args["window"] = window
    loop_args["hologram"] = reference_hologram
    loop_args["num_to_avg"] = args.num_to_avg
    return loop_args


def calibration_loop(i, j, loop_args):
    subdomain_size = loop_args["subdomain_size"]
    precision = loop_args["precision"]
    cam = loop_args["cam"]
    hologram = loop_args["hologram"] # TODO: find out whether this works
    i_real = i * subdomain_size
    j_real = j * subdomain_size
    k = 0
    intensity_list = [[], []]
    while k < precision:
        loop_args["hologram"] = add_subdomain(loop_args["hologram"], loop_args["samples_list"][k], (j_real, i_real), subdomain_size)
        display_image_on_external_screen_img(loop_args["window"], loop_args["hologram"]) # displays hologram on an external dispaly (SLM)
        intensity = 0
        for _ in range(loop_args["num_to_avg"]):
            frame = cam.snap()
            intensity += get_intensity_coordinates(frame, loop_args["intensity_coord"])
        intensity /= loop_args["num_to_avg"]
        # intensity = get_intensity_coordinates(average_frames(cam, loop_args["num_to_avg"]), loop_args["intensity_coord"])
        if intensity == 255:
            print("maximal intensity was reached, adapting...")
            cam.set_exposure(cam.get_exposure() * 0.9) # 10 % decrease of exposure time
            k = 0
            intensity_list = [[], []]
            continue
        phase = k * 256 // precision
        intensity_list[0].append(phase)
        intensity_list[1].append(intensity)
        k += 1
    clear_subdomain(loop_args["hologram"], (i_real, j_real), subdomain_size)
    return intensity_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for calibrating an optical path by SLM")

    help_ref_coord = "pseudo coordinates of reference subdomain. use form: x_y, multiply by subdomain_size to find out real coordinates of reference subdomain. maximal allowed coords: (slm_width // ss, slm_height // ss) where ss is subdomain size"

    parser.add_argument('calibration_name', type=str)
    parser.add_argument('-ss', '--subdomain_size', type=int, default=32)
    parser.add_argument('-p', '--precision', type=int, default=8, help='"color depth" of the phase mask')
    parser.add_argument('-a', '--angle', type=str, default="1_1", help="use form: xdecline_ydecline (angles in constants.u unit)")
    parser.add_argument('-c', '--reference_coordinates', type=str, default="16_12", help=help_ref_coord)
    parser.add_argument('-avg', '--num_to_avg', type=int, default=1, help="number of frames to average when measuring intensity")
    parser.add_argument('-ct2pi', '--correspond_to2pi', type=int, default=256, help="value of pixel corresponding to 2pi phase shift")
    parser.add_argument('-skip', '--skip_subdomains_out_of_inscribed_circle', action="store_true", help="subdomains out of the inscribed circle will not be callibrated. use when the SLM is not fully illuminated and the light beam is circular.")
    parser.add_argument('-smooth', '--smooth_phase_mask', action="store_true", help="the phase mask will be smoothed")

    args = parser.parse_args()
    start = time()
    calibrate(args)
    print("execution_time: ", round((time() - start) / 60, 1),  " min")
