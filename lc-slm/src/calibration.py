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


def calibrate(args):
    loop_args = make_loop_args(args)
    best_phase = naive
    H, W = get_number_of_subdomains(args.subdomain_size)
    i0, j0 = read_reference_coordinates(args.reference_coordinates)
    phase_mask = np.zeros((H, W))
    for i in range(H):
        print(f"{i}/{H}")
        for j in range(W):
            if i == i0 and j == j0:
                continue
            intensity_list = calibration_loop(i, j, loop_args)
            phase_mask[i, j] = best_phase(intensity_list)
    specification = make_specification(args)
    create_phase_mask(phase_mask, args.subdomain_size, specification)

def make_loop_args(args):
    loop_args = {}
    subdomain_size = args.subdomain_size
    cam = uc480.UC480Camera()
    window = create_tk_window()
    samples_list = make_sample_holograms(args.angle, args.precision)
    rx, ry = read_reference_coordinates(args.reference_coordinates)
    real_reference_coordinates = (rx * subdomain_size, ry * subdomain_size)
    reference_hologram = add_subdomain(im.fromarray(np.zeros((c.slm_height, c.slm_width))), samples_list[0], real_reference_coordinates, subdomain_size)
    set_exposure_wrt_reference_img((256 / 4 - 20, 256 / 4), cam, window, reference_hologram) # in fully-constructive interference the value of amplitude could be twice as high, therefore intensity four times as high 
    loop_args["intensity_coord"] = get_highest_intensity_coordinates_img(cam, window, reference_hologram)
    loop_args["precision"] = args.precision
    loop_args["subdomain_size"] = subdomain_size
    loop_args["samples_list"] = samples_list
    loop_args["cam"] = cam
    loop_args["window"] = window
    loop_args["hologram"] = reference_hologram
    return loop_args


def calibration_loop(i, j, loop_args):
    num_to_avg = 8 # TODO: let this user decide, set just default value
    subdomain_size = loop_args["subdomain_size"]
    precision = loop_args["precision"]
    cam = loop_args["cam"]
    hologram = loop_args["hologram"] # TODO: find out whether this works
    i_real = i * subdomain_size
    j_real = j * subdomain_size
    k = 0
    intensity_list = [[], []]
    while k < precision:
        loop_args["hologram"] = add_subdomain(loop_args["hologram"], loop_args["samples_list"][k], loop_args["subdomain_position"], subdomain_size)
        display_image_on_external_screen_img(loop_args["window"], loop_args["hologram"]) # displays hologram on an external dispaly (SLM)
        intensity = 0
        for _ in range(num_to_avg):
            frame = cam.snap()
            intensity += get_intensity_coordinates(frame,loop_args["intensity_coordinates"])
        intensity /= num_to_avg
        if intensity == 255:
            print("maximal intensity was reached, adapting...")
            cam.set_exposure(cam.get_exposure() * 0.9) # 10 % decrease of exposure time
            k = 0
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
    parser.add_argument('-c', '--reference_coordinates', type=str, default="12, 16", help=help_ref_coord)

    args = parser.parse_args()
    start = time()
    calibrate(args)
    print("execution_time: ", round((time() - start) / 60, 1),  " min")
