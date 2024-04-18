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
    precision = args.precision
    subdomain_size = args.subdomain_size
    cam = uc480.UC480Camera()
    window = create_tk_window()
    H, W = get_number_of_subdomains(subdomain_size)
    reference_coordinates = extract_reference_coordinates(args.coord_ratio, subdomain_size, (H, W))
    sample = make_sample_holograms(args.angle, precision)
    reference_hologram = add_subdomain(im.fromarray(np.zeros((c.slm_height, c.slm_width))), sample[0], reference_coordinates, subdomain_size)
    set_exposure_wrt_reference_img(cam, window, reference_hologram)
    phase_mask = np.zeros((H, W))
    i0, j0 = reference_coordinates
    intensity_coord = get_highest_intensity_coordinates_img(cam, window, reference_hologram) # TODO: rename
    hologram = reference_hologram
    for i in range(H):
        print(f"{i}/{H}")
        i_real = i * subdomain_size
        for j in range(W):
            j_real = j * subdomain_size
            if i_real == i0 and j_real == j0:
                continue
            phase_mask[i, j] = calibration_loop(window, cam, sample, (i_real, j_real), subdomain_size, precision, intensity_coord)
            clear_subdomain(hologram, (i_real, j_real), subdomain_size)
    specification = make_specification(args)
    create_phase_mask(phase_mask, subdomain_size, specification)

def calibration_loop(window, cam, hologram, sample, subdomain_position, subdomain_size, precision, coordinates):
    k = 0
    top_intensity = 0
    opt_value = 0
    while k < precision:
        hologram = add_subdomain(hologram, sample[k], subdomain_position, subdomain_size)
        display_image_on_external_screen_img(window, hologram) # displays hologram on an external dispaly (SLM)
        frame = cam.snap()
        intensity = get_intensity_coordinates(frame, coordinates)
        if intensity > top_intensity:
            top_intensity = intensity
            opt_value = k * 256 // precision
            if intensity == 255:
                print("maximal intensity was reached, adapting...")
                cam.set_exposure(cam.get_exposure() * 0.9) # 10 % decrease of exposure time
                k = 0
                continue
        k += 1
    return opt_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for calibrating an optical path by SLM")

    help_coord_ratio = "use form 'ynumerator_ydenominator_xnumerator_xdenominator'. example: 1_2_3_4 -> y coordinate will be roughly half of slm height, x coordinate will be roughly three quarters of slm width"

    parser.add_argument('calibration_name', type=str)
    parser.add_argument('-ss', '--subdomain_size', type=int, default=32)
    parser.add_argument('-p', '--precision', type=int, default=8, help='"color depth" of the phase mask')
    parser.add_argument('-a', '--angle', type=str, default="1_1", help="use from: xdecline_ydecline (angles in constants.u unit)")
    parser.add_argument('-c', '--coord_ratio', type=str, default="1_2_1_2", help=help_coord_ratio)

    args = parser.parse_args()
    start = time()
    calibrate(args)
    print("execution_time: ", (time() - start) / 60,  " min")
