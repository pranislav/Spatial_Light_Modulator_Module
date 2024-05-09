'''script that experimentally determines the relationship between
a value of a pixel in a hologram and real phase shift that SLM applies on the pixel
in contrast to the other one, here there are just two subdomains placed in a way
that intensity is equally divided between them
subdomains are significantly big'''


import explore_calibration as e
import calibration_lib as cl
import color_phase_relation as cp
import constants as c
import numpy as np
import argparse
from pylablib.devices import uc480
from time import sleep
from PIL import Image as im



def main(precision, runs, wait, floor):
    fit = cp.fit_intensity_floorc if floor else cp.fit_intensity_generalc
    cam = uc480.UC480Camera()
    window = cl.create_tk_window()
    subdomain_size = int(np.sqrt(1 / 5) * c.slm_height)
    sample_list = e.make_sample_holograms((1, 1), precision)
    upper_left_corner = np.array((c.slm_width // 2 - subdomain_size, (c.slm_height - subdomain_size) // 2))
    black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))
    reference = cl.add_subdomain(black_hologram, sample_list[0], upper_left_corner, subdomain_size)
    hologram_set = make_hologram_set(reference, sample_list, upper_left_corner + (subdomain_size, 0), subdomain_size)
    cl.set_exposure_wrt_reference_img(cam, window, (220, 240), hologram_set[0], 8)
    intensity_coords = cl.get_highest_intensity_coordinates_img(cam, window, hologram_set[0], 8)
    fit_params_dict = {"amplitude_shift": [], "amplitude": [], "wavelength": [], "phase_shift": []}
    for _ in range(runs):
        two_big_loop(precision, cam, window, hologram_set, intensity_coords, fit_params_dict, fit)
        sleep(wait)
    avg_params, std = cp.average_fit_params(fit_params_dict)
    cp.params_printout(avg_params, std)



def two_big_loop(precision, cam, window, hologram_set, intensity_coords, fit_params_dict, fit):
    intensity_list = [[], []]
    k = 0
    while k < precision:
        cl.display_image_on_external_screen_img(window, hologram_set[k])
        frame = cam.snap()
        intensity = cl.get_intensity_coordinates(frame, intensity_coords)
        if intensity == 255:
            print("maximal intensity was reached, adapting...")
            cam.set_exposure(cam.get_exposure() * 0.9) # 10 % decrease of exposure time
            k = 0
            intensity_list = [[], []]
            continue
        phase = k * 256 // precision
        intensity_list[0].append(phase)
        intensity_list[1].append(intensity)
    fit(intensity_list, fit_params_dict)



def make_hologram_set(reference, sample_list, coords, subdomain_size):
    hologram_set = []
    for sample in sample_list:
        hologram_set.append(cl.add_subdomain(reference, sample, coords, subdomain_size))
    return hologram_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--precision", type=int, default=32, help="number of different phase values to be tested")
    parser.add_argument("-r", "--runs", type=int, default=8, help="number of runs to average the results")
    parser.add_argument("-f", "--floor", action="store_true", help="presume that minimal intensity is almost zero")
    parser.add_argument("-w", "--wait", type=float, default=0.5, help="time to wait between runs")
    args = parser.parse_args()
    main(args.precision, args.runs, args.wait, args.floor)
