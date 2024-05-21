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
import fit_stuff as f
import time
import os



def main(args):
    fit_func = f.positive_cos_floor() if args.floor else f.positive_cos
    cam = uc480.UC480Camera()
    window = cl.create_tk_window()
    # subdomain_size = int(np.sqrt(1 / 5) * c.slm_height)
    sample_list = e.make_sample_holograms((1, 1), args.precision, args.ct2pi)
    upper_left_corner = np.array((c.slm_width // 2 - args.subdomain_size, (c.slm_height - args.subdomain_size) // 2))
    black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))
    reference = cl.add_subdomain(black_hologram, sample_list[0], upper_left_corner, args.subdomain_size)
    hologram_set = make_hologram_set(reference, sample_list, upper_left_corner + (args.subdomain_size, 0), args.subdomain_size)
    cl.set_exposure_wrt_reference_img(cam, window, (220, 240), hologram_set[0], 8)
    intensity_coords = cl.get_highest_intensity_coordinates_img(cam, window, hologram_set[0], 8)
    fit_params_dict = {param: [] for param in fit_func.__code__.co_varnames[1:]}
    time_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    for i in range(args.runs):
        intensity_list = two_big_loop(args.precision, cam, window, hologram_set, intensity_coords)
        param_dict = f.fit_intensity_general(intensity_list, fit_func)
        cp.fill_fit_params_dict(fit_params_dict, param_dict)
        make_plot(intensity_list, fit_params_dict, i, time_name)
        print(f"run {i + 1}/{args.runs}")
        sleep(args.wait)
    avg_params, std = cp.average_fit_params(fit_params_dict)
    cp.params_printout(avg_params, std)


def make_plot(intensity_list, fit_params_dict, time_name, i):
    intensity_fit = e.plot_fit(fit_params_dict)
    plot_image = e.create_plot_img(intensity_list, intensity_fit, (500, 200), 0, 0)
    dest_dir = "lc-slm/images/fit_2big"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    plot_image.save(f"{dest_dir}/{time_name}_{i}.png")


def two_big_loop(precision, cam, window, hologram_set, intensity_coords, fit_params_dict, fit_fun):
    intensity_list = [[], []]
    k = 0
    while k < precision:
        cl.display_image_on_external_screen_img(window, hologram_set[k])
        frame = cam.snap()
        intensity = cl.get_intensity_on_coordinates(frame, intensity_coords)
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
    return intensity_list



def make_hologram_set(reference, sample_list, coords, subdomain_size):
    hologram_set = []
    for sample in sample_list:
        hologram_set.append(cl.add_subdomain(reference, sample, coords, subdomain_size))
    return hologram_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--precision", type=int, default=32, help="number of different phase values to be tested")
    parser.add_argument("-ss", "--subdomain_size", type=int, default=64, help="subdomain size")
    parser.add_argument('-ct2pi', '--correspond_to2pi', type=int, default=256, help="value of pixel corresponding to 2pi phase shift")
    parser.add_argument("-r", "--runs", type=int, default=8, help="number of runs to average the results")
    parser.add_argument("-f", "--floor", action="store_true", help="presume that minimal intensity is almost zero")
    parser.add_argument("-w", "--wait", type=float, default=0.5, help="time to wait between runs")
    args = parser.parse_args()
    main(args)
