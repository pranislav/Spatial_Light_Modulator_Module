'''script for measuring impact of observed fluctuations of the light on the wavefront_correction procedure
motivation: - set_intensity is unstable, probably because of the light fluctuations. need to prove or disprove it
'''


import wavefront_correction_lib as cl
import display_holograms as dh
import constants as c
import argparse
import numpy as np
import PIL.Image as im
import time
from pylablib.devices import uc480
from matplotlib import pyplot as plt
import os



def measure_fluctuations(args):
    cam = uc480.UC480Camera()
    window = cl.create_tk_window()
    hologram = create_wavefront_correction_hologram(args)
    if args.exposure is not None:
        cam.set_exposure(args.exposure)
    else:
        cl.set_exposure_wrt_reference_img(cam, window, (210, 240), hologram, 1)
    intensity_coords = cl.get_highest_intensity_coordinates_img(cam, window, hologram, 1)
    cl.display_image_on_external_screen(window, hologram)
    intensity_evolution = {"time": [], "intensity": []}
    start = time.time()
    while time.time() - start < args.time:
        frame = cam.snap()
        intensity = cl.get_intensity_on_coordinates(frame, intensity_coords)
        intensity_evolution["time"].append(time.time() - start)
        intensity_evolution["intensity"].append(intensity)
    expo = cam.get_exposure()
    plot_and_save(intensity_evolution, expo)


def plot_and_save(intensity_evolution, expo):
    plt.plot(intensity_evolution["time"], intensity_evolution["intensity"])
    plt.ylim(0, 270)
    plt.xlabel("time [s]")
    plt.ylabel("intensity [a.u.]")
    expo_round = round(expo, 5)
    plt.title(f"Intensity evolution on static hologram, exposure time: {expo_round} s")
    time_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    dest_dir = "images/measure_fluctuations"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    plt.savefig(f"{dest_dir}/fluctuations_{time_name}.png")


def create_wavefront_correction_hologram(args):
    black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width), dtype=np.uint8))
    sample = cl.decline(args.decline, args.correspond_to2pi)
    reference_coordinates = read_and_expand_coords(args.reference_coordinates, args.subdomain_size)
    subdomain_coordinates = read_and_expand_coords(args.subdomain_coordinates, args.subdomain_size)
    reference_subdomain = cl.add_subdomain(black_hologram, sample, reference_coordinates, args.subdomain_size)
    second_subdomain = cl.add_subdomain(reference_subdomain, sample, subdomain_coordinates, args.subdomain_size)
    return second_subdomain

def read_and_expand_coords(coords, subdomain_size):
    H, W = cl.get_number_of_subdomains(subdomain_size)
    x, y = (H // 2, W // 2) if coords is None else coords
    return int(x) * subdomain_size, int(y) * subdomain_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for measuring impact of fluctuations of the light on the wavefront_correction procedure")
    parser.add_argument('-t', '--time', metavar="SECONDS", type=int, default=5, help="time of the measurement in seconds")
    parser.add_argument('-ss', '--subdomain_size', metavar="INT", type=int, default=32)
    parser.add_argument('-d', '--decline', metavar=("X_ANGLE", "Y_ANGLE"), nargs=2, type=float, default=(0.5, 0.5), help="angle to decline the light in x and y direction (in constants.u unit)")
    parser.add_argument('-c', '--reference_coordinates', metavar=("X_COORD", "y_COORD"), nargs=2, type=int, default=None, help="subdomain-scale coordinates of reference subdomain. use form: x_y, multiply by subdomain_size to find out real coordinates of reference subdomain. maximal allowed coords: (slm_width // ss, slm_height // ss) where ss is subdomain size. Default parameter assigns the reference subdomain to the middle one.")
    parser.add_argument('-sc', '--subdomain_coordinates', metavar=("X_COORD", "y_COORD"), nargs=2, type=int, default=(0, 0), help="coordinates of second subdomain")
    parser.add_argument('-e', '--exposure', metavar="SECONDS", type=float, default=None, help="exposure time in seconds")
    parser.add_argument('-ct2pi', '--correspond_to2pi', metavar="INT", type=int, requird=True, help="value of pixel corresponding to 2pi phase shift")
    args = parser.parse_args()
    measure_fluctuations(args)
