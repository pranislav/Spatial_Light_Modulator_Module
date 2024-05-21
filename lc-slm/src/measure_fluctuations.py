'''script for measuring impact of observed fluctuations of the light on the calibration procedure
motivation: - set_intensity is unstable, probably because of the light fluctuations. need to prove or disprove it
'''


import calibration_lib as cl
import display_holograms as dh
import constants as c
import argparse
import numpy as np
import PIL.Image as im
import time
from pylablib.devices import uc480
from matplotlib import pyplot as plt



def measure_fluctuations(args):
    cam = uc480.UC480Camera()
    window = cl.create_tk_window()
    hologram = create_calibration_hologram(args)
    cl.set_exposure_wrt_reference_img(cam, window, (210, 240), hologram, 1)
    intensity_coords = cl.get_highest_intensity_coordinates_img(cam, window, hologram, 1)
    cl.display_image_on_external_screen_img(window, hologram)
    intensity_evolution = {"time": [], "intensity": []}
    start = time.time()
    while time.time() - start < args.time:
        frame = cam.snap()
        intensity = cl.get_intensity_on_coordinates(frame, intensity_coords)
        intensity_evolution["time"].append(time.time() - start)
        intensity_evolution["intensity"].append(intensity)
    plot_and_save(intensity_evolution, args.time)


def plot_and_save(intensity_evolution, time):
    plt.plot(intensity_evolution["time"], intensity_evolution["intensity"])
    plt.xlabel("time [s]")
    plt.ylabel("intensity [a.u.]")
    plt.title(f"Intensity evolution during {time} seconds")
    time_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"lc-slm/images/fluctuations_{time_name}.png")
    plt.show()


def create_calibration_hologram(args):
    black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width), dtype=np.uint8))
    sample = cl.decline(args.angle, 0, 256)
    reference_subdomain = cl.add_subdomain(black_hologram, sample, args.reference_coordinates, args.subdomain_size)
    second_subdomain = cl.add_subdomain(reference_subdomain, sample, args.subdomain_coordinates, args.subdomain_size)
    return second_subdomain

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for measuring impact of fluctuations of the light on the calibration procedure")
    parser.add_argument('time', type=int, default=5, help="time of the measurement in seconds")
    parser.add_argument('-ss', '--subdomain_size', type=int, default=32)
    parser.add_argument('-a', '--angle', type=str, default="1_1", help="use form: xdecline_ydecline (angles in constants.u unit)")
    parser.add_argument('-c', '--reference_coordinates', type=str, default="16_12", help="pseudo coordinates of reference subdomain. use form: x_y, multiply by subdomain_size to find out real coordinates of reference subdomain. maximal allowed coords: (slm_width // ss, slm_height // ss) where ss is subdomain size")
    parser.add_argument('-sc', '--subdomain_coordinates', type=str, default="16_13", help="pseudo coordinates of second subdomain. use form: x_y, multiply by subdomain_size to find out real coordinates of reference subdomain. maximal allowed coords: (slm_width // ss, slm_height // ss) where ss is subdomain size")