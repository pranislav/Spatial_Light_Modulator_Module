from wavefront_correction_lib import *
from wavefront_correction import *
import numpy as np
import argparse
from time import time


def fit_maps(args):
    args.sqrted_number_of_source_pixels = 1 # TODO: make use of this parameter - average the maps, the results will be better
    initialize(args)
    H, W = get_number_of_subdomains(args.subdomain_size)
    j0, i0 = (H // 2, W // 2) if args.reference_coordinates is None else args.reference_coordinates
    fit_func = f.positive_cos
    param_maps = initiate_param_maps((H, W), fit_func)
    start_loops = time()
    print("mainloop start. estimate of remaining time comes after first row. actual row:")
    for i in range(H):
        if i == 1: print_estimate(H, start_loops)
        print(f"{i + 1}/{H}")
        for j in range(W):
            if i == i0 and j == j0:
                continue
            intensity_list = wavefront_correction_loop(i, j, args)
            param_dict = f.fit_intensity_general(intensity_list, fit_func)
            fill_maps(param_maps, param_dict, (i, j))
    create_param_maps(param_maps, args)


def fill_maps(param_maps, param_dict, coord):
    for key in param_dict.keys():
        param_maps[key][coord] = param_dict[key]

def create_param_maps(param_maps, args):
    for key in param_maps.keys():
        produce_phase_mask_single(param_maps[key], key, args)


def initiate_param_maps(shape, fit_func):
    H, W = shape
    param_maps = {}
    for key in fit_func.__code__.co_varnames[1:]:
        param_maps[key] = np.zeros((H, W))
    return param_maps


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for calibrating an optical path by SLM")

    help_ref_coord = "pseudo coordinates of reference subdomain. use form: x_y, multiply by subdomain_size to find out real coordinates of reference subdomain. maximal allowed coords: (slm_width // ss, slm_height // ss) where ss is subdomain size"

    parser.add_argument('wavefront_correction_name', type=str)
    parser.add_argument('-ss', '--subdomain_size', metavar="INT", type=int, default=32)
    parser.add_argument('-spp', '--samples_per_period', metavar="INT", type=int, default=8, help='"color depth" of the phase mask')
    parser.add_argument('-d', '--decline', metavar=("X_ANGLE", "Y_ANGLE"), nargs=2, type=float, default=(0.5, 0.5), help="angle to decline the light in x and y direction (in constants.u unit)")
    parser.add_argument('-c', '--reference_coordinates', metavar=("X_COORD", "Y_COORD"), nargs=2, type=int, default=None, help="subdomain-scale coordinates of reference subdomain. use form: x_y, multiply by subdomain_size to find out real coordinates of reference subdomain. maximal allowed coords: (slm_width // ss, slm_height // ss) where ss is subdomain size. Default parameter assigns the reference subdomain to the middle one.")
    parser.add_argument('-avg', '--num_to_avg', metavar="INT", type=int, default=1, help="number of frames to average when measuring intensity")

    args = parser.parse_args()
    args.dest_dir = "holograms/fit_maps"
    start = time()
    fit_maps(args)
    print("execution_time: ", round((time() - start) / 60, 1),  " min")