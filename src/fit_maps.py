from wavefront_correction_lib import *
from wavefront_correction import *
import numpy as np
import argparse
from time import time


def fit_maps(args):
    loop_args = make_loop_args(args) # & set exposure
    H, W = get_number_of_subdomains(args.subdomain_size)
    j0, i0 = read_reference_coordinates(args.reference_coordinates)
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
            intensity_list = wavefront_correction_loop(i, j, loop_args)
            param_dict = f.fit_intensity_general(intensity_list, fit_func)
            fill_maps(param_maps, param_dict, (i, j))
    create_param_maps(param_maps, args.subdomain_size)


def fill_maps(param_maps, param_dict, coord): # if error, check if coord is in the right order
    for key in param_dict.keys():
        param_maps[key][coord] = param_dict[key]

def create_param_maps(param_maps, subdomain_size):
    dest_dir = "holograms/fit_maps"
    for key in param_maps.keys():
        specification = key + "_" + make_specification(args)
        create_phase_mask(param_maps[key], subdomain_size, specification, dest_dir)


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
    parser.add_argument('-ss', '--subdomain_size', type=int, default=32)
    parser.add_argument('-spp', '--samples_per_period', type=int, default=8, help='"color depth" of the phase mask')
    parser.add_argument('-a', '--angle', type=str, default="1_1", help="use form: xdecline_ydecline (angles in constants.u unit)")
    parser.add_argument('-c', '--reference_coordinates', type=str, default="16_12", help=help_ref_coord)
    parser.add_argument('-avg', '--num_to_avg', type=int, default=1, help="number of frames to average when measuring intensity")

    args = parser.parse_args()
    start = time()
    fit_maps(args)
    print("execution_time: ", round((time() - start) / 60, 1),  " min")