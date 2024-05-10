'''script that experimentally determines the relationship between
a value of a pixel in a hologram and real phase shift that SLM applies on the pixel'''

import argparse
import calibration as ca
from calibration_lib import *
from functools import partial
import fit_stuff as fs
import time


def main(args):
    loop_args = ca.make_loop_args(args)
    calibration_loop = partial(ca.calibration_loop, loop_args=loop_args)
    fit_func = fs.positive_cos_floor() if args.floor else fs.positive_cos
    fit_params_dict = {param: [] for param in fit_func.__code__.co_varnames[1:]}
    intensity_lists = []
    H, W = get_number_of_subdomains(args.subdomain_size)
    j0, i0 = read_reference_coordinates(args.reference_coordinates)
    do_loop = partial(circle, (H, W), H // 4)
    for i in range(H):
        print(f"{i + 1}/{H}")
        for j in range(W):
            if i == i0 and j == j0:
                continue
            if do_loop(i, j):
                intensity_lists.append(calibration_loop(i, j))
                try: # TODO: catch just the exception that is thrown when fitting is unsuccessful
                    param_dict = fs.fit_intensity_general(intensity_lists[-1], fit_func)
                except:
                    print("fit unsuccessful")
                    continue
                fill_fit_params_dict(fit_params_dict, param_dict)
    avg_params, std = average_fit_params(fit_params_dict)
    file_name = "lc-slm/fit_params.txt"
    print_info(args, file_name)
    params_printout(avg_params, std, file_name)
    if args.fix_amplitude:
        fit_func = fs.positive_cos_floor_fixed_amp(avg_params["amplitude"]) if args.floor else fs.positive_cos_fixed_amp(avg_params["amplitude"])
        fit_params_dict = {param: [] for param in fit_func.__code__.co_varnames[1:]}
        for intensity_list in intensity_lists:
            param_dict = fs.fit_intensity_general(intensity_list, fit_func)
            fill_fit_params_dict(fit_params_dict, param_dict)
        avg_params, std = average_fit_params(fit_params_dict)
        with open(file_name, "a") as f:
            f.write("fit with fixed amplitude\n")
        params_printout(avg_params, std, file_name)


def print_info(args, file_name):
    with open(file_name, "a") as f:
        f.write(time.strftime("%Y-%m-%d_%H-%M-%S") + "\n")
        for key in args.__dict__.keys():
            f.write(f"{key}: {args.__dict__[key]}\n")
        f.write("\n")

def params_printout(avg_params, std, file_name):
    with open(file_name, "a") as f:
        f.write("average fit parameters:\n")
        for key in avg_params.keys():
            f.write(f"{key}: {avg_params[key]} +- {std[key]}\n")
        f.write("\n")

def average_fit_params(fit_params_dict):
    avg_params = {}
    std = {}
    for key in fit_params_dict.keys():
        if fit_params_dict[key] == []: continue
        avg_params[key] = np.mean(fit_params_dict[key])
        std[key] = np.std(fit_params_dict[key])
    return avg_params, std   

# -------------------


def fill_fit_params_dict(dct, params):
    for key in params.keys():
        dct[key].append(params[key])


def circle(dimensions, radius, i, j):
    H, W = dimensions
    i0 = H // 2
    j0 = W // 2
    if (i - i0)**2 + (j - j0)**2 < radius**2:
        return True
    return False

# def do_loop_coords_special(i, j):
#     return j == 3 and (i ==2 or i == 3) or j == 4 and (i == 2 or i == 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for calibrating an optical path by SLM")

    help_ref_coord = "pseudo coordinates of reference subdomain. use form: x_y, multiply by subdomain_size to find out real coordinates of reference subdomain. maximal allowed coords: (slm_width // ss, slm_height // ss) where ss is subdomain size"

    parser.add_argument('-ss', '--subdomain_size', type=int, default=64)
    parser.add_argument('-p', '--precision', type=int, default=16, help="number of phase shifts")
    parser.add_argument('-a', '--angle', type=str, default="1_1", help="use form: xdecline_ydecline (angles in constants.u unit)")
    parser.add_argument('-c', '--reference_coordinates', type=str, default="8_6", help=help_ref_coord)
    parser.add_argument('-avg', '--num_to_avg', type=int, default=8, help="number of frames to average when measuring intensity")
    parser.add_argument('-f', '--floor', action='store_true', help="when fitting, it is supposed that minimal intensity is almost zero")
    parser.add_argument('-amp', '--fix_amplitude', action='store_true', help="makes second round of fitting with fixed amplitude (determined in previous round)")
    parser.add_argument('-ct2pi', '--correspond_to2pi', type=int, default=256, help="value of pixel corresponding to 2pi phase shift")
    args = parser.parse_args()

    main(args)
