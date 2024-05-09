from calibration_lib import *
from calibration import *
import numpy as np
import argparse
from pylablib.devices import uc480
from time import time
import color_phase_relation as cp
from scipy.optimize import curve_fit
from color_phase_relation import general_cos


def fit_maps(args):
    loop_args = make_loop_args(args) # & set exposure
    H, W = get_number_of_subdomains(args.subdomain_size)
    j0, i0 = read_reference_coordinates(args.reference_coordinates)
    param_maps = initiate_param_maps((H, W))
    start_loops = time()
    print("mainloop start. estimate of remaining time comes after first row. actual row:")
    for i in range(H):
        if i == 1: print_estimate(H, start_loops)
        print(f"{i + 1}/{H}")
        for j in range(W):
            if i == i0 and j == j0:
                continue
            intensity_list = calibration_loop(i, j, loop_args)
            fit_intensity_generalc_maps(intensity_list, param_maps, (i, j))
    create_param_maps(param_maps, args.subdomain_size)


def create_param_maps(param_maps, subdomain_size):
    dest_dir = "lc-slm/holograms/fit_maps"
    for key in param_maps.keys():
        specification = key + "_" + make_specification(args)
        create_phase_mask(param_maps[key], subdomain_size, specification, dest_dir)


def fit_intensity_generalc_maps(intensity_data, param_maps, coords):
    xdata, ydata = intensity_data
    intensity_range = 256
    phase_range = 256
    supposed_wavelength = phase_range
    p0 = [intensity_range/2, supposed_wavelength, 0, intensity_range/2]
    lower_bounds = [0, supposed_wavelength * 0.6, 0, 0]
    upper_bounds = [intensity_range, supposed_wavelength * 1.5, phase_range, intensity_range]
    try:
        params, _ = curve_fit(general_cos, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
    except:
        print("fit unsuccessful")
        return
    amp_shift, wavelength, phase_shift, amplitude = params
    param_maps["amplitude_shift"][coords] = amp_shift
    param_maps["amplitude"][coords] = amplitude
    param_maps["wavelength"][coords] = wavelength
    param_maps["phase_shift"][coords] = phase_shift
    param_maps["min_val+128"][coords] = amplitude - amp_shift + 128
    param_maps["min_val%256"][coords] = (amplitude - amp_shift) % 256
    param_maps["max_val"][coords] = amplitude + amp_shift
    param_maps["wavelength-128"][coords] = wavelength - 128


def initiate_param_maps(shape):
    H, W = shape
    param_maps = {}
    param_maps["amplitude_shift"] = np.zeros((H, W))
    param_maps["wavelength"] = np.zeros((H, W))
    param_maps["phase_shift"] = np.zeros((H, W))
    param_maps["amplitude"] = np.zeros((H, W))
    param_maps["min_val+128"] = np.zeros((H, W))
    param_maps["min_val%256"] = np.zeros((H, W))
    param_maps["max_val"] = np.zeros((H, W))
    param_maps["wavelength-128"] = np.zeros((H, W))
    return param_maps


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for calibrating an optical path by SLM")

    help_ref_coord = "pseudo coordinates of reference subdomain. use form: x_y, multiply by subdomain_size to find out real coordinates of reference subdomain. maximal allowed coords: (slm_width // ss, slm_height // ss) where ss is subdomain size"

    parser.add_argument('calibration_name', type=str)
    parser.add_argument('-ss', '--subdomain_size', type=int, default=32)
    parser.add_argument('-p', '--precision', type=int, default=8, help='"color depth" of the phase mask')
    parser.add_argument('-a', '--angle', type=str, default="1_1", help="use form: xdecline_ydecline (angles in constants.u unit)")
    parser.add_argument('-c', '--reference_coordinates', type=str, default="16_12", help=help_ref_coord)
    parser.add_argument('-avg', '--num_to_avg', type=int, default=1, help="number of frames to average when measuring intensity")

    args = parser.parse_args()
    start = time()
    fit_maps(args)
    print("execution_time: ", round((time() - start) / 60, 1),  " min")