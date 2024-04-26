'''script that experimentally determines the relationship between
a value of a pixel in a hologram and real phase shift that SLM applies on the pixel'''

import argparse
import calibration as ca
from calibration_lib import *
from functools import partial
import explore_calibration as e
from scipy.optimize import curve_fit


def main(args):
    loop_args = ca.make_loop_args(args)
    calibration_loop = partial(ca.calibration_loop, loop_args=loop_args)
    fit_params_dict = {"amplitude_shift": [], "amplitude": [], "frequency": [], "phase_shift": []}
    fit = fit_intensity_floorc if args.floor else fit_intensity_generalc
    intensity_lists = []
    H, W = get_number_of_subdomains(args.subdomain_size)
    i0, j0 = read_reference_coordinates(args.reference_coordinates)
    do_loop = partial(circle, (H, W), c.slm_height // 4)
    for i in range(H):
        print(f"{i + 1}/{H}")
        for j in range(W):
            if i == i0 and j == j0:
                continue
            if do_loop(i, j):
                intensity_lists.append(calibration_loop(i, j))
                fit(intensity_lists[-1], fit_params_dict)
    avg_params, std = average_fit_params(fit_params_dict)
    params_printout(avg_params, std)
    if args.fix_amplitude:
        fit = fit_intensity_floorc_amp if args.floor else fit_intensity_generalc_amp
        fit_params_dict = {"amplitude_shift": [], "frequency": [], "phase_shift": []}
        for intensity_list in intensity_lists:
            fit(intensity_list, fit_params_dict)
        avg_params, std = average_fit_params(fit_params_dict)
        print("fitted params with fixed amplitude:")
        params_printout(avg_params, std)



def params_printout(avg_params, std):
    print("average fit parameters:")
    for key in avg_params.keys():
        print(f"{key}: {avg_params[key]} +- {std[key]}")

def average_fit_params(fit_params_dict):
    avg_params = {}
    std = {}
    for key in fit_params_dict.keys():
        avg_params[key] = np.mean(fit_params_dict[key])
        std[key] = np.std(fit_params_dict[key])
    return avg_params, std   


def fit_intensity_generalc(intensity_data, param_dict):
    xdata, ydata = intensity_data
    intensity_range = 256
    phase_range = 256
    supposed_frequency = 2 * np.pi / phase_range
    p0 = [intensity_range/2, intensity_range/2, supposed_frequency, 0]
    lower_bounds = [0, 0, supposed_frequency * 0.6, 0]
    upper_bounds = [intensity_range, intensity_range, supposed_frequency * 1.5, phase_range]
    params, _ = curve_fit(general_cos, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
    param_dict["amplitude_shift"].append(params[0])
    param_dict["amplitude"].append(params[1])
    param_dict["wavelength"].append(params[2])
    param_dict["phase_shift"].append(params[3])

def fit_intensity_generalc_amp(intensity_data, param_dict, amplitude):
    xdata, ydata = intensity_data
    intensity_range = 256
    phase_range = 256
    supposed_frequency = 2 * np.pi / phase_range
    p0 = [intensity_range/2, intensity_range/2, supposed_frequency, 0]
    lower_bounds = [0, 0, supposed_frequency * 0.6, 0]
    upper_bounds = [intensity_range, intensity_range, supposed_frequency * 1.5, phase_range]
    params, _ = curve_fit(partial(general_cos, amplitude=amplitude), xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
    param_dict["amplitude_shift"].append(params[0])
    param_dict["wavelength"].append(params[1])
    param_dict["phase_shift"].append(params[2])

def fit_intensity_floorc(intensity_data, param_dict):
    xdata, ydata = intensity_data
    intensity_range = 256
    phase_range = 256
    supposed_frequency = 2 * np.pi / phase_range
    p0 = [intensity_range/2, supposed_frequency, 0]
    lower_bounds = [0, supposed_frequency * 0.6, 0]
    upper_bounds = [intensity_range, supposed_frequency * 1.5, phase_range]
    params, _ = curve_fit(floor_cos, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
    param_dict["amplitude"].append(params[0])
    param_dict["wavelength"].append(params[1])
    param_dict["phase_shift"].append(params[2])


def fit_intensity_floorc_amp(intensity_data, param_dict, amplitude):
    xdata, ydata = intensity_data
    intensity_range = 256
    phase_range = 256
    supposed_frequency = 2 * np.pi / phase_range
    p0 = [intensity_range/2, supposed_frequency, 0]
    lower_bounds = [0, supposed_frequency * 0.6, 0]
    upper_bounds = [intensity_range, supposed_frequency * 1.5, phase_range]
    params, _ = curve_fit(partial(floor_cos, amplitude=amplitude), xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds))
    param_dict["wavelength"].append(params[0])
    param_dict["phase_shift"].append(params[1])


def general_cos(x, amplitude_shift, amplitude, wavelength, phase_shift):
    return amplitude_shift + amplitude * np.cos((2 * np.pi / wavelength) * (x - phase_shift))

def floor_cos(x, amplitude, wavelength, phase_shift):
    return amplitude * (1 + np.cos((2 * np.pi / wavelength) * (x - phase_shift)))


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for calibrating an optical path by SLM")

    help_ref_coord = "pseudo coordinates of reference subdomain. use form: x_y, multiply by subdomain_size to find out real coordinates of reference subdomain. maximal allowed coords: (slm_width // ss, slm_height // ss) where ss is subdomain size"
    help_fit_func = "a + b*cos(c(x + d)), 'all' fits all params, 'floor' supposes a = b/2, 'uni_amp' performs two runs, first one determines ampliude, second one does , 'uni_amp_floor' "

    parser.add_argument('calibration_name', type=str)
    parser.add_argument('-ss', '--subdomain_size', type=int, default=32)
    parser.add_argument('-p', '--precision', type=int, default=8, help='"color depth" of the phase mask')
    parser.add_argument('-a', '--angle', type=str, default="1_1", help="use form: xdecline_ydecline (angles in constants.u unit)")
    parser.add_argument('-c', '--reference_coordinates', type=str, default="12_16", help=help_ref_coord)
    parser.add_argument('-avg', '--num_to_avg', type=int, default=1, help="number of frames to average when measuring intensity")
    parser.add_argument('-f', '--floor', action='store_true', help="when fitting, it is supposed that minimal intensity is almost zero")
    parser.add_argument('-amp', 'fix_amplitude', action='store_true', help="makes second round of fitting with fixed amplitude (determined in previous round)")

    args = parser.parse_args()
