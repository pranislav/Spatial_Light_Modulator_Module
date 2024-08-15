import argparse
import wavefront_correction as wfc
from functools import partial
import fit_stuff as fs
import time
import os
import help_messages_wfc
import numpy as np


def main(args):
    args.sqrted_number_of_source_pixels = 1
    wfc.initialize(args)
    wavefront_correction_loop = partial(wfc.wavefront_correction_loop, args=args)
    fit_func = fs.positive_cos_floor() if args.floor else fs.positive_cos
    fit_params_dict = {param: [] for param in fit_func.__code__.co_varnames[1:]}
    intensity_lists = []
    H, W = wfc.get_number_of_subdomains(args.subdomain_size)
    j0, i0 = (
        (H // 2, W // 2)
        if args.reference_subdomain_coordinates is None
        else args.reference_subdomain_coordinates
    )
    do_loop = partial(circle, (H, W), H // 4)
    for i in range(H):
        print(f"{i + 1}/{H}")
        for j in range(W):
            if i == i0 and j == j0:
                continue
            if do_loop(i, j):
                intensity_lists.append(wavefront_correction_loop(i, j))
                try:  # TODO: catch just the exception that is thrown when fitting is unsuccessful
                    param_dict = fs.fit_intensity_general(intensity_lists[-1], fit_func)
                except:
                    print("fit unsuccessful")
                    continue
                fill_fit_params_dict(fit_params_dict, param_dict)
    avg_params, std = average_fit_params(fit_params_dict)
    if not os.path.exists("documents"):
        os.makedirs("documents")
    file_name = "documents/fit_params.txt"
    print_info(args, file_name)
    params_printout(avg_params, std, file_name)
    if args.fix_amplitude:
        fit_func = (
            fs.positive_cos_floor_fixed_amp(avg_params["amplitude"])
            if args.floor
            else fs.positive_cos_fixed_amp(avg_params["amplitude"])
        )
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
        if fit_params_dict[key] == []:
            continue
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
    if (i - i0) ** 2 + (j - j0) ** 2 < radius**2:
        return True
    return False



if __name__ == "__main__":
    description = """Experimentally determine the relationship between
    a value of a pixel in a hologram and real phase shift that SLM applies on the pixel
    (grayscale-phase response) using wavefront correction procedure.
    The program fits the intensity values of the pixels to a cosine function
    and then records and averages the parameters of the function.
    The values are printed to a file in the documents directory.
    The wavelength of the function corresponds to the correspond_to2pi value.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description
    )
    parser.add_argument(
        "-ss",
        "--subdomain_size",
        metavar="INT",
        type=int,
        default=64,
        help=help_messages_wfc.subdomain_size,
    )
    parser.add_argument(
        "-spp",
        "--samples_per_period",
        metavar="INT",
        type=int,
        default=16,
        help=help_messages_wfc.samples_per_period,
    )
    parser.add_argument(
        "-d",
        "--deflect",
        metavar=("X_ANGLE", "Y_ANGLE"),
        nargs=2,
        type=float,
        default=(0.5, 0.5),
        help=help_messages_wfc.deflect,
    )
    parser.add_argument(
        "-c",
        "--reference_subdomain_coordinates",
        metavar=("X_COORD", "Y_COORD"),
        nargs=2,
        type=int,
        default=None,
        help=help_messages_wfc.reference_subdomain_coordinates,
    )
    parser.add_argument(
        "-avg",
        "--num_to_avg",
        metavar="INT",
        type=int,
        default=8,
        help="number of frames to average when measuring intensity",
    )  # TODO will this be used?
    parser.add_argument(
        "-f",
        "--floor",
        action="store_true",
        help="when fitting, it is supposed that minimal intensity is zero",
    )
    parser.add_argument(
        "-amp",
        "--fix_amplitude",
        action="store_true",
        help="makes second round of fitting with fixed amplitude (determined in previous round)",
    )
    args = parser.parse_args()

    main(args)
