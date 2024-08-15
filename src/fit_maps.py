import wavefront_correction as wfc
import numpy as np
import argparse
from time import time
import help_messages_wfc
import os
import fit_stuff as f


def fit_maps(args):
    args.sqrted_number_of_source_pixels = 1  # TODO: make use of this parameter - average the maps, the results will be better
    wfc.initialize(args)
    H, W = wfc.get_number_of_subdomains(args.subdomain_size)
    j0, i0 = (
        (H // 2, W // 2)
        if args.reference_subdomain_coordinates is None
        else args.reference_subdomain_coordinates
    )
    fit_func = f.positive_cos
    param_maps = initiate_param_maps((H, W), fit_func)
    start_loops = time()
    print(
        "mainloop start. estimate of remaining time comes after first row. actual row:"
    )
    for i in range(H):
        if i == 1:
            wfc.print_estimate(H, start_loops)
        print(f"{i + 1}/{H}")
        for j in range(W):
            if i == i0 and j == j0:
                continue
            intensity_list = wfc.wavefront_correction_loop(i, j, args)
            param_dict = f.fit_intensity_general(intensity_list, fit_func)
            fill_maps(param_maps, param_dict, (i, j))
    create_param_maps(param_maps, args)


def fill_maps(param_maps, param_dict, coord):
    for key in param_dict.keys():
        param_maps[key][coord] = param_dict[key]


def create_param_maps(param_maps, args):
    for key in param_maps.keys():
        produce_map(param_maps[key], key, args)


def initiate_param_maps(shape, fit_func):
    H, W = shape
    param_maps = {}
    for key in fit_func.__code__.co_varnames[1:]:
        param_maps[key] = np.zeros((H, W))
    return param_maps


def produce_map(map, type, args):
    specification = type + "_" + make_specification_map(args)
    big_map = wfc.expand_phase_mask(map * args.correspond_to2pi, args.subdomain_size)
    wfc.save_phase_mask(big_map, args.dest_dir, specification)


def make_specification_map(args):
    return f"ss{args.subdomain_size}_spp{args.samples_per_period}_defl_{args.deflect[0]}_{args.deflect[1]}_ref{args.reference_subdomain_coordinates[0]}_{args.reference_subdomain_coordinates[1]}"


if __name__ == "__main__":
    dest_dir = "holograms/fit_maps"
    description = f"""perform wavefront correction procedure,
    intensity values are fitted to a cosine function
    and all four parameters are recorded (not just phase shift).
    Something like the phase mask is created for each parameter
    and saved as .png file in the directory {dest_dir}.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description
    )
    parser.add_argument("wavefront_correction_name", type=str)
    parser.add_argument(
        "-ss",
        "--subdomain_size",
        metavar="INT",
        type=int,
        default=32,
        help=help_messages_wfc.subdomain_size,
    )
    parser.add_argument(
        "-spp",
        "--samples_per_period",
        metavar="INT",
        type=int,
        default=8,
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
        default=1,
        help="number of frames to average when measuring intensity",
    )

    args = parser.parse_args()
    args.dest_dir = dest_dir
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    start = time()
    fit_maps(args)
    print("execution_time: ", round((time() - start) / 60, 1), " min")
