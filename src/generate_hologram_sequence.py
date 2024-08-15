import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image as im
from algorithms import gerchberg_saxton
import numpy as np
import help_messages_wfc


def generate_hologram_sequence(args):
    dest_dir_holograms = f"holograms/{args.source_dir}_{args.version}_holograms"
    dest_dir_preview = f"images/moving_traps/{args.source_dir}_{args.version}_preview"
    for dest_dir in [dest_dir_holograms, dest_dir_preview]:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
    source_dir_path = f"images/moving_traps/{args.source_dir}"
    error_evolution_list = []
    files = os.listdir(source_dir_path)
    for i in range(len(files)):
        print(f"\rcreating {i}. hologram ", end="")
        demanded_output = np.array(im.open(f"{source_dir_path}/{i}.png"))
        hologram, expected_target, error_evolution = gerchberg_saxton(
            demanded_output, args
        )
        error_evolution_list.append(error_evolution)
        np.save(f"{dest_dir_holograms}/{i}.npy", hologram)
        # im.fromarray(hologram).convert("L").save(f"{dest_dir_holograms}/{i}.png")
        if args.preview:
            im.fromarray(expected_target).convert("L").save(
                f"{dest_dir_preview}/{i}.png"
            )
    plot_error_evolution(error_evolution_list)


def plot_error_evolution(err_evl_list):
    """plots error evolution for each trap into one plot"""
    for i, err_evl in enumerate(err_evl_list):
        plt.plot(err_evl, label=i)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    description = """Transform a sequence of images of traps into
    a sequence of holograms corresponding to those traps.
    The directory with images of traps has to be in images/moving_traps directory.
    Hologram sequence is saved in holograms/<dest_dir>_<version>_holograms.
    If preview is set, also directory with expected images resulting from the holograms is created
    in images/moving_traps/<source_dir>_<version>_preview.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description
    )
    parser.add_argument(
        "source_dir",
        type=str,
        help="name of the directory with images of traps (has to be in images/moving_traps directory)",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="string added to the name of the directory with holograms and preview images to distinguish between different versions of the same sequence of traps",
    )
    parser.add_argument(
        "-ii",
        "--incomming_intensity",
        metavar="PATH",
        type=str,
        default="uniform",
        help="path to the incomming intensity image from images directory or 'uniform' for uniform intensity",
    )
    parser.add_argument(
        "-ct2pi",
        "--correspond_to2pi",
        metavar="INT",
        required=True,
        type=int,
        help=help_messages_wfc.ct2pi,
    )
    parser.add_argument(
        "-tol",
        "--tolerance",
        metavar="FLOAT",
        default=0,
        type=float,
        help="algorithm stops when error descends under tolerance",
    )
    parser.add_argument(
        "-loops",
        "--max_loops",
        metavar="INT",
        default=5,
        type=int,
        help="algorithm performs no more than max_loops loops no matter what error it is",
    )
    parser.add_argument(
        "-p",
        "--preview",
        action="store_true",
        help="creates also directory with expected images resulting from the holograms",
    )
    args = parser.parse_args()
    args.gif = False
    args.plot_error = False
    args.print_info = False

    generate_hologram_sequence(args)
