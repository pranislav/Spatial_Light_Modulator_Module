import argparse
import os
import algorithms as alg
import constants as c
import numpy as np


def fill_GS_arguments(GS_args):
    GS_args.incomming_intensity = "uniform"
    GS_args.tolerance = 0.0
    GS_args.max_loops = 2
    GS_args.gif = False
    GS_args.print_info = False
    GS_args.plot_error = False
    GS_args.print_progress = False


def grid(height, width):
    return [(i, j) for i in range(height) for j in range(width)]

def main(args):
    os.makedirs(args.save_address, exist_ok=True)
    GS_args = argparse.Namespace()
    fill_GS_arguments(GS_args)
    black_image = np.zeros((c.slm_height, c.slm_width), dtype=np.uint8)
    print("Creating holograms")
    holograms_number = (c.slm_height // 2) * (c.slm_width // 2)
    n = 0
    for i in range(c.slm_height // 2):
        os.makedirs(args.save_address + f"/{i}", exist_ok=True)
        for j in range(c.slm_width // 2):
            n += 1
            name = args.save_address + f"/{i}/{j}.npy"
            if os.path.exists(name):
                continue
            print(f"\r{n}/{holograms_number}", end='')
            black_image[i, j] = 255
            hologram, _, _ = alg.GS(black_image, GS_args)
            black_image[i, j] = 0
            np.save(name, hologram)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_address", help="address to save the holograms")
    args = parser.parse_args()
    main(args)