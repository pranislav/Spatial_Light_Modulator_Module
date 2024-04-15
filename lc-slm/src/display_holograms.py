import calibration_lib as cl
from PIL import Image as im
import numpy as np
import argparse
import os


def display_holograms(args):
    window = cl.create_tk_window()
    directory = get_path(args.directory)
    mask_name = args.mask_name
    if args.mask_name:
        mask_im = im.open(f"lc-slm/holograms_for_calibration/calibration_phase_masks/{args.mask_name}")
        mask_arr = np.array(mask_im)
    while True:
        name = input("display hologram by typing its name, change directory with cd, change mask with cm >> ")
        if name == "q": break
        if name[0:3] == "cd ":
            new_dir = name[3:-1]
            if not(os.path.isdir(new_dir)):
                print("specified directory does not exist")
                continue
            directory = new_dir
            print(f"directory changed to {new_dir}")
            continue
        if name[0:3] == "cm ":
            mask_name = name[3:-1]
            continue
        path = f"{directory}/{name}"
        if mask_name:
            cl.display_image_on_external_screen_img(window, mask_hologram(path, mask_arr))
        else:
            cl.display_image_on_external_screen(window, path)


def mask_hologram(path, mask_arr):
    hologram_im = im.open(path).convert("L")
    hologram_arr = np.array(hologram_im)
    masked_hologram_arr = (hologram_arr + mask_arr) % 256
    masked_hologram_im = im.fromarray(masked_hologram_arr).convert("RGB")
    return masked_hologram_im


def get_path(mode):
    if mode == 'p':
        return ''
    elif mode == 'h':
        return "lc-slm/holograms"
    elif mode == 'c':
        return "lc-slm/holograms_for_calibration"
    elif mode == 'i':
        return "lc-slm/images"
    else:
        print("Error: Invalid directory value. Use one of: h, c, i, p. Type \"display_holograms.py --help for help\"") # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="displays selected images from specified directory. Images can be masked with mask on given path")

    # Adding help message for directory argument
    directory_help = """specifies directory containig images to be displayed:
                        'h' for lc-slm/holograms
                        'c' for lc-slm/holograms_for_calibration
                        'i' for lc-slm/images
                        'p' if you wish to specify full path to displayed image each time"""

    parser.add_argument('mask_name', nargs='?', default=None, type=str, help="just name of the mask. it have to be in holograms_for_calibration/calibration_phase_mask")
    parser.add_argument('-d', '--directory', choices=['h', 'c', 'i', 'p'], default='h', help=directory_help)

    args = parser.parse_args()
    display_holograms(args)
