import calibration_lib as cl
from PIL import Image as im
import numpy as np
import argparse
import os


def display_holograms(args):
    window = cl.create_tk_window()
    directory = set_dir(args.directory)
    mask_arr = set_mask(args.mask_name)
    while True:
        name = input("do an action or type 'help' >>")
        if name == "help":
            print_help()
            continue
        if name == "q":
            break
        if name[0:3] == "cd ":
            directory = set_dir(name[3:])
            continue
        if name[0:2] == "cm":
            if len(name) == 2: 
                mask_arr = None
                continue
            mask_arr = set_mask(name[3:-1])
            continue
        path = set_path_to_hologram(directory, name)
        if mask_arr:
            cl.display_image_on_external_screen_img(window, mask_hologram(path, mask_arr))
        else:
            cl.display_image_on_external_screen(window, path)

def set_path_to_hologram(directory, name):
    if not os.path.isfile(f"{directory}/{name}"):
        print("Error: specified hologram does not exist.")
        return
    return f"{directory}/{name}"

def print_help():
    print(f"cd <directory> - change directory - {directory_help}")
    print("<hologram_name> - display hologram in current directory")
    print(f"cm <mask_name> - change mask - {mask_help}")
    print("help - display this message")
    print("q - quit")

def set_mask(mask_name):
    if mask_name is None or mask_name == "" or mask_name == "none":
        return None
    if not os.path.isfile(f"lc-slm/holograms_for_calibration/calibration_phase_masks/{mask_name}"):
        print("Error: specified mask does not exist. setting mask to None")
        mask_arr = None
        return    
    mask_im = im.open(f"lc-slm/holograms_for_calibration/calibration_phase_masks/{args.mask_name}")
    mask_arr = np.array(mask_im)
    return mask_arr

def set_dir(directory):
    if not(os.path.isdir(directory)):
        print("specified directory does not exist. make sure you are in project root and path is correct.")
        return
    print(f"directory changed to {directory}")
    return directory

def mask_hologram(path, mask_arr):
    hologram_im = im.open(path).convert("L")
    hologram_arr = np.array(hologram_im)
    masked_hologram_arr = (hologram_arr + mask_arr) % 256
    masked_hologram_im = im.fromarray(masked_hologram_arr).convert("RGB")
    return masked_hologram_im


# def get_path(mode):
#     if mode == 'p':
#         return ''
#     elif mode == 'h':
#         return "lc-slm/holograms"
#     elif mode == 'c':
#         return "lc-slm/holograms_for_calibration"
#     elif mode == 'i':
#         return "lc-slm/images"
#     else:
#         print("Error: Invalid directory value. Use one of: h, c, i, p. Type \"display_holograms.py --help for help\"") # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="displays selected images from specified directory. Images can be masked with mask of given name")

    # Adding help message for directory argument
    # directory_help = """specifies directory containig images to be displayed:
    #                     'h' for lc-slm/holograms
    #                     'c' for lc-slm/holograms_for_calibration
    #                     'i' for lc-slm/images
    #                     'p' if you wish to specify full path to displayed image each time"""
    
    mask_help = "leave blank or type 'none' for no mask. otherwise type name of the mask. it have to be in lc-slm/holograms_for_calibration/calibration_phase_mask"
    directory_help = "path (from project root) to directory containing images to be displayed"

    parser.add_argument('mask_name', nargs='?', default=None, type=str, help=mask_help)
    parser.add_argument('-d', '--directory', default="lc-slm/holograms/animals_and_stuff", type=str, help=directory_help)

    args = parser.parse_args()
    display_holograms(args)
