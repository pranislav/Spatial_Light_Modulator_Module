import wavefront_correction as wfc
from PIL import Image as im
import numpy as np
import argparse
import os
import explore_wavefront_correction as e
import constants as c
import time
import keyboard
from functools import partial
import help_messages_wfc

mask_dir = "holograms/wavefront_correction_phase_masks"
default_hologram_dir = "holograms"

def display_holograms(args):
    window = wfc.create_tk_window()
    directory = set_dir(args.directory)
    mask_arr = set_mask(args.mask_name)
    name = None
    while True:
        command = input("do an action or type 'help' >> ")
        if command == "help":
            print_help()
            continue
        if command == "q":
            break
        if command[0:5] == "ct2pi":
            args.correspond_to2pi = int(command[5:])
            continue
        if command[0:2] == "cd":
            if len(command) == 2:
                directory = set_dir(default_hologram_dir)
                continue
            maybe_dir = set_dir(command[3:])
            if maybe_dir is not None: directory = maybe_dir
            continue
        if command[0:2] == "cm":
            if len(command) == 2: 
                mask_arr = None
                if name is not None:
                    display_with_mask(window, name, directory, mask_arr, args.correspond_to2pi)
                continue
            maybe_mask_arr = set_mask(command[3:])
            if maybe_mask_arr is not None: mask_arr = maybe_mask_arr # TODO: delete this?
            if name is not None:
                display_with_mask(window, name, directory, mask_arr, args.correspond_to2pi)
            continue
        if command[:2] == 'c':
            print("in mode for displaying instant wavefront_correction holograms")
            display_instant_wavefront_correction_holograms(window)
            continue
        if command[:2] == "s ":
            if " " not in command[2:]:
                dir = command[2:]
                wait = 0
            else:
                dir, wait = command[2:].split(" ")
                wait = float(wait)
            display_holograms_in_sequence(window, dir, wait, mask_arr, args.correspond_to2pi)
            continue
        name = command
        display_with_mask(window, name, directory, mask_arr, args.correspond_to2pi)


def display_holograms_in_sequence(window, dir, wait, mask_arr, ct2pi):
    quit_func = False
    key_wait = 0.3
    micro_wait = 0.01
    frame_micro_wait = min(micro_wait, wait)
    i = -1
    i_max = len(os.listdir(dir)) - 1
    display_with_mask_partial = partial(display_with_mask, window, directory=dir, mask_arr=mask_arr, ct2pi=ct2pi)
    while i < i_max:
        i += 1
        display_with_mask_partial(f"{i}.npy")
        start = time.time()
        while True:
            if keyboard.is_pressed("space"):
                time.sleep(key_wait) # to avoid multiple toggles from a single key press due to key repeat
                i, quit_func = stop_mode(display_with_mask_partial, i, i_max, key_wait, micro_wait, dir)
                break
            if keyboard.is_pressed("left"):
                i = max(i - 2, 0)
                break
            if keyboard.is_pressed("right"):
                break
            if keyboard.is_pressed("esc"):
                quit_func = True
                break
            time.sleep(frame_micro_wait)
            if time.time() - start >= wait:
                break
        if quit_func:
            break

def stop_mode(display_with_mask_partial, i, i_max, key_wait, micro_wait, dir):
    quit_func = False
    while True:
        if keyboard.is_pressed("space"):
            time.sleep(key_wait)
            break
        if keyboard.is_pressed("left"):
            time.sleep(key_wait)
            i = max(i - 1, 0)
            display_with_mask_partial(f"{i}.png")
        if keyboard.is_pressed("right"):
            time.sleep(key_wait)
            i = min(i + 1, i_max)
            display_with_mask_partial(f"{i}.png")
        if keyboard.is_pressed("esc"):
            quit_func = True
            break
        time.sleep(micro_wait)
    return i, quit_func
                  

def display_instant_wavefront_correction_holograms(window):
    params = default_params()
    while True:
        get_params(params)
        sample_hologram_2pi = wfc.deflect_2pi(params["deflect"], params["correspond_to_2pi"])
        sample_hologram = wfc.convert_2pi_hologram_to_int_hologram(sample_hologram_2pi, params["correspond_to_2pi"])
        hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))
        reference_position = e.real_subdomain_position(params["reference_position"], params["subdomain_size"])
        subdomain_position = e.real_subdomain_position(params["subdomain_position"], params["subdomain_size"])
        hologram = wfc.add_subdomain(hologram, sample_hologram, reference_position, params["subdomain_size"])
        if params["phase_shift"] != 0:
            sample_hologram = wfc.deflect(params["deflect"], params["correspond_to_2pi"]) + params["phase_shift"]
        hologram = wfc.add_subdomain(hologram, sample_hologram, subdomain_position, params["subdomain_size"])
        wfc.display_image_on_external_screen(window, hologram)
        command = input("press enter to continue, type 'q' to quit this mode >> ")
        if command == 'q':
            print("leaving mode for displaying instant wavefront_correction holograms")
            break


def default_params():
    params = {}
    params["correspond_to_2pi"] = 256
    params["subdomain_size"] = 32
    params["reference_position"] = (15, 11)
    params["subdomain_position"] = (14, 11)
    params["phase_shift"] = 0
    params["deflect"] = (1, 1)
    return params

def get_params(params):
    print("enter parameters for wavefront_correction")
    print("press enter to leave current value")
    for key in params.keys():
        value = input(f"{key} (current: {params[key]}) >> ")
        if value != '':
            params[key] = eval(value)


def display_with_mask(window, name, directory, mask_arr, ct2pi): 
    path = set_path_to_hologram(directory, name)
    if path is None:
        return 
    if mask_arr is not None:
        wfc.display_image_on_external_screen(window, mask_hologram(path, mask_arr, ct2pi))
        return
    if path[-4:] == ".npy":
        hologram_arr = np.load(path)
        hologram_arr_int = wfc.convert_2pi_hologram_to_int_hologram(hologram_arr, ct2pi)
        hologram_im = im.fromarray(hologram_arr_int).convert("L")
    else:
        hologram_im = im.open(path).convert("L")
    wfc.display_image_on_external_screen(window, hologram_im)
    


def set_path_to_hologram(directory, name):
    if not os.path.isfile(f"{directory}/{name}"):
        print("Error: specified hologram does not exist.")
        return None
    return f"{directory}/{name}"

def print_help():
    print("available commands:")
    print("- ct2pi <value> - change value of pixel corresponding to 2pi phase shift")
    print(f"- cd <directory> - change directory, enter {directory_help}")
    print("- <hologram_name> - display hologram in current directory.")
    print(f"- cm <mask_name> - change mask - {mask_help}")
    print("- c - enter mode for displaying instant wavefront_correction holograms")
    print("- s <directory> <wait_time> - display holograms in sequence from specified directory. wait_time is time between frames in seconds. if not specified, wait_time is 0. press space to pause, left and right to navigate, esc to quit")
    print("- help - display this message")
    print("- q - quit")

def set_mask(mask_name):
    if mask_name is None or mask_name == "":
        return None
    if not os.path.isfile(f"{mask_dir}/{mask_name}"):
        print("Error: specified mask does not exist")
        return None
    if mask_name[-4:] != ".npy":
        print("Error: mask has to be in .npy format")
        return None
    if np.amax(np.load(f"{mask_dir}/{mask_name}")) > 2 * np.pi:
        print("Error: mask values have to be in range [0, 2*pi)")
        return None
    return np.load(f"{mask_dir}/{mask_name}")

def set_dir(directory):
    if not(os.path.isdir(directory)):
        print("specified directory does not exist. make sure you are in project root and path is correct.")
        return None
    print(f"directory changed to {directory}")
    return directory

def mask_hologram(path, mask_arr, ct2pi):
    base, ext = os.path.splitext(path)
    if ext == ".npy":
        hologram_arr_2pi = np.load(path)
        corrected_hologram_arr_2pi = (hologram_arr_2pi + mask_arr) % (2 * np.pi)
        corrected_hologram_arr = corrected_hologram_arr_2pi/(2*np.pi) * ct2pi
    else:
        hologram_im = im.open(path).convert("L")
        hologram_arr = np.array(hologram_im).astype(np.int16)
        corrected_hologram_arr = (hologram_arr + (mask_arr/(2*np.pi) * ct2pi)) % ct2pi
    corrected_hologram_im = im.fromarray(corrected_hologram_arr).convert("L")
    return corrected_hologram_im


if __name__ == "__main__":
    # TODO: it actually does not need parsing now, and add ct2pi to change while run
    description = '''Displays images from specified directory on external screen
    without window porter or taskbar. Images can be masked with mask of given name.
    There is also a mode for displaying instant wavefront_correction holograms
    and a mode for displaying holograms in sequence.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description)

    mask_help = f"Type name of the wavefront correction mask to be added to displayed holograms or leave blank for no mask. Mask has to be in .npy format in {mask_dir}. Mask values has to be in range [0, 2*pi)."
    directory_help = "path (from project root) to directory containing images to be displayed"

    parser.add_argument('-ct2pi', '--correspond_to2pi', metavar="INT", type=int, required=True, help=f"{help_messages_wfc.ct2pi}. you can change this parameter later by typing 'ct2pi <value>' in the console.")
    parser.add_argument('mask_name', nargs='?', default=None, type=str, help=mask_help + " you can change mask later by typing 'cm <mask_name>' in the console.")
    parser.add_argument('-d', '--directory', default=default_hologram_dir, type=str, help=directory_help+" you can change directory later by typing 'cd <directory>' in the console.")

    args = parser.parse_args()
    display_holograms(args)
