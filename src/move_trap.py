import argparse
import subprocess
import keyboard
import numpy as np
import os
import constants as c
import time
import wavefront_correction_lib as cl
import PIL.Image as im


def main(args):
    window = None #cl.create_tk_window()
    mask = np.load(f"holograms/wavefront_correction_phase_masks/{args.mask_name}.npy")
    mask = mask * args.correspond_to2pi / (2 * np.pi)
    coords = [c.slm_height // 4, c.slm_width // 4]
    mask_flag = True
    while True:
        hologram = np.load(f"{args.holograms_dir}/{coords[0]}/{coords[1]}.npy")
        display_hologram(window, hologram, mask, mask_flag, args.correspond_to2pi)
        read_keyboard_input(coords, mask_flag)
        if keyboard.is_pressed("q"):
            break


def display_hologram(window, hologram, mask, mask_flag, ct2pi):
    if mask_flag:
        hologram = (hologram + mask) % (2 * np.pi)
    hologram_int = (hologram + np.pi) * ct2pi / (2 * np.pi)
    hologram_img = im.fromarray(hologram_int)
    cl.display_image_on_external_screen(hologram_img, window)


def read_keyboard_input(coords, mask_flag):
    while True:
        if keyboard.is_pressed("right"):
            coords[1] += 1
            return
        if keyboard.is_pressed("left"):
            coords[1] -= 1
            return
        if keyboard.is_pressed("up"):
            coords[0] -= 1
            return
        if keyboard.is_pressed("down"):
            coords[0] += 1
            return
        if keyboard.is_pressed("m"):
            mask_flag = not mask_flag
            return
        if keyboard.is_pressed("shift+right"):
            coords[1] += 10
            return
        if keyboard.is_pressed("shift+left"):
            coords[1] -= 10
            return
        if keyboard.is_pressed("shift+up"):
            coords[0] -= 10
            return
        if keyboard.is_pressed("shift+down"):
            coords[0] += 10
            return
        time.sleep(0.1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_name", help="name of the mask file")
    parser.add_argument("-ct2pi", "--correspond_to2pi", type=int, required=True, help="value of pixel corresponding to 2pi phase shift")
    args = parser.parse_args()

    holograms_dir = "holograms/single_trap_grid_holograms"
    if not os.path.exists(holograms_dir):
        subprocess.run(['python', 'make_single_trap_grid_holograms.py', holograms_dir], capture_output=True, text=True)
    args.holograms_dir = holograms_dir
    

    main(args)

