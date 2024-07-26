import argparse
import subprocess
import keyboard
import numpy as np
import os
import constants as c
import time
import wavefront_correction_lib as cl
import PIL.Image as im
import make_single_trap_grid_holograms as mstgh
import algorithms as alg


def main(args):
    window = cl.create_tk_window()
    mask = np.load(f"holograms/wavefront_correction_phase_masks/{args.mask_name}")
    # black_image = np.zeros((c.slm_height, c.slm_width), dtype=np.uint8)
    coords = [c.slm_height // 4, c.slm_width // 4]
    flags = {"mask": True, "quit": False}
    while True:
        # black_image[coords[0]][coords[1]] = 255
        hologram = np.load(f"{args.holograms_dir}/{coords[0]}/{coords[1]}.npy") #cl.decline_2pi(coords)
        # black_image[coords[0]][coords[1]] = 0
        display_hologram(window, hologram, mask, flags["mask"], args.correspond_to2pi)
        read_keyboard_input(coords, flags, args.big_step)
        if flags["quit"]:
            break


def display_hologram(window, hologram, mask, mask_flag, ct2pi):
    if mask_flag:
        hologram = hologram + mask
    hologram_int = (hologram % (2 * np.pi) * ct2pi / (2 * np.pi)).astype(np.uint8)
    hologram_img = im.fromarray(hologram_int)
    cl.display_image_on_external_screen(window, hologram_img)


def read_keyboard_input(coords, mask_flag, big_step):
    while True:
        if keyboard.is_pressed("shift"):
            if keyboard.is_pressed("left"):
                coords[1] += big_step
                return
            if keyboard.is_pressed("right"):
                coords[1] -= big_step
                return
            if keyboard.is_pressed("down"):
                coords[0] -= big_step
                return
            if keyboard.is_pressed("up"):
                coords[0] += big_step
                return
        if keyboard.is_pressed("left"):
            coords[1] += 1
            return
        if keyboard.is_pressed("right"):
            coords[1] -= 1
            return
        if keyboard.is_pressed("down"):
            coords[0] -= 1
            return
        if keyboard.is_pressed("up"):
            coords[0] += 1
            return
        if keyboard.is_pressed("m"):
            mask_flag["mask"] = not mask_flag["mask"]
            return
        if keyboard.is_pressed("esc"):
            mask_flag["quit"] = True
            return
        time.sleep(0.1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_name", help="name of the mask file")
    parser.add_argument("-ct2pi", "--correspond_to2pi", type=int, required=True, help="value of pixel corresponding to 2pi phase shift")
    parser.add_argument("-bs", "--big_step", type=int, default=20, help="big step size")
    args = parser.parse_args()

    # holograms_dir = "holograms/single_trap_grid_holograms"
    # if not os.path.exists(holograms_dir):
    #     subprocess.run(['python', 'src/make_single_trap_grid_holograms.py', holograms_dir], capture_output=True, text=True, check=True)
    args.holograms_dir = "holograms/single_trap_grid_holograms"
    

    main(args)

