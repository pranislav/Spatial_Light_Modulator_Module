import argparse
import keyboard
import numpy as np
import constants as c
import time
import wavefront_correction_lib as cl
import PIL.Image as im
from scipy.fft import ifft2
import help_messages_wfc

def main(args):
    window = cl.create_tk_window()
    mask = np.load(f"holograms/wavefront_correction_phase_masks/{args.mask_name}")
    black_image = np.zeros((c.slm_height, c.slm_width), dtype=np.uint8)
    height_border = c.slm_height // 2
    width_border = c.slm_width // 2
    coords = [0, 0]
    flags = {"mask": True, "quit": False}
    while True:
        black_image[coords[0]][coords[1]] = 255
        hologram = np.angle(ifft2(black_image)) #np.load(f"{args.holograms_dir}/{coords[0]}/{coords[1]}.npy") #cl.deflect_2pi(coords)
        black_image[coords[0]][coords[1]] = 0
        display_hologram(window, hologram, mask, flags["mask"], args.correspond_to2pi)
        read_keyboard_input(coords, flags, args.big_step, height_border, width_border, args.mirror)
        if flags["quit"]:
            break


def display_hologram(window, hologram, mask, mask_flag, ct2pi):
    if mask_flag:
        hologram = hologram + mask
    hologram_int = (hologram % (2 * np.pi) * ct2pi / (2 * np.pi)).astype(np.uint8)
    hologram_img = im.fromarray(hologram_int)
    cl.display_image_on_external_screen(window, hologram_img)


def read_keyboard_input(coords, mask_flag, big_step, height_border, width_border, mirror):
    while True:
        if keyboard.is_pressed("shift"):
            step = big_step
        else:
            step = 1
        if mirror:
            step = - step
        
        if keyboard.is_pressed("left"):
            coords[1] = (coords[1] - step) % width_border
            return
        if keyboard.is_pressed("right"):
            coords[1] = (coords[1] + step) % width_border
            return
        if keyboard.is_pressed("down"):
            coords[0] = (coords[0] + step) % height_border
            return
        if keyboard.is_pressed("up"):
            coords[0] = (coords[0] - step) % height_border
            return
        if keyboard.is_pressed("m"):
            mask_flag["mask"] = not mask_flag["mask"]
            return
        if keyboard.is_pressed("esc"):
            mask_flag["quit"] = True
            return
        time.sleep(0.1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("mask_name", help="name of the mask file")
    parser.add_argument("-ct2pi", "--correspond_to2pi", type=int, required=True, help=help_messages_wfc.ct2pi)
    parser.add_argument("-bs", "--big_step", type=int, default=20, help="big step size")
    parser.add_argument("-m", "--mirror", action="store_true", help="mirrors left-right and up-down")
    args = parser.parse_args()

    # holograms_dir = "holograms/single_trap_grid_holograms"
    # if not os.path.exists(holograms_dir):
    #     subprocess.run(['python', 'src/make_single_trap_grid_holograms.py', holograms_dir], capture_output=True, text=True, check=True)
    args.holograms_dir = "holograms/single_trap_grid_holograms"
    

    main(args)

