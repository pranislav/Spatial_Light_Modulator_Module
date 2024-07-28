import argparse
import keyboard
import numpy as np
import constants as c
import time
import wavefront_correction_lib as cl
import PIL.Image as im
from scipy.fft import ifft2
import threading

def main(args):
    window = cl.create_tk_window()
    mask = np.load(f"holograms/wavefront_correction_phase_masks/{args.mask_name}")
    black_image = np.zeros((c.slm_height, c.slm_width), dtype=np.uint8)
    height_border = c.slm_height // 2
    width_border = c.slm_width // 2
    coords = [[0, 0], [0, 0]]
    flags = {"mask": True, "quit": False, "which": 0, "split": False, "key_change": False, "last_key": "", "is_pressed": False, "shift": False}
    read_key_thread = threading.Thread(target=last_pressed_key, args=(flags,), daemon=True)
    read_key_thread.start()
    holograms = [update_hologram(black_image, coords, 0), update_hologram(black_image, coords, 1)]
    i = 0
    while True:
        while True:
            display_hologram(window, holograms[i%2], mask, flags["mask"], args.correspond_to2pi)
            if not flags["split"]:
                wait_for_key(flags)
                break
            i += 1
            if flags["is_pressed"] or flags["key_change"]:
                break
            time.sleep(0.05)
        change_coords(coords, flags, args.big_step, height_border, width_border, args.mirror)
        if flags["quit"]:
            break
        holograms[flags["which"]] = update_hologram(black_image, coords, i)


def update_hologram(black_image, coords, i):
    black_image[coords[i%2][0]][coords[i%2][1]] = 255
    hologram = np.angle(ifft2(black_image))
    black_image[coords[i%2][0]][coords[i%2][1]] = 0
    return hologram


def wait_for_key(flags):
    while not (flags["is_pressed"] or flags["key_change"]):
        return
    time.sleep(0.1)

def change_coords(coords, flags, big_step, height_border, width_border, mirror):
    if not flags["key_change"] or not flags["is_pressed"]:
        return
    flags["key_change"] = False

    if flags["shift"]:
        step = big_step
    else:
        step = 1
    if mirror:
        step = - step
    
    if flags["last_key"] == "left":
        coords[flags["which"]][1] = (coords[flags["which"]][1] - step) % width_border
        return
    if flags["last_key"] == "right":
        coords[flags["which"]][1] = (coords[flags["which"]][1] + step) % width_border
        return
    if flags["last_key"] == "down":
        coords[flags["which"]][0] = (coords[flags["which"]][0] + step) % height_border
        return
    if flags["last_key"] == "up":
        coords[flags["which"]][0] = (coords[flags["which"]][0] - step) % height_border
        return
    if flags["last_key"] == "m":
        flags["mask"] = not flags["mask"]
        return
    if flags["last_key"] == "ctrl":
        time.sleep(0.1)
        flags["which"] = (flags["which"] + 1) % 2
        return
    if flags["last_key"] == "s":
        time.sleep(0.1)
        flags["split"] = not flags["split"]
        if flags["split"]:
            coords[(flags["which"] + 1) % 2] = coords[flags["which"]].copy()
        return
    if flags["last_key"] == "esc":
        flags["quit"] = True
        return


def last_pressed_key(flags):
    while True:
        event = keyboard.read_event()
        if event.name == "shift":
            flags["shift"] = event.event_type == keyboard.KEY_DOWN
            continue
        if event.event_type == keyboard.KEY_DOWN:
            flags["key_change"] = True
            flags["last_key"] = event.name
            flags["is_pressed"] = True
        if event.event_type == keyboard.KEY_UP:
            flags["is_pressed"] = False



def display_hologram(window, hologram, mask, mask_flag, ct2pi):
    if mask_flag:
        hologram = hologram + mask
    hologram_int = (hologram % (2 * np.pi) * ct2pi / (2 * np.pi)).astype(np.uint8)
    hologram_img = im.fromarray(hologram_int)
    cl.display_image_on_external_screen(window, hologram_img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_name", help="name of the mask file")
    parser.add_argument("-ct2pi", "--correspond_to2pi", type=int, required=True, help="value of pixel corresponding to 2pi phase shift")
    parser.add_argument("-bs", "--big_step", type=int, default=20, help="big step size")
    parser.add_argument("-m", "--mirror", action="store_true", help="mirrors left-right and up-down")
    args = parser.parse_args()

    # holograms_dir = "holograms/single_trap_grid_holograms"
    # if not os.path.exists(holograms_dir):
    #     subprocess.run(['python', 'src/make_single_trap_grid_holograms.py', holograms_dir], capture_output=True, text=True, check=True)
    args.holograms_dir = "holograms/single_trap_grid_holograms"
    

    main(args)

