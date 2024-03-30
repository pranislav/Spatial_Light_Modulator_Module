import calibration_lib as cl
from PIL import Image as im
import numpy as np
import sys
import constants as c


def display_holograms(mask_name):
    if mask_name == "none":
        mask_arr = np.zeros((c.h, c.w))
    else:
        mask_im = im.open(f"lc-slm/holograms_for_calibration/calibration_phase_mask/{mask_name}")
        mask_arr = np.array(mask_im)
    window = cl.create_tk_window()
    while True:
        name = input("gimme a name of a hologram or quit with \"q\"")
        if name == "q": break
        path = f"lc-slm/holograms/{name}"
        hologram_im = im.open(path)
        hologram_arr = np.array(hologram_im)
        masked_hologram_arr = (hologram_arr + mask_arr) % 256
        masked_hologram_im = im.fromarray(masked_hologram_arr)
        cl.display_image_on_external_screen_img(window, masked_hologram_im)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python display_holograms.py <calibration_mask_name>\nrun from project root. \"none\" for no mask")
        sys.exit(1)
    
    mask_name = sys.argv[1]

    display_holograms(mask_name)
