'''generates hologram series for calibrating SLM'''

from PIL import Image as im
import numpy as np
from random import randint
import constants as c
import os
from copy import deepcopy


x_angle = 1
y_angle = 1
unit = c.u
subdomain_size = 32
precision = 8

subdomains_number_x = c.slm_width // subdomain_size
subdomains_number_y = c.slm_height // subdomain_size

reference_position = (subdomain_size * 3 * subdomains_number_x // 5, subdomain_size * 3 * subdomains_number_y // 5)

def make_holograms():
    hologram_set_name = f"size{subdomain_size}_precision{precision}_x{x_angle}_y{y_angle}"
    dest_dir = f"lc-slm/holograms_for_calibration/{hologram_set_name}"
    if not os.path.exists(dest_dir): os.makedirs(dest_dir)
    reference_hologram = make_hologram(np.zeros((c.slm_height, c.slm_width)), reference_position)
    im.fromarray(reference_hologram).convert("L").save(f"{dest_dir}/{reference_position}.png")
    phase_step = 256 // precision
    for i in range(subdomains_number_y):
        for j in range(subdomains_number_x):
            if not os.path.exists(f"{dest_dir}/{i}/{j}"):
                os.makedirs(f"{dest_dir}/{i}/{j}")
            sbd_position = (subdomain_size * i, subdomain_size * j)
            hologram_arr = make_hologram(deepcopy(reference_hologram), sbd_position)
            im.fromarray(hologram_arr).convert("L").save(f"{dest_dir}/{i}/{j}/0.png")
            for k in range(1, precision):
                hologram_arr = shift_hologram(hologram_arr, sbd_position, phase_step)
                im.fromarray(hologram_arr).convert("L").save(f"{dest_dir}/{i}/{j}/{k}.png")



def make_hologram(substrate: np.array, subdomain_position):
    i_0, j_0 = subdomain_position
    for i in range(subdomain_size):
        for j in range(subdomain_size):
            const = 256 * c.px_distance / c.wavelength # 256 gives more accurate result
            new_phase = const * (np.sin(y_angle * unit) * i + np.sin(x_angle * unit) * j)
            substrate[i + i_0, j + j_0] = new_phase % 256
    return substrate


# for this implementation the continuity check works just because of a coincidence
# for arbitrary angle it generally should not work
def continuity_check():
    '''chcecks whether perfectly flat slm + optical path without aberrations
    would yield uniform phase mask (it should)
    '''
    hologram = np.zeros((c.slm_height, c.slm_width))
    for i in range(subdomains_number_y):
        for j in range(subdomains_number_x):
            make_hologram(hologram, (subdomain_size*i, subdomain_size*j))
    im.fromarray(hologram).show()

def shift_hologram(substrate: np.array, subdomain_position, phase_step):
    i_0, j_0 = subdomain_position
    for i in range(subdomain_size):
        for j in range(subdomain_size - 1):
            substrate[i + i_0, j + j_0] = (substrate[i + i_0, j + j_0] + phase_step) % 256
    return substrate

if __name__ == "__main__":
    continuity_check()
