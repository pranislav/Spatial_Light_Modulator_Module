'''transforms sequence of images of traps into
a gif of holograms corresponding to those traps'''

import numpy as np
import os
import matplotlib.pyplot as plt
from algorithms import GD_for_moving_traps, generate_initial_input
from PIL import Image as im
from helping_functions_for_slm_generate_etc import remove_files_in_dir, create_gif
from constants import slm_width as w, slm_height as h


def generate_hologram_gif(source_dir: str):
    '''transforms sequence of images of traps into
    a gif of holograms corresponding to those traps'''
    dest_dir = "holograms/gif_source"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    remove_files_in_dir(dest_dir)
    err_evl_list = []
    initial_guess = generate_initial_input(w, h)
    for i, file in enumerate(os.listdir(source_dir)):
        img = im.open(f"{source_dir}/{file}")
        target = np.sqrt(np.array(img))
        print(f"creating hologram for {i}. trap: ", '')
        hologram, initial_guess, err_evl = GD_for_moving_traps(target, initial_guess)
        hologram_img = im.fromarray(hologram)
        hologram_img.convert("RGB").save(f"{dest_dir}/{i}.png", quality=100) # do i need quality if saving as png?
        err_evl_list.append(err_evl)
        if i == 2: break
    plot_err_evl(err_evl_list)
    name = "duhh" # create_name()
    create_gif(dest_dir, f"holograms/{name}.gif")


def create_name():
    '''creates name for gif based on given parameters'''
    pass


def plot_err_evl(err_evl_list):
    '''plots error evolution for each trap into one plot'''
    for err_evl in err_evl_list:
        plt.plot(err_evl)
    plt.show()
    

generate_hologram_gif("images/moving_traps/moving_traps_4")
    