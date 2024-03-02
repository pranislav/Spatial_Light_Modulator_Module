import numpy as np
import os
import matplotlib.pyplot as plt
from algorithms import GD_for_moving_traps, generate_initial_input
from PIL import Image as im
from helping_functions_for_slm_generate_etc import remove_files_in_dir, create_gif
from constants import slm_width as w, slm_height as h


def generate_hologram_gif(source_dir: str, preview: bool=False, learning_rate: float=0.002,
       mask_relevance: float=10, tolerance: float=0.001, max_loops: int=10, unsettle=0):
    '''transforms sequence of images of traps into
    a gif of holograms corresponding to those traps'''
    dest_dir_holograms = "holograms/gif_source"
    dest_dir_preview = "images/gif_preview"
    for dest_dir in [dest_dir_holograms, dest_dir_preview]:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        remove_files_in_dir(dest_dir)
    err_evl_list = []
    initial_guess = generate_initial_input(w, h)
    for i, file in enumerate(os.listdir(source_dir)):
        img = im.open(f"{source_dir}/{file}")
        target = np.sqrt(np.array(img))
        print(f"creating hologram for {i}. trap: ", '')
        hologram, initial_guess, exp_tar, err_evl = GD_for_moving_traps(target, initial_guess, learning_rate, mask_relevance, tolerance, max_loops, unsettle)
        hologram_img = im.fromarray(hologram)
        hologram_img.convert("RGB").save(f"{dest_dir_holograms}/{i}.png", quality=100) # do i need quality if saving as png?
        err_evl_list.append(err_evl)
        tolerance = err_evl[-1]
        if preview:
            exp_tar_img = im.fromarray(exp_tar)
            exp_tar_img.convert("RGB").save(f"{dest_dir_preview}/exp_tar_{i}.png", quality=100)
        # if i == 2: break
    plot_err_evl(err_evl_list)
    name = f"{os.path.basename(source_dir)}_lr={learning_rate:.2}_mr={mask_relevance}_tol={tolerance:.2}_loops={max_loops}_unsettle={unsettle}"
    create_gif(dest_dir_holograms, f"holograms/{name}.gif")
    if preview:
        create_gif(dest_dir_preview, f"images/{name}_exp_tar.gif")


def plot_err_evl(err_evl_list):
    '''plots error evolution for each trap into one plot'''
    for i, err_evl in enumerate(err_evl_list):
        plt.plot(err_evl, label=i)
    plt.legend()
    plt.show()
    