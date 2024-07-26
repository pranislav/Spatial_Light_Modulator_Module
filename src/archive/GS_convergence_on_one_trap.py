import display_holograms as dh
import argparse
import os
import algorithms as alg
import constants as c
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

args = argparse.Namespace()
args.incomming_intensity = "uniform"
args.tolerance = 0.0
args.max_loops = 2
args.gif = False
args.print_info = False
args.plot_error = False
args.correspond_to2pi = 255


def plot_error_evolutions(list_of_error_evolutions):
    for error_evl in list_of_error_evolutions:
        plt.plot(error_evl)
    plt.show()

def generate_random_tuples(N):
    random_tuples = [(random.randint(0, c.slm_height), random.randint(0, c.slm_width)) for _ in range(N)]
    return random_tuples



black_image = np.zeros((c.slm_height, c.slm_width), dtype=np.uint8)
list_of_error_evolutions = []
for coords in generate_random_tuples(30):
    black_image[coords] = 255
    _, _, error_evl = alg.GS(black_image, args)
    list_of_error_evolutions.append(error_evl)
    black_image[coords] = 0
plot_error_evolutions(list_of_error_evolutions)