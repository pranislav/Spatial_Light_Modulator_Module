"""script for comparing the time of one loop of the Gerchberg-Saxton algorithm and the gradient descent algorithm
"""

import time
from algorithms import gerchberg_saxton, gradient_descent
import generate_hologram as gh
import argparse


def measure_loop_time(algorithm):
    args = make_some_args()
    args.max_loops = 100
    target = gh.prepare_target("duck.png", args)
    start_time = time.time()
    algorithm(target, args)
    end_time = time.time()
    loop_time = (end_time - start_time) / args.max_loops
    print(f"Time for one loop: {loop_time} seconds")


def make_some_args():
    args = argparse.Namespace()
    args.incomming_intensity = "uniform"
    args.tolerance = 0
    args.white_attention = 1
    args.invert = True
    args.quarterize = True
    args.unsettle = 0
    args.learning_rate = 0.005
    args.algorithm = "gerchberg_saxton"
    args.gif = False
    args.gif_type = None
    args.gif_dir = None
    args.print_info = False
    args.correspond_to2pi = 256
    args.initial_guess = "random"
    return args


for algorithm in [gerchberg_saxton, gradient_descent]:
    measure_loop_time(algorithm)
