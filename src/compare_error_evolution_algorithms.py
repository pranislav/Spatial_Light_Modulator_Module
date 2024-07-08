import generate_hologram as gh
from algorithms import GD, GS
import argparse
import matplotlib.pyplot as plt
import os
import compare_error_evolution_GD_params as compare

def compare_error_evolution_algorithms(args):
    target = gh.prepare_target(args.img_name, args)
    compare.fill_unnecessary_args(args)
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    print("GD")
    _, _, error_evolution = GD(target, args)
    plt.plot(error_evolution, label=f"GD")
    print("GS")
    _, _, error_evolution = GS(target, args)
    plt.plot(error_evolution, label=f"GS")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.legend()
    save_plot(args)

def save_plot(args):
    name, specification = compare.create_name(args)
    plt.savefig(f"{args.dest_dir}/{name}_{specification}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compare error evolution of computing hologram with GD for different learning rates') 
    parser.add_argument('img_name', type=str, help='path to the image')
    parser.add_argument('-l', '--max_loops', type=int, default=10, help='max loops')
    parser.add_argument('-wa', '--white_attention', type=float, default = 1, help='white attention')
    parser.add_argument('-i', '--invert', action='store_true', help='invert image')
    parser.add_argument('-q', '--quarterize', action='store_true', help='quarterize')
    parser.add_argument('-u', '--unsettle', type=int, default=0, help='unsettle')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='learning rate')
    parser.add_argument('-ig', '--initial_guess', type=str, choices={"random", "fourier"}, default="random", help='initial guess')
    args = parser.parse_args()
    args.dest_dir = "compare_error_evolution_algorithms"

    compare_error_evolution_algorithms(args)