import generate_hologram as gh
from algorithms import GD
import argparse
import matplotlib.pyplot as plt
import os

def compare_error_evolution(args):
    target = gh.prepare_target(args.img_name, args)
    fill_unnecessary_args(args)
    if not os.path.exists("learning_rate_comparison"):
        os.makedirs("learning_rate_comparison")
    for learning_rate in args.learning_rates:
        args.learning_rate = learning_rate
        _, _, error_evolution = GD(target, args)
        plt.plot(error_evolution, label=f"learning rate: {learning_rate}")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.legend()
    save_plot(args)

def save_plot(args):
    specification = f"tolerance_{args.tolerance}_max_loops_{args.max_loops}_white_attention_{args.white_attention}_invert_{args.invert}_quarterize_{args.quarterize}_unsettle_{args.unsettle}"
    leraning_rates = "_".join([str(lr) for lr in args.learning_rates])
    plt.savefig(f"learning_rate_comparison/{specification}_learning_rates_{leraning_rates}.png")


def fill_unnecessary_args(args):
    args.gif = False
    args.gif_type = None
    args.gif_dir = None
    args.preview = False
    args.decline = None
    args.lens = None
    args.correspond_to2pi = 256
    args.incomming_intensity = "uniform"
    args.print_info = False





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compare error evolution of computing hologram with GD for different learning rates') 
    parser.add_argument('img_name', type=str, help='path to the image')
    parser.add_argument('-tolerance', type=float, default=0, help='tolerance for error')
    parser.add_argument('-max_loops', type=int, default=10, help='max loops')
    parser.add_argument('-wa', '--white_attention', type=float, default = 1, help='white attention')
    parser.add_argument('-i', '--invert', action='store_true', help='invert image')
    parser.add_argument('-quarterize', action='store_true', help='quarterize')
    parser.add_argument('-unsettle', type=int, default=0, help='unsettle')
    parser.add_argument('-learning_rates', nargs='+', type=float, help='learning rates')
    args = parser.parse_args()

    compare_error_evolution(args)