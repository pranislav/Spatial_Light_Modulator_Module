import generate_hologram as gh
from algorithms import GD
import argparse
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

def compare_error_evolution(args):
    target = gh.prepare_target(args.img_name, args)
    fill_unnecessary_args(args)
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    name_of_vary = choose_name_of_vary(args)
    for value in args.values:
        set_argument(args, value)
        _, _, error_evolution = GD(target, args)
        plt.plot(error_evolution, label=f"{name_of_vary}: {value}")
    plt.xlabel("iteration")
    # plt.xticks(range(int(len(error_evolution)) + 1))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("error")
    plt.legend()
    save_plot(args)

def set_argument(args, value):
    if args.vary == 'wa':
        args.white_attention = value
    elif args.vary == 'lr':
        args.learning_rate = value
    elif args.vary == 'u':
        args.unsettle = value
    elif args.vary == 'ii':
        args.initial_input = value
        

def choose_name_of_vary(args):
    if args.vary == 'wa':
        return "white_attention"
    elif args.vary == 'lr':
        return "learning_rate"
    elif args.vary == 'u':
        return "unsettle"
    elif args.vary == 'ii':
        return "initial_input"
    else:
        return "unknown"

def save_plot(args):
    name, specification = create_name(args)
    values = "_".join(map(str, args.values))
    name_to_be = f"{args.dest_dir}/{name}_varying_{choose_name_of_vary(args)}_values_{values}_{specification}"
    while os.path.exists(name_to_be + ".png"):
        name_to_be += "I"
    plt.savefig(name_to_be + ".png")


def create_name(args):
    name = args.img_name.split("/")[-1].split(".")[0]
    if args.quarterize:
        name += "_quarterized"
    if args.invert:
        name += "_inverted"
    specification = f"learning_rate{args.learning_rate}_attention_{args.white_attention}_unsettle_{args.unsettle}_loops_{args.max_loops}"
    return name, specification


def fill_unnecessary_args(args):
    args.gif = False
    args.gif_type = None
    args.gif_dir = None
    args.decline = None
    args.lens = None
    args.correspond_to2pi = 256
    args.incomming_intensity = "uniform"
    args.print_info = False
    args.tolerance = 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compare error evolution of computing hologram with GD for different learning rates') 
    parser.add_argument('img_name', type=str, help='path to the image')
    parser.add_argument('-l', '--max_loops', type=int, default=10, help='max loops')
    parser.add_argument('-wa', '--white_attention', type=float, default = 1, help='white attention')
    parser.add_argument('-i', '--invert', action='store_true', help='invert image')
    parser.add_argument('-q', '--quarterize', action='store_true', help='quarterize')
    parser.add_argument('-u', '--unsettle', type=int, default=0, help='unsettle')
    parser.add_argument('-lr', '--learning_rate', default=0.005, type=float, help='learning rates')
    parser.add_argument('-ig', '--initial_guess', default="random", choices=["random", "fourier"], help='initial input')
    parser.add_argument('-v', '--vary', choices=['wa', 'lr', 'u', 'ig'], help='vary white attention, learning rate, unsettle or initial guess')
    parser.add_argument('values', nargs='+', type=float, help='values to vary')
    args = parser.parse_args()
    args.dest_dir = "compare_error_evolution_GD_params"
    if args.vary == 'ii':
        args.values = ["random", "ifft2"]

    compare_error_evolution(args)
    