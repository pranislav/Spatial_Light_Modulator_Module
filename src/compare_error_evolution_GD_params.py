import generate_hologram as gh
from algorithms import GD
import argparse
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
import wavefront_correction as wfc

def compare_error_evolution(args):
    target = gh.prepare_target(args.img_name, args)
    fill_unnecessary_args(args)
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    name_of_vary = choose_name_of_vary(args)
    plt.figure(figsize=(10, 5))
    for value in args.values:
        set_argument(args, value)
        _, _, error_evolution = GD(target, args)
        plt.plot(error_evolution, label=f"{name_of_vary}: {value}")
    plt.ylim(bottom=0)
    plt.xlabel("iteration")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("error")
    plt.legend()
    save_plot(args)
    if args.show:
        plt.show()

def set_argument(args, value):
    if args.vary == 'wa':
        args.white_attention = value
    elif args.vary == 'lr':
        args.learning_rate = value
    elif args.vary == 'u':
        args.unsettle = value
    elif args.vary == 'ig':
        args.initial_guess = value
    elif args.vary == 'rs':
        args.random_seed = value
        

def choose_name_of_vary(args):
    if args.vary == 'wa':
        return "white_attention"
    elif args.vary == 'lr':
        return "learning_rate"
    elif args.vary == 'u':
        return "unsettle"
    elif args.vary == 'ig':
        return "initial_guess"
    elif args.vary == 'rs':
        return "random_seed"
    else:
        return "unknown"

def save_plot(args):
    img_name, specification = create_name(args)
    values = "_".join(map(str, args.values))
    name_to_be = f"{args.dest_dir}/{img_name}_varying_{choose_name_of_vary(args)}_values_{values}_{specification}.png"
    plt.savefig(wfc.originalize_name(name_to_be), bbox_inches='tight')


def create_name(args):
    img_name = args.img_name.split("/")[-1].split(".")[0]
    if args.quarterize:
        img_name += "_quarterized"
    if args.invert:
        img_name += "_inverted"
    specification = f"lr{args.learning_rate}_wa_{args.white_attention}_unstl_{args.unsettle}_ig_{args.initial_guess}_l_{args.max_loops}"
    return img_name, specification


def fill_unnecessary_args(args):
    args.gif = False
    args.gif_type = None
    args.gif_dir = None
    args.deflect = None
    args.lens = None
    args.correspond_to2pi = 256
    args.incomming_intensity = "uniform"
    args.print_info = False
    args.tolerance = 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='compare error evolution of computing hologram with GD for different learning rates') 
    parser.add_argument('img_name', type=str, help='path to the image')
    parser.add_argument('-l', '--max_loops', type=int, default=10, help='max loops')
    parser.add_argument('-wa', '--white_attention', type=float, default = 1, help='white attention')
    parser.add_argument('-i', '--invert', action='store_true', help='invert image')
    parser.add_argument('-q', '--quarterize', action='store_true', help='quarterize')
    parser.add_argument('-u', '--unsettle', type=int, default=0, help='unsettle')
    parser.add_argument('-lr', '--learning_rate', default=0.005, type=float, help='learning rates')
    parser.add_argument('-ig', '--initial_guess', default="random", choices=["random", "fourier", "unnormed", "zeros", "ones", "old"], help='initial input')
    parser.add_argument('-rs', '--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('-v', '--vary', choices=['wa', 'lr', 'u', 'ig', 'n', 'rs'], help='vary white attention, learning rate, unsettle, initial guess or random seed')
    parser.add_argument('values', nargs='*', type=float, help='values to vary')
    parser.add_argument('-s', '--show', action='store_true', help='show the plot')
    args = parser.parse_args()
    args.dest_dir = "images/compare_error_evolution_GD_params"
    if args.vary == 'ig':
        args.values = ["random", "fourier", "unnormed", "zeros", "ones", "old"]

    compare_error_evolution(args)
    