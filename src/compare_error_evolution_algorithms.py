import generate_hologram as gh
from algorithms import gradient_descent, gerchberg_saxton
import argparse
import matplotlib.pyplot as plt
import os
import compare_error_evolution_gradient_descent_params as compare
import wavefront_correction as wfc

def compare_error_evolution_algorithms(args):
    target = gh.prepare_target(args.img_name, args)
    compare.fill_unnecessary_args(args)
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    print("gradient_descent")
    _, _, error_evolution = gradient_descent(target, args)
    plt.figure(figsize=(10, 5))
    plt.plot(error_evolution, label=f"gradient_descent")
    print("gerchberg_saxton")
    _, _, error_evolution = gerchberg_saxton(target, args)
    plt.plot(error_evolution, label=f"gerchberg_saxton")
    plt.ylim(bottom=0)
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.legend()
    save_plot(args)
    if args.show:
        plt.show()

def save_plot(args):
    img_name, specification = compare.create_name(args)
    name = f"{args.dest_dir}/{img_name}_{specification}.png"
    plt.savefig(wfc.originalize_name(name), bbox_inches='tight')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='compare error evolution of computing hologram with gradient_descent for different learning rates') 
    parser.add_argument('img_name', type=str, help='path to the image')
    parser.add_argument('-l', '--max_loops', type=int, default=10, help='max loops')
    parser.add_argument('-wa', '--white_attention', type=float, default = 1, help='white attention')
    parser.add_argument('-i', '--invert', action='store_true', help='invert image')
    parser.add_argument('-q', '--quarterize', action='store_true', help='quarterize')
    parser.add_argument('-u', '--unsettle', type=int, default=0, help='unsettle')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='learning rate')
    parser.add_argument('-ig', '--initial_guess', type=str, choices={"random", "fourier"}, default="random", help='initial guess')
    parser.add_argument('-rs', '--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('-show', action='store_true', help='show plot')
    args = parser.parse_args()
    args.dest_dir = "images/compare_error_evolution_algorithms"

    compare_error_evolution_algorithms(args)