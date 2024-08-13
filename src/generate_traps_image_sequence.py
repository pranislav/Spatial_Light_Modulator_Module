from traps_images import create_sequence_dots
import argparse
import numpy as np
import constants as c


def main(args):
    args.parametrization = choose_parametrization(args.parametrization)
    list_of_position_lists = create_list_of_position_lists(args.parametrization, args.number_of_frames, args.rescale_parameter)
    create_sequence_dots(list_of_position_lists, args.name)


def choose_parametrization(parametrization):
    if parametrization == "circulating_dot_quarterized":
        return circulating_dot_quarterized
    if parametrization == "two_circulating_dots_quarterized":
        return two_circulating_dots_quarterized
    if parametrization == "two_circulating_dots":
        return two_circulating_dots
    else:
        return None

def two_circulating_dots_quarterized(n: int):
    t = n * (2 * np.pi) / 360
    w_ = c.w // 2
    h_ = c.h // 2
    return [(w_ * (1/2 + 1/3 * np.cos(t)), h_ * (1/2 + 1/3 * np.sin(t))),
            (w_ * (1/2 + 1/3 * np.cos(t + np.pi/2)), h_ * (1/2 + 1/3  * np.sin(t + np.pi/2)))]

def circulating_dot_quarterized(n: int):
    t = n * (2 * np.pi) / 360
    w_ = c.w // 2
    h_ = c.h // 2
    return [(w_ * (1/2 + 1/10 * np.cos(t)), h_ * (1/2 + 1/10 * np.sin(t)))]


def two_circulating_dots(n: int):
    t = n * (2 * np.pi) / 360
    return [(c.w * (1/2 + 1/3 * np.cos(t)), c.h * (1/2 + 1/3 * np.sin(t))),
            (c.w * (1/2 + 1/3 * np.cos(t + np.pi/2)), c.h * (1/2 + 1/3  * np.sin(t + np.pi/2)))]

def create_list_of_position_lists(parametrization: function, number_of_frames: int, rescale_parameter: int=1):
    '''creates list of lists of N tuples
    from given parametrization of paths of N points
    and given number of frames.
    i-th inner list contains coordinates of each point in i-th time point'''
    list_of_position_lists = []
    for i in range(number_of_frames):
        list_of_position_lists.append(parametrization(rescale_parameter * i))
    return list_of_position_lists


if __name__ == "__main__":
    description = '''Create sequence of images capturing moving dots
    with given parametrization of their paths.
    Images are saved in the directory holograms/traps_images/<name>.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description)
    parser.add_argument("name", type=str, help="name of the sequence")
    parser.add_argument("-p", "--parametrization", type=str, choices=["circulating_dot_quarterized", "two_circulating_dots_quarterized", "two_circulating_dots"], default="circulating_dot_quarterized", help="parametrization of the paths of the dots")
    parser.add_argument("-r", "--rescale_parameter", metavar="FLOAT", type=float, default=1, help="rescales parameter (faster (> 1) or slower (< 1)) motion")
    parser.add_argument("-n", "--number_of_frames", metavar="INT", type=int, default=360)
    args = parser.parse_args()

    main(args)
