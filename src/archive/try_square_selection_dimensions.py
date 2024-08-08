import wavefront_correction as wcl
import numpy as np
import argparse

def print_square(n):
    a = np.zeros((5*n, 5*n))
    i, j = 2*n, 2*n
    i_ul, j_ul = wcl.get_upper_left_corner_coords((i, j), n)
    i_lr, j_lr = wcl.get_lower_right_corner_coords((i, j), n)
    b = wcl.square_selection(a, (i_ul, j_ul), (i_lr, j_lr))
    print(b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="side length of the square")
    args = parser.parse_args()
    print_square(args.n)