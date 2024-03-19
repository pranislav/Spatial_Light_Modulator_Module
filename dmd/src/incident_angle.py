'''computes optimal angle between
axis normal to DMD plane and incident laser ray.
the optimal angle is alpha for which holds:
same_phase_distance(alpha) = n * wavelength/a, where n is int
'''

import numpy as np
from dmd_constants import wavelength, diagonal_spacing as a, pixel_on_angle
from functools import partial
import matplotlib.pyplot as plt


resolution = 1000
alpha = np.array([np.pi * i/resolution for i in range(-resolution//2, resolution//2)])


def same_phase_distance(alpha: float) -> float:
    return np.sin(alpha + 2 * pixel_on_angle) - np.sin(alpha)


def halving(func: callable, offset, x_max: float, x_min: float,\
        x_med: float, resolution: float) -> float:
    '''general interval halving method
    finds a value in domain of function (on interval (x_max, x_min))
    for which the value of the function is equal to the offset value 
    '''
    if abs(x_max - x_min) < resolution:
        return (x_max + x_min)/2
    if np.sign(func(x_med) - offset) == np.sign(func(x_max) - offset):
        return halving(func, offset, x_med, x_min, (x_med + x_min)/2, resolution)
    else:
        return halving(func, offset, x_max, x_med, (x_max + x_med)/2, resolution)


def make_horisontal_lines():
    fmax = max(same_phase_distance(alpha))
    fmin = min(same_phase_distance(alpha))
    y_max = int(fmax * (a/wavelength))
    y_min = int(fmin * (a/wavelength))
    step = wavelength/a
    hlines_list = [n * step for n in range(y_min, y_max+1)]
    return hlines_list, y_min, y_max, step


def plot_same_phase_distance():
    hlines_list, y_min, y_max, step = make_horisontal_lines()
    plt.plot(alpha, same_phase_distance(alpha))
    plt.hlines(hlines_list, -np.pi/2, np.pi/2, color='k')
    for n in range(y_min, y_max+1):
        plt.text(0, (n + 0.1) * step, f"n = {n}")
    plt.show()


def optimal_angle_user_defined_order():
    n = int(input("choose order of optimal angle: "))
    angle = halving(same_phase_distance, n*wavelength/a, 0, np.pi/2, 1, 0.0001)
    print(f"optimal angle of order {n} is {angle} rad")


plot_same_phase_distance()


first_positive_angle_order = 8
fpao = first_positive_angle_order
the_angle = halving(same_phase_distance, fpao * wavelength/a, 0, np.pi/2, 1, 0.0001)
