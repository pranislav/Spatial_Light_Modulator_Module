'''computes optimal angle between
axis normal to DMD plane and incident laser ray
'''

import numpy as np
from dmd_constants import wavelength, a
from functools import partial
import matplotlib.pyplot as plt


resolution = 1000
alpha = np.array([np.pi * i/resolution for i in range(-resolution//2, resolution//2)])


def optimal_angle(alpha: float) -> float:
    '''the optimal angle is alpha for which holds:
    optimal_angle(alpha) = n * wavelength/a, where n is int
    '''
    gamma = 12* 2*np.pi/360
    return np.sin(alpha + 2*gamma) - np.sin(alpha)


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


# plotting

# horisontal lines
fmax = max(optimal_angle(alpha))
fmin = min(optimal_angle(alpha))
y_max = int(fmax * (a/wavelength))
y_min = int(fmin * (a/wavelength))
step = wavelength/a
hlines_list = [n * step for n in range(y_min, y_max+1)]

plt.plot(alpha, optimal_angle(alpha))
plt.hlines(hlines_list, -np.pi/2, np.pi/2, color='k')
for n in range(y_min, y_max+1):
    plt.text(0, (n + 0.1) * step, f"n = {n}")
plt.show()


# finding and printing optimal angle

# n = int(input("choose n: "))
angle = halving(optimal_angle, n*wavelength/a, 0, np.pi/2, 1, 0.0001)
print(f"optimal angle of order {n} is {angle} rad")

# the_angle = partial(halving, func=optimal_angle, x_max=0, x_min=np.pi/2, x_med=1, resolution=0.001)
the_angle = halving(optimal_angle, 8*wavelength/a, 0, np.pi/2, 1, 0.0001)
