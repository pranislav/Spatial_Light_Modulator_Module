'''from given parametrization of path of N points
and given number of frames creates list of lists of N tuples.
i-th inner list contains coordinates of each point in i-th time point'''

from __future__ import annotations
import numpy as np
from constants import slm_height as h, slm_width as w

# number_of_frames = 10


def two_circulating_dots_quarterized(n: int):
    t = n * (2 * np.pi) / 360
    w_ = w // 2
    h_ = h // 2
    return [(w_ * (1/2 + 1/3 * np.cos(t)), h_ * (1/2 + 1/3 * np.sin(t))),
            (w_ * (1/2 + 1/3 * np.cos(t + np.pi/2)), h_ * (1/2 + 1/3  * np.sin(t + np.pi/2)))]


def two_circulating_dots(n: int):
    t = n * (2 * np.pi) / 360
    return [(w * (1/2 + 1/3 * np.cos(t)), h * (1/2 + 1/3 * np.sin(t))),
            (w * (1/2 + 1/3 * np.cos(t + np.pi/2)), h * (1/2 + 1/3  * np.sin(t + np.pi/2)))]

def create_list_of_position_lists(parametrization: function, number_of_frames: int, rescale_param: int=1):
    list_of_position_lists = []
    for i in range(number_of_frames):
        list_of_position_lists.append(parametrization(rescale_param * i))
    return list_of_position_lists

# print(create_list_of_position_lists(my_parametrization, number_of_frames))