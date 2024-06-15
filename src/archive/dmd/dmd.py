from __future__ import annotations
from PIL import Image as im
from functools import partial
import numpy as np
from incident_angle import the_angle
from dmd_constants import diagonal_spacing as a, i_size, j_size, wavelength


# alpha = 0.0 # np.pi/4 # rad
alpha = the_angle


# construtctor functions

def conversion(i: int, j: int) -> tuple[float]:
    '''converts given indices to coordinates
    '''
    i_0 = (i_size + (j_size%2)/2) / 2
    j_0 = j_size/2
    x = (i + (j%2)/2 - i_0) * a
    y = (j - j_0)/2 * a
    return (x, y)


def scan(condition: callable) -> im:
    img = im.new('L', (i_size, j_size))
    for i in range(i_size):
        for j in range(j_size):
            x, y = conversion(i, j)
            if condition(x, y):
                img.putpixel((i, j), 255)
    return img


def my_scan(value):
    img = im.new('L', (i_size, j_size))
    for i in range(i_size):
        for j in range(j_size):
            img.putpixel((i, j), value)
    return img



# conditions for functions

def circle_cond(R: float, x: float, y: float) -> bool:
    return R**2 > x**2 + y**2


def lens_cond(focal_length: float, x: float, y: float) -> bool:
    f = focal_length
    r_sq = x**2 + y**2
    d_phi = 2*np.pi/wavelength * (np.sqrt(f**2 + r_sq) - f - x*np.sin(alpha))
    phi = 2*np.pi*f/wavelength
    return (phi - d_phi + np.pi/2) % (2*np.pi) < np.pi


def lens_plus_decline_x_cond(focal_length: float, angle: float, x: float, y: float) -> bool:
    f = focal_length
    r_sq = x**2 + y**2
    d_phi = 2*np.pi/wavelength * (np.sqrt(f**2 + r_sq - 2*f*x*np.sin(angle)) - f - x*np.sin(alpha))
    phi = 2*np.pi*f/wavelength
    return (phi - d_phi + np.pi/2) % (2*np.pi) < np.pi


def decline_x_cond(angle: float, x: float, y: float) -> bool:
    d_d = x * (np.sin(alpha) - np.sin(angle))
    return (2*np.pi*d_d/wavelength + np.pi/2) % (2*np.pi) < np.pi


def decline_y_cond(angle: float, x: float, y: float) -> bool:
    d_d = y * (- np.sin(angle))
    return (2*np.pi*d_d/wavelength + np.pi/2) % (2*np.pi) < np.pi


# actual functions

def circle_img(R: float) -> im:
    return scan(partial(circle_cond, R))


def lens_img(focal_length: float) -> im:
    return scan(partial(lens_cond, focal_length))


def lens_plus_decline_x_img(focal_length: float, angle: float) -> im:
    return scan(partial(lens_plus_decline_x_cond, focal_length, angle))

def fun_and(fun1, fun2, x, y):
    return fun1(x, y) and fun2(x, y)

def lens_plus_decline_x_img_II(focal_length: float, angle: float) -> im:
    return scan(partial(fun_and, partial(lens_cond, focal_length), partial(decline_x_cond, angle)))

def decline_x_img(angle: float) -> im:
    return scan(partial(decline_x_cond, angle))


def decline_y_img(angle: float) -> im:
    return scan(partial(decline_y_cond, angle))


# circle_img(150*a).show()

