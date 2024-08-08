from __future__ import annotations
import numpy as np
import PIL.Image as im
import constants as c
import os

white = 255

def make_traps_image(coordinates: list[tuple], object: function, size_x: float=1, size_y: float=1) -> im:
    img = im.new('L', (c.slm_width, c.slm_height))
    for point in coordinates:
        object(img, point, size_x, size_y, white)
    return img


'''creates a sequence of images of multiple moving points
from given lists of coordinates
and saves them into a file'''
def create_sequence_dots(list_of_position_lists: list[list[tuple]], name: str):
    if not os.path.exists(f"images/moving_traps/{name}"):
        os.makedirs(f"images/moving_traps/{name}")
    for i, position_list in enumerate(list_of_position_lists):
        img = make_traps_image(position_list, dot)
        img.save(f"images/moving_traps/{name}/{i}.png")


# coordinate generators --------------------------------------------

def grating(x_num: int, y_num: int) -> list[tuple]:
    coordinates = []
    x_dist = int(c.slm_width / (x_num + 1))
    y_dist = int(c.slm_height / (y_num + 1))
    for i in range(x_num):
        for j in range(y_num):
            x = (i + 1) * x_dist
            y = (j + 1) * y_dist
            coordinates.append((x, y))
    return coordinates

def fract_position(x_fract, y_fract):
    return [(int(c.slm_width / x_fract), int(c.slm_height / y_fract))]

def dec_position(x_dec, y_dec):
    return [(int(c.slm_width * x_dec), int(c.slm_height * y_dec))]

def user_defined(x, y):
    pass


# objects --------------------------------------------

def ellipse_arr(target_img: im, coor: tuple[int], x_d: float, y_d: float, color: int) -> None:
    x_coor, y_coor = coor
    w, h = target_img.size
    target_arr = np.array(target_img)
    for i in range(w):
        for j in range(h):
            if ((i - x_coor) / x_d)**2 + ((j - y_coor) / y_d)**2 < 1:
                target_arr[j, i] = color
    return im.fromarray(target_arr)

def ellipse(target_img: im, coor: tuple[int], x_d: float, y_d: float, color: int) -> None:
    x_coor, y_coor = coor
    w, h = target_img.size
    for i in range(w):
        for j in range(h):
            if ((i - x_coor) / x_d)**2 + ((j - y_coor) / y_d)**2 < 1:
                target_img.putpixel((i, j), color)
    return target_img

def dot(target_img: im, coords: tuple[int], ___, __, color: int) -> None:
    x_coor = round(coords[0])
    y_coor = round(coords[1])
    target_img.putpixel((x_coor, y_coor), color)
    return target_img


def rectangle(target_img: im, coor: tuple[int], side_x: float, side_y: float, color: int):
    x_coor, y_coor = coor
    w, h = target_img.size # naopak to tu bolo!! a kto vie, kde este je | nie, daco ine muselo byt zle, lebo pri centrovani som nemohol vyjst z rangeu pri prehodenych suradniciach
    for i in range(w):
        for j in range(h):
            x_cond = (x_coor - side_x // 2) < i <= (x_coor + side_x // 2) or i == x_coor
            y_cond = (y_coor - side_y // 2) < j <= (y_coor + side_y // 2) or j == y_coor
            if x_cond and y_cond:
                target_img.putpixel((i, j), color)
    return target_img

def random_coordinates_list(num: int) -> list[tuple]:
    return [(np.random.randint(0, c.slm_width), np.random.randint(0, c.slm_height)) for _ in range(num)]