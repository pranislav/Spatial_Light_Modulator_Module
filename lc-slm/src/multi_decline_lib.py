from __future__ import annotations
import PIL.Image as im
import constants as c
import os

white = 255

def multi_decline_img(coordinates: list[tuple], object: function, size_x: float, size_y: float) -> im:
    img = im.new('L', (c.slm_width, c.slm_height))
    for point in coordinates:
        object(img, point, size_x, size_y, white)
    return img


trap_radius = 10

'''creates a sequence of images of multiple moving points
from given lists of coordinates
and saves them into a file'''
def create_sequence_dots(list_of_position_lists: list[list[tuple]], name: str):
    if not os.path.exists(f"images/moving_traps/{name}"):
        os.makedirs(f"images/moving_traps/{name}")
    for i, position_list in enumerate(list_of_position_lists):
        img = multi_decline_img(position_list, ellipse, trap_radius, trap_radius)
        img.save(f"images/moving_traps/{name}/{i}.png", quality=100)


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

# outdated, not compatible; use more general function 'ellipse'
def circle(target_img: im, coor: tuple[int], radius: float, color: int) -> None:
    x_coor, y_coor = coor
    w, h = target_img.size
    for i in range(w):
        for j in range(h):
            if (i - x_coor)**2 + (j - y_coor)**2 < radius**2:
                target_img.putpixel((i, j), color)
    return target_img


def ellipse(target_img: im, coor: tuple[int], x_d: float, y_d: float, color: int) -> None:
    x_coor, y_coor = coor
    w, h = target_img.size
    for i in range(w):
        for j in range(h):
            if ((i - x_coor) / x_d)**2 + ((j - y_coor) / y_d)**2 < 1:
                target_img.putpixel((i, j), color)
    return target_img



# outdated, not compatible; use more general function 'rectangle'
def square(target_img: im, coor: tuple[int], side: float, color: int):
    x_coor, y_coor = coor
    w, h = target_img.size
    for i in range(w):
        for j in range(h):
            x_cond = (x_coor - side // 2) < i <= (x_coor + side // 2)
            y_cond = (y_coor - side // 2) < j <= (y_coor + side // 2)
            if x_cond and y_cond:
                target_img.putpixel((i, j), color)
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
