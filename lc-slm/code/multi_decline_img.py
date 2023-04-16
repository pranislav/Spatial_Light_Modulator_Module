from __future__ import annotations
import numpy as np
import PIL.Image as im
import constants as c


white = 255
obj_size = 1

# name = f"multidecline_grating_1x2_circle{obj_size}"
name = "left_dot_5"


def multi_decline_img(coordinates: list[tuple], object: function, size: float) -> im:
    img = im.new('L', (c.slm_width, c.slm_height))
    for point in coordinates:
        object(img, point, size, white)
    return img


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


def circle(target_img: im, coor: tuple[int], diam: float, color: int) -> None:
    x_coor, y_coor = coor
    w = c.slm_width
    h = c.slm_height
    for i in range(h):
        for j in range(w):
            if (i - x_coor)**2 + (j - y_coor)**2 < diam**2:
                target_img.putpixel((i, j), color)
    return target_img

def square():
    return


coordinates = grating(2, 1)
img = multi_decline_img([(int(c.slm_width/3), int(c.slm_height/2))], circle, obj_size)

img.save(f"images/{name}.jpg", quality=100)