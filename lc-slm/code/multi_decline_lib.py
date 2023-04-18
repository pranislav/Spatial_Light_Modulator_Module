from __future__ import annotations
import PIL.Image as im
import constants as c

white = 255

def multi_decline_img(coordinates: list[tuple], object: function, size_x: float, size_y: float) -> im:
    img = im.new('L', (c.slm_width, c.slm_height))
    for point in coordinates:
        object(img, point, size_x, size_y, white)
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

def fract_position(x_fract, y_fract):
    return [int(c.slm_width / x_fract), int(c.slm_height / y_fract)]

def dec_position(x_dec, y_dec):
    return [(int(c.slm_width * x_dec), int(c.slm_height * y_dec))]


# outdated, not compatible; use more general function 'ellipse'
def circle(target_img: im, coor: tuple[int], radius: float, color: int) -> None:
    x_coor, y_coor = coor
    w = c.slm_width
    h = c.slm_height
    for i in range(h):
        for j in range(w):
            if (i - x_coor)**2 + (j - y_coor)**2 < radius**2:
                target_img.putpixel((i, j), color)
    return target_img


def ellipse(target_img: im, coor: tuple[int], x_d: float, y_d: float, color: int) -> None:
    x_coor, y_coor = coor
    w, h = target_img.size
    for i in range(h):
        for j in range(w):
            if ((i - x_coor) / x_d)**2 + ((j - y_coor) / y_d)**2 < 1:
                target_img.putpixel((i, j), color)
    return target_img


# outdated, not compatible; use more general function 'rectangle'
def square(target_img: im, coor: tuple[int], side: float, color: int):
    x_coor, y_coor = coor
    w, h = target_img.size
    for i in range(h):
        for j in range(w):
            x_cond = (x_coor - side // 2) < i <= (x_coor + side // 2)
            y_cond = (y_coor - side // 2) < j <= (y_coor + side // 2)
            if x_cond and y_cond:
                target_img.putpixel((i, j), color)
    return target_img

def rectangle(target_img: im, coor: tuple[int], side_x: float, side_y: float, color: int):
    x_coor, y_coor = coor
    w, h = target_img.size
    for i in range(h):
        for j in range(w):
            x_cond = (x_coor - side_x // 2) < i <= (x_coor + side_x // 2)
            y_cond = (y_coor - side_y // 2) < j <= (y_coor + side_y // 2)
            if x_cond and y_cond:
                target_img.putpixel((i, j), color)
    return target_img