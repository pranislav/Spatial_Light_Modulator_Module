import numpy as np
import PIL.Image as im
import constants as c

white = 255
halfwidth = (c.slm_height // 2)

name = "multidecline_row_2"


def multi_decline_img(coordinates: list[tuple]) -> im:
    img = im.new('L', (c.slm_width, c.slm_height))
    for point in coordinates:
        img.putpixel(point, white)
    return img


def dots_in_row(num: int, vertical_coor: int) -> list[tuple]:
    coordinates = []
    dist = int(c.slm_width / (num + 1))
    for i in range(num):
        coordinates.append(((i + 1) * dist, vertical_coor))
    return coordinates


coordinates = dots_in_row(2, halfwidth)
img = multi_decline_img(coordinates)

img.save(f"images/{name}.jpg", quality=100)