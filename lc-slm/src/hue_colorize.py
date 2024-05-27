import os
import PIL.Image as im
import numpy as np
from PIL import ImageOps


def clut_coloring(img: im, clut: list) -> im:
    img = img.point(clut)
    return img


def dipolar_coloring(img: im, black: tuple[int], white: tuple[int]) -> im:
    img.convert('L')
    img_out = ImageOps.colorize(img, black, white)
    return img_out

# TODO: do not make clut every time, but save it somewhere
def make_clut(source_img_path: str) -> list:
    clut_img = im.open(source_img_path).resize((256, 1))
    clut_lst = [[] for _ in range(256)]
    for i in range(256):
        color = clut_img.getpixel((i, 0))
        for j in range(3):
            clut_lst[i].append(color[j])
    clut = np.array(clut_lst).reshape(-1, 3).T.flatten()
    return clut

# function that takes all images from given file, call given function on them and saves them to different file
def process_images_in_dir(source_dir: str, dest_dir: str, func, *args):
    '''takes all images from given file, call given function on them and saves them to different file'''
    for file in os.listdir(source_dir):
        img = im.open(f"{source_dir}/{file}")
        img = func(img, *args)
        f_name, f_ext = os.path.splitext(file)
        img.save(f"{dest_dir}/{f_name}{func.__name__}{f_ext}")


source_dir = "holograms/pre_tyca"


if __name__ == "__main__":
    process_images_in_dir("holograms/pre_tyca", "holograms/pre_tyca", clut_coloring, make_clut("images/my_clut.png"))
