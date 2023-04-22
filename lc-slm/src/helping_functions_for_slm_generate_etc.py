import imageio
import PIL.Image as im
import os
import slm_screen as sc
import constants as c
import numpy as np
from PIL import ImageOps


def quarter(image: im) -> im:
    '''returns mostly blank image with original image pasted in upper-left corner
    when generated hologram for such a transformed image, there will be no overlap
    between different diffraction order of displayed image
    '''
    w, h = image.size
    resized = image.resize((w // 2, h // 2))
    ground = im.new("L", (w, h))
    ground.paste(resized)
    return ground


def transform_hologram(hologram, angle, focal_len):
    x_angle, y_angle = angle
    if x_angle or y_angle:
        return sc.Screen(hologram).decline('x', x_angle).decline('y', y_angle)
    if focal_len:
        return sc.Screen(hologram).lens(focal_len)
    else:
        return sc.Screen(hologram)
    

def create_gif(img_dir, outgif_path):
    '''creates gif from images in img_dir
    and saves it as outgif_path
    '''
    with imageio.get_writer(outgif_path, mode='I') as writer:
        for file in os.listdir(img_dir):
            image = imageio.imread(f"{img_dir}/{file}")
            writer.append_data(image)


def remove_files_in_dir(dir: str):
    '''removes all files in given directory'''
    for file in os.listdir(dir):
        os.remove(f"{dir}/{file}")


# function that takes all images from given file, call given function on them and saves them to different file
def process_images_in_dir(source_dir: str, dest_dir: str, func, *args):
    '''takes all images from given file, call given function on them and saves them to different file'''
    for file in os.listdir(source_dir):
        img = im.open(f"{source_dir}/{file}")
        img = func(img, *args)
        f_name, f_ext = os.path.splitext(file)
        img.save(f"{dest_dir}/{f_name}{func.__name__}{f_ext}")


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