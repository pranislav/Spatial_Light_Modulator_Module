import numpy as np
import PIL.Image as im
from PIL import ImageOps


def trippy_coloring(img_path: str) -> None:
    clut_img = im.open('images/my_clut.png').resize((256, 1))

    clut_lst = [[] for _ in range(256)]
    for i in range(256):
        color = clut_img.getpixel((i, 0))
        for j in range(3):
            clut_lst[i].append(color[j])

    clut = np.array(clut_lst).reshape(-1, 3).T.flatten()

    img = im.open(img_path)
    img = img.point(clut)

    img.show()

# trippy_coloring("holograms/1over2_left_dot_5_hologram_x=0u_y=0u_lens=False_alg=GD_invert=False_mask_relevance=0_unsettle=3.jpg")

def dipolar_coloring(img_path: str, black: tuple[int], white: tuple[int]) -> None:
    img = im.open(img_path).convert('L')
    img_out = ImageOps.colorize(img, black, white)
    img_out.show()
    

name = "holograms/1over2_left_dot_5_hologram_x=0u_y=0u_lens=False_alg=GD_invert=False_mask_relevance=0_unsettle=3.jpg"
blue = (0, 38, 126)
red = (255, 0, 0)
dark_blue = (0, 100, 100)
tyrquoise = (163, 255, 255)
white = (255, 255, 255)
dipolar_coloring(name, dark_blue, red)