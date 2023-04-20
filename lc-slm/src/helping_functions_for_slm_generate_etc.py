import imageio
import PIL.Image as im
import os
import slm_screen as sc
import constants as c


def quarter(image: im) -> im:
    '''returns mostly blank image with original image pasted in upper-left corner
    when generated hologram for such a transformed image, there will be no overlap
    between different diffraction order of displayed image
    '''
    w, h = image.size
    image.resize((w // 2, h // 2))
    ground = im.new("L", (w, h))
    ground.paste(image)
    return ground


def transform_hologram(hologram, angle, focal_len):
    if angle:
        x_angle, y_angle = angle
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
