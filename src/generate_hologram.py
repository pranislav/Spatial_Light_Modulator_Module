# TODO: there is a problem with plotting error

from algorithms import GS, GD
import numpy as np
from PIL import Image as im
import PIL.ImageOps
import constants as c
import argparse
import os
import imageio
import wavefront_correction_lib as cl
import time



def main(args):
    if args.img_name is None:
        hologram = np.zeros((c.slm_height, c.slm_width))
    else:
        hologram, expected_outcome = make_hologram(args)
    if args.preview:
        expected_outcome.show()
    hologram = transform_hologram(hologram, args)
    save_hologram_and_gif(hologram, args)


def make_hologram(args):
    algorithm = GS if args.algorithm == "GS" else GD
    target = prepare_target(args.img_name, args)
    add_gif_source_address(args)
    hologram, expected_outcome, _ = algorithm(target, args)
    return hologram, expected_outcome

def transform_hologram(hologram, args):
    if args.decline is not None:
        hologram = decline_hologram(hologram, args.decline, args.correspond_to2pi)
    if args.lens:
        hologram = add_lens(hologram, args.lens, args.correspond_to2pi)
    return hologram


def add_gif_source_address(args):
    if not args.gif:
        args.gif_dir = None
        return
    if args.gif_type == "h":
        args.gif_dir = "holograms"
    elif args.gif_type == "i":
        args.gif_dir = "images"
    if not os.path.exists(args.gif_dir):
        os.makedirs(args.gif_dir)
        

def prepare_target(img_name, args):
    target_img = im.open(f"images/{img_name}").convert('L').resize((int(c.slm_width), int(c.slm_height)))
    if args.invert:
        target_img = PIL.ImageOps.invert(target_img)
    if args.quarterize:
        target_img = quarter(target_img)
    return np.array(target_img)

def save_hologram_and_gif(hologram, args):
    img_name = os.path.basename(args.img_name).split(".")[0] if args.img_name else "analytical"
    dest_dir = args.destination_directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    hologram_name = make_hologram_name(args, img_name)
    im.fromarray(hologram).convert("L").save(f"{dest_dir}/{hologram_name}.png")
    if args.gif:
        create_gif(f"{args.gif_dir}/gif_source", f"{args.gif_dir}/{hologram_name}.gif")


def make_hologram_name(args, img_name):
    alg_params = ""
    transforms = ""
    img_transforms = ""
    if args.lens or args.decline:
        transforms += f"_decline_x{args.decline[0]}_y{args.decline[1]}_lens_{args.lens}"
    if args.algorithm == "GD":
        alg_params += f"_lr{args.learning_rate}_mr{args.mask_relevance}_unsettle{args.unsettle}"
    if args.quarterize:
        img_transforms += "_quarter"
    if args.invert:
        img_transforms += "_inverted"
    time_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    if args.img_name is None:
        return f"{img_name}{transforms}"
    return f"{img_name}{img_transforms}_{args.algorithm}{alg_params}__ct2pi{args.correspond_to2pi}_loops{args.max_loops}{transforms}_{time_name}"


def args_to_string(args):
    arg_string = ""
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            arg_string += f"{arg_name}_{arg_value}_"
    # Remove the trailing underscore
    arg_string = arg_string.rstrip('_')
    return arg_string

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


def decline_hologram(hologram: np.array, angle: tuple, correspond_to2pi: int=256):
    '''declines hologram by angle, returns declined hologram
    '''
    decline = cl.decline(angle, correspond_to2pi)
    declined_hologram = (hologram + decline) % correspond_to2pi
    return declined_hologram

def add_lens(hologram: np.array, focal_len: float, correspond_to2pi: int=256):
    return (hologram + lens(focal_len, correspond_to2pi, hologram.shape)) % correspond_to2pi

def lens(focal_length, correspond_to2pi, shape):
    '''simulates lens with focal length 'focal_length' in meters
    '''
    h, w = shape
    hologram = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            r = c.px_distance * np.sqrt((i - h / 2) ** 2 + (j - w / 2) ** 2)
            phase_shift = 2 * np.pi * focal_length / c.wavelength * \
                (1 - np.sqrt(1 + r ** 2 / focal_length ** 2))
            hologram[i, j] = (phase_shift * correspond_to2pi / (2 * np.pi)) % correspond_to2pi
    return hologram
    

def create_gif(img_dir, outgif_path):
    '''creates gif from images in img_dir
    and saves it as outgif_path
    '''
    with imageio.get_writer(outgif_path, mode='I') as writer:
        for file in os.listdir(img_dir):
            image = imageio.imread(f"{img_dir}/{file}")
            writer.append_data(image)

def remove_files_in_dir(dir_name):
    '''removes all files in a directory'''
    for file in os.listdir(dir_name):
        os.remove(f"{dir_name}/{file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_name", nargs="?", default=None, type=str, help="path to the target image from images directory. Leave empty if you want to create pure decline/lens hologram")
    parser.add_argument("-ii", "--incomming_intensity", type=str, default="uniform", help="path to the incomming intensity image from images directory or 'uniform' for uniform intensity")
    # "images/incomming_intensity_images/paper_shade_01_intensity_mask.png"
    parser.add_argument("-dest_dir", "--destination_directory", type=str, default="holograms", help="directory where the hologram will be saved")
    parser.add_argument("-q", "--quarterize", action="store_true", help="original image is reduced to quarter and pasted to black image of its original size ")
    parser.add_argument("-i", "--invert", action="store_true", help="invert colors of the target image")
    parser.add_argument("-alg", "--algorithm", default="GS", choices=["GS", "GD"], help="algorithm to use: GS for Gerchberg-Saxton, GD for gradient descent")
    parser.add_argument("-ct2pi", "--correspond_to2pi", required=True, metavar='INTEGER', type=int, help="color value corresponding to 2pi phase change on SLM")
    parser.add_argument("-tol", "--tolerance", default=0, metavar='FLOAT', type=float, help="algorithm stops when error descends under tolerance")
    parser.add_argument("-loops", "--max_loops", default=42, metavar='INTEGER', type=int, help="algorithm performs no more than max_loops loops no matter what error it is")
    parser.add_argument("-lr", "--learning_rate", default=0.005, type=float, help="learning rate for GD algorithm (how far the solution jumps in direction of the gradient)")
    parser.add_argument("-mr", "--mask_relevance", metavar='FLOAT', default=1, type=float, help="mask relevance for GD algorithm, sets higher priority to white areas by making error on those areas mask_relevance-times higher")
    parser.add_argument("-unsettle", default=0, metavar='INTEGER', type=int, help="unsettle for GD algorithm; learning rate is (unsettle - 1) times doubled")
    parser.add_argument("-gif", action="store_true", help="create gif from hologram computing evolution")
    parser.add_argument("-gif_t", "--gif_type", choices=["h", "i"], help="type of gif: h for hologram, i for image (result)")
    parser.add_argument("-gif_skip", default=1, type=int, metavar='INTEGER', help="each gif_skip-th frame will be in gif")
    parser.add_argument("-plot_error", action="store_true", help="plot error evolution")
    parser.add_argument("-p", "--preview", action="store_true", help="show expected outcome at the end of the program run")
    parser.add_argument("-decline", nargs=2, type=float, metavar=('X_ANGLE', 'Y_ANGLE'), default=None, help="add hologram for decline to computed hologram. Effect: shifts resulting image on Fourier plane by given angle (in units of quarter of first diffraction maximum)")
    parser.add_argument("-lens", default=None, type=float, metavar='FOCAL_LENGTH', help="add lens to hologram with given focal length in meters")
    args = parser.parse_args()
    args.print_info = True

    main(args)
