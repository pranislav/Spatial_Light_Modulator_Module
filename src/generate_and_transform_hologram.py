# TODO: there is a problem with plotting error

from algorithms import GS, GD
import numpy as np
from PIL import Image as im
import PIL.ImageOps
# import slm_screen as sc
import constants as c
import argparse
import os
import imageio
import wavefront_correction_lib as cl
import analytical_holograms as ah
import time



# # SETTINGS
# # name and type of image which should be projected by SLM
# target_name = "multidecline_grating_1x1_dot" # "moving_traps/two_circulating_traps_radius1px/3" # "multidecline_user_defined_5432_dot_2x2"
# target_type = "png"
# #
# path_to_incomming_intensity = "lc-slm/images/incomming_intensity_images/paper_shade_01_intensity_mask.png"
# # ...
# save_result = True
# preview = False
# plot_error = True
# # other settings
# invert = False
# quarterize = True # original image is reduced to quarter and pasted to black image of its original size | helpful when imaging - there is no overlap between diffraction maxima of different order
# algorithm = "GS"    # GD for gradient descent, GS for Gerchberg-Saxton
# # stopping parameters
# tolerance = 0.0001 # algorithm stops when error descends under tolerance
# max_loops = 42 # algorithm performs no more than max_loops loops no matter what error it is
# # transform parameters
# x_decline = 0
# y_decline = 0
# unit = c.u # c.u for one quarter of 1st diff maximum, 1 for radians | ubiquity in filename - units not in the name
# focal_len = False
# # for GD:
# learning_rate = 0.005 # how far our solution jump in direction of the gradient. Too low - slow convergence; too high - oscilations or even none reasonable improvement at all
# mask_relevance = 100 # very helpful when target is predominantly black (multidecline dots)
# unsettle = 0 # learning rate is (unsettle - 1) times doubled. it may improve algorithm performance, and it also may cause peaks in error evolution
# # gif creation
# gif_target = "" # "h" for hologram, "i" for image (result) and empty string for no gif
# gif_skip = 2 # each gif_skip-th frame will be in gif

# correspond_to2pi = 256 # color value corresponding to 2pi phase change on SLM



# # loading image and creating array target
# target_img = im.open(f"lc-slm/images/{target_name}.{target_type}").convert('L').resize((int(c.slm_width), int(c.slm_height)))
# if invert:
#     target_img = PIL.ImageOps.invert(target_img)
# if quarterize:
#     target_img = quarter(target_img)
# target = np.array(target_img)


# enhance_mask = target / 255 # normed to 1 | enhance the error to get lower on light areas

# # creating gif data structure (primarily for GD arguments reducing)
# gif = gif_struct()
# gif.type = gif_target
# gif.skip_frames = gif_skip

# if gif_target:
#     directory = "images" if gif_target == "i" else "holograms"
#     gif.source_address = f"{directory}/gif_source"
#     # making place for gif images
#     remove_files_in_dir(gif.source_address)


# # compouting phase distribution
# if algorithm == "GS":
#     source_phase_array, exp_tar_array, loops = GS(target, path_to_incomming_intensity, tolerance, max_loops, gif, plot_error, correspond_to2pi)

# if algorithm == "GD":
#     source_phase_array, exp_tar_array, loops = GD(target, path_to_incomming_intensity, learning_rate, enhance_mask,\
#                     mask_relevance, tolerance, max_loops, unsettle, gif, plot_error, correspond_to2pi)


# source_phase = im.fromarray(source_phase_array) # this goes into SLM
# expected_target = im.fromarray(exp_tar_array)


# are_transforms = x_decline or y_decline or focal_len
# if are_transforms:
#     hologram = transform_hologram(source_phase, (x_decline*unit, y_decline*unit), focal_len)
#     def u_name(unit):
#         return "u" if unit==c.u else "rad"
#     transforms = f"x={x_decline}{u_name(unit)}_y={y_decline}{u_name(unit)}_lens={focal_len}"
# else:
#     hologram = sc.Screen(source_phase)
#     transforms = ""

# if algorithm == "GD":
#     alg_params = f"_learning_rate={learning_rate}_mask_relevance={mask_relevance}_unsettle={unsettle}"
# else:
#     alg_params = ""

# target_transforms = f"inverted={invert}_quarter={quarterize}"
# general_params = f"loops={loops}_ct2pi={correspond_to2pi}"

# if save_result:
#     hologram_name = f"{target_name}_{target_transforms}_{transforms}_hologram_alg={algorithm}_{general_params}_{alg_params}"
#     hologram.img.convert("L").save(f"lc-slm/holograms/{hologram_name}.png")
#     expected_target.convert("L").save(f"lc-slm/images/{hologram_name}_exp_tar.png")

# if gif_target:
#     create_gif(gif.source_address, f"{directory}/gif_{hologram_name}.gif")

# # preview of results: what goes into SLM and what it should look like
# if preview:
#     source_phase.show()
#     expected_target.show()


# TODO: make algs eat nonsqrted target image and return just hologram as im object
# TODO: independent function for visualizing expected image
# what about final loop number?


def main(args):
    hologram, expected_outcome = make_hologram(args)
    if args.preview:
        expected_outcome.show()
    hologram = transform_hologram(hologram, args)
    save_hologram_and_gif(hologram, args)



def make_hologram(args):
    algorithm = GS if args.algorithm == "GS" else GD
    target = prepare_target(args.img_name, args)
    args.path_to_incomming_intensity = "lc-slm/images/incomming_intensity_images/paper_shade_01_intensity_mask.png"
    add_gif_source_address(args)
    hologram, expected_outcome = algorithm(target, args)
    return hologram, expected_outcome

def transform_hologram(hologram, args):
    if args.decline != (0, 0):
        hologram = decline_hologram(hologram, args.decline, args.correspond_to2pi)
    if args.lens:
        hologram = add_lens(hologram, args.lens, args.correspond_to2pi)
    return hologram


def add_gif_source_address(args):
    if not args.gif:
        args.gif_dir = None
        return
    if args.gif_type == "h":
        args.gif_dir = "lc-slm/holograms"
    elif args.gif_type == "i":
        args.gif_dir = "lc-slm/images"
    if not os.path.exists(args.gif_dir):
        os.makedirs(args.gif_dir)
        

def prepare_target(img_name, args):
    target_img = im.open(f"lc-slm/images/{img_name}").convert('L').resize((int(c.slm_width), int(c.slm_height)))
    if args.invert:
        target_img = PIL.ImageOps.invert(target_img)
    if args.quarterize:
        target_img = quarter(target_img)
    return np.array(target_img)

def save_hologram_and_gif(hologram, args):
    img_name = os.path.basename(args.img_name).split(".")[0]
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
    if args.lens or args.decline != (0, 0):
        transforms += f"_decline_x{args.decline[0]}_y{args.decline[1]}_lens_{args.lens}"
    if args.algorithm == "GD":
        alg_params += f"_lr{args.learning_rate}_mr{args.mask_relevance}_unsettle{args.unsettle}"
    if args.quarterize:
        img_transforms += "_quarter"
    if args.invert:
        img_transforms += "_inverted"
    time_name = time.strftime("%Y-%m-%d_%H-%M-%S")
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


# def transform_hologram(hologram, angle, focal_len):
#     x_angle, y_angle = angle
#     hologram_screen = sc.Screen(hologram)
#     if x_angle:
#         hologram_screen.decline('x', x_angle)
#     if y_angle:
#         hologram_screen.decline('y', y_angle)
#     if focal_len:
#         hologram_screen.lens(focal_len)
#     return hologram_screen

def decline_hologram(hologram: np.array, angle: tuple, correspond_to2pi: int=256):
    '''declines hologram by angle, returns declined hologram
    '''
    decline = cl.decline(angle, correspond_to2pi)
    declined_hologram = (hologram + decline) % correspond_to2pi
    return declined_hologram

def add_lens(hologram: np.array, focal_len: float, correspond_to2pi: int=256):
    return (hologram + ah.lens(focal_len, correspond_to2pi, hologram.shape)) % correspond_to2pi
    

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_name", type=str, help="path to the target image from lc-slm/images directory")
    parser.add_argument("-dest_dir", "--destination_directory", type=str, default="lc-slm/holograms", help="directory where the hologram will be saved")
    parser.add_argument("-q", "--quarterize", action="store_true", help="original image is reduced to quarter and pasted to black image of its original size ")
    parser.add_argument("-i", "--invert", action="store_true", help="invert colors of the target image")
    parser.add_argument("-alg", "--algorithm", default="GS", choices=["GS", "GD"], help="algorithm to use: GS for Gerchberg-Saxton, GD for gradient descent")
    parser.add_argument("-ct2pi", "--correspond_to2pi", default=256, type=int, help="color value corresponding to 2pi phase change on SLM")
    parser.add_argument("-tol", "--tolerance", default=0, type=float, help="error tolerance")
    parser.add_argument("-loops", "--max_loops", default=42, type=int, help="maximum number of loops")
    parser.add_argument("-lr", "--learning_rate", default=0.005, type=float, help="learning rate for GD algorithm (how far the solution jumps in direction of the gradient)")
    parser.add_argument("-mr", "--mask_relevance", default=100, type=float, help="mask relevance for GD algorithm, sets higher priority to white areas by making error on those areas mask_relevance-times higher")
    parser.add_argument("-unsettle", default=0, type=int, help="unsettle for GD algorithm; learning rate is (unsettle - 1) times doubled")
    parser.add_argument("-gif", action="store_true", help="create gif from hologram computing evolution")
    parser.add_argument("-gif_t", "--gif_type", choices=["h", "i"], help="type of gif: h for hologram, i for image (result)")
    parser.add_argument("-gif_skip", default=1, type=int, help="each gif_skip-th frame will be in gif")
    parser.add_argument("-plot_error", action="store_true", help="plot error evolution")
    parser.add_argument("-p", "--preview", action="store_true", help="show expected outcome at the end of the program run")
    parser.add_argument("-decline", nargs=2, default=(0, 0), help="decline hologram by given angle in units of quarter of first diffraction maximum ")
    parser.add_argument("-lens", default=None, type=float, help="add lens to hologram with given focal length in meters")
    args = parser.parse_args()

    main(args)
