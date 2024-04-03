# TODO: there is a problem with plotting error

from algorithms import GS, GD
import numpy as np
from PIL import Image as im
import PIL.ImageOps
import slm_screen as sc
import constants as c
from structs import gif_struct
import helping_functions_for_slm_generate_etc as hf



# SETTINGS
# name and type of image which should be projected by SLM
target_name = "multidecline_grating_2x2_dot" # "moving_traps/two_circulating_traps_radius1px/3" # "multidecline_user_defined_5432_dot_2x2"
target_type = "png"
# ...
save_result = True
preview = False
plot_error = True
# other settings
invert = False
quarterize = True # original image is reduced to quarter and pasted to black image of its original size | helpful when imaging - there is no overlap between diffraction maxima of different order
algorithm = "GD"    # GD for gradient descent, GS for Gerchberg-Saxton
# stopping parameters
tolerance = 0.001 # algorithm stops when error descends under tolerance
max_loops = 50 # algorithm performs no more than max_loops loops no matter what error it is
# transform parameters
x_decline = 0
y_decline = 0
unit = c.u # c.u for one quarter of 1st diff maximum, 1 for radians | ubiquity in filename - units not in the name
focal_len = False
# for GD:
learning_rate = 0.005 # how far our solution jump in direction of the gradient. Too low - slow convergence; too high - oscilations or even none reasonable improvement at all
mask_relevance = 100 # very helpful when target is predominantly black (multidecline dots)
unsettle = 0 # learning rate is (unsettle - 1) times doubled. it may improve algorithm performance, and it also may cause peaks in error evolution
# gif creation
gif_target = "" # "h" for hologram, "i" for image (result) and empty string for no gif
gif_skip = 2 # each gif_skip-th frame will be in gif



# loading image and creating array target
target_img = im.open(f"lc-slm/images/{target_name}.{target_type}").convert('L').resize((int(c.slm_width), int(c.slm_height)))
if invert:
    target_img = PIL.ImageOps.invert(target_img)
if quarterize:
    target_img = hf.quarter(target_img)
target = np.sqrt(np.array(target_img))


enhance_mask = np.array(target_img) / 255 # normed to 1 | enhance the error to get lower on light areas

# creating gif data structure (primarily for GD arguments reducing)
gif = gif_struct()
gif.type = gif_target
gif.skip_frames = gif_skip

if gif_target:
    directory = "images" if gif_target == "i" else "holograms"
    gif.source_address = f"{directory}/gif_source"
    # making place for gif images
    hf.remove_files_in_dir(gif.source_address)


# compouting phase distribution
if algorithm == "GS":
    source_phase_array, exp_tar_array, loops = GS(target, tolerance, max_loops, gif, plot_error)

if algorithm == "GD":
    source_phase_array, exp_tar_array, loops = GD(target, learning_rate, enhance_mask,\
                    mask_relevance, tolerance, max_loops, unsettle, gif, plot_error)


source_phase = im.fromarray(source_phase_array) # this goes into SLM
expected_target = im.fromarray(exp_tar_array)


are_transforms = x_decline or y_decline or focal_len
if are_transforms:
    hologram = hf.transform_hologram(source_phase, (x_decline*unit, y_decline*unit), focal_len)
    def u_name(unit):
        return "u" if unit==c.u else "rad"
    transforms = f"x={x_decline}{u_name(unit)}_y={y_decline}{u_name(unit)}_lens={focal_len}"
else:
    hologram = sc.Screen(source_phase)
    transforms = ""

if algorithm == "GD":
    alg_params = f"_leaning_rate={learning_rate}_mask_relevance={mask_relevance}_unsettle={unsettle}"
else:
    alg_params = ""

target_transforms = f"inverted={invert}_quarter={quarterize}"
general_params = f"loops={loops}"

if save_result:
    hologram_name = f"{target_name}_{target_transforms}_{transforms}_hologram_alg={algorithm}_{general_params}_{alg_params}"
    hologram.img.convert("RGB").save(f"lc-slm/holograms/{hologram_name}.png")
    expected_target.convert("RGB").save(f"lc-slm/images/{hologram_name}_exp_tar.png")

if gif_target:
    hf.create_gif(gif.source_address, f"{directory}/gif_{hologram_name}.gif")

# preview of results: what goes into SLM and what it should look like
if preview:
    source_phase.show()
    expected_target.show()
