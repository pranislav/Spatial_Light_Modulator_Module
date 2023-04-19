from algorithms import GS, GD, GD_uniphase as GDU, GS_pure as GSp, GD_uniphaseII as GDUII
import numpy as np
from PIL import Image as im
import PIL.ImageOps
import slm_screen as sc
import constants as c
import os
from create_gif import create_gif


# SETTINGS
# name and type of image which should be projected by SLM
target_name = "multidecline_fract_position_3x2_rectangle_5x5" # "multidecline_grating_2x2_circle5"
target_type = "jpg"
# other settings
invert = False
quarterize = False
algorithm = "GD"    # GD for gradient descent, GS for Gerchberg-Saxton
# transform parameters
x_decline = 0
y_decline = 0
unit = c.u # c.u for one quarter of 1st diff maximum, 1 for radians | ubiquity in filename - units not in the name
focal_len = False
# for GD:
mask_relevance = 10
unsettle = 0
learning_rate = 0.5
tolerance = 0.001
max_loops = 150
gif = 5

# loading image
target_img = im.open(f"images/{target_name}.{target_type}").convert('L').resize((int(c.slm_width), int(c.slm_height)))
if invert:
    target_img = PIL.ImageOps.invert(target_img)


def quarter(image: im) -> im:
    '''returns mostly blank image with original image pasted in upper-left corner
    when generated hologram for such a transformed image, there will be no overlap
    between different diffraction order of displayed image
    '''
    w, h = image.size
    image.resize(w / 2, h / 2)
    ground = im.new("L", (c.slm_width, c.slm_height))
    ground.paste(image)
    return ground

if quarterize:
    target_img = quarter(target_img)

target = np.sqrt(np.array(target_img))

enhance_mask = np.array(target_img) / 255 # normed to 1 | engance the error to get lower on light areas


# gif_source_address = f"holograms/gif_{target_name}"
# if gif and not os.path.exists(gif_source_address):
#         os.mkdir(gif_source_address)

# compouting phase distribution
if algorithm == "GS":
    source_phase_array, exp_tar_array, loops = GS(target, tolerance, max_loops, plot_error=True)
if algorithm == "GSp":
    source_phase_array, exp_tar_array = GSp(target, tolerance, max_loops, plot_error=True)

if algorithm == "GD":
    source_phase_array, exp_tar_array, loops = GD(target, learning_rate, enhance_mask,\
                                           mask_relevance, tolerance, max_loops, unsettle, plot_error=True, gif=gif, gif_source_address="holograms/gif_source")
if algorithm == "GDU":
    source_phase_array, exp_tar_array = GDU(target, learning_rate=0.5, tolerance=0.5, max_loops=100, plot_error=True)
if algorithm == "GDUII":
    source_phase_array, exp_tar_array = GDUII(target, learning_rate=100, tolerance=0.5, calib=2, angle=0, max_loops=1000, plot_error=True)


source_phase = im.fromarray(source_phase_array) # this goes into SLM
expected_target = im.fromarray(exp_tar_array)


def transform_hologram(hologram, angle, focal_len):
    if angle:
        x_angle, y_angle = angle
        return sc.Screen(hologram).decline('x', x_angle).decline('y', y_angle)
    if focal_len:
        return sc.Screen(hologram).lens(focal_len)
    else:
        return sc.Screen(hologram)

# transforming image
hologram = transform_hologram(source_phase, (x_decline*unit, y_decline*unit), focal_len)


# name of the hologram and saving
are_transforms = x_decline or y_decline or focal_len
if are_transforms:
    def u_name(unit):
        return "u" if unit==c.u else "rad"
    transforms = f"x={x_decline}{u_name(unit)}_y={y_decline}{u_name(unit)}_lens={focal_len}"
else:
    transforms = ""

if algorithm == "GD":
    alg_params = f"_leaning_rate={learning_rate}_mask_relevance={mask_relevance}_unsettle={unsettle}"
else:
    alg_params = ""

general_params = f"loops={loops}"

hologram_name = f"{target_name}_inverted={invert}_{transforms}_hologram_alg={algorithm}_{general_params}_{alg_params}"
hologram.img.convert("RGB").save(f"holograms/{hologram_name}.jpg", quality=100)

if gif:
    create_gif("holograms/gif_source", f"holograms/gif_{hologram_name}.gif")


# preview of results: what goes into SLM and what it should look like
# source_phase.show()
# expected_target.show()
