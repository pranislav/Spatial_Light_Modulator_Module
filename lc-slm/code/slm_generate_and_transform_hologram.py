from algorithms import GS, GD, GD_uniphase as GDU, GS_pure as GSp, GD_uniphaseII as GDUII
import numpy as np
from PIL import Image as im
import PIL.ImageOps
import slm_screen as sc
import constants as c


# SETTINGS
# name and type of image which should be projected by SLM
target_name = "1over2_left_dot_1" # "multidecline_grating_2x2_circle5"
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
unsettle = 5

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


# compouting phase distribution
if algorithm == "GS":
    source_phase_array, exp_tar_array = GS(target, tolerance=0.005, max_loops=200, plot_error=True)
if algorithm == "GSp":
    source_phase_array, exp_tar_array = GSp(target, tolerance=0.5, max_loops=200, plot_error=True)

if algorithm == "GD":
    source_phase_array, exp_tar_array = GD(target, learning_rate=0.5, enhance_mask=enhance_mask,\
                                           mask_relevance=mask_relevance, tolerance=0.000001, max_loops=500, unsettle=unsettle, plot_error=True)
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

def u_name(unit):
    return "u" if unit==c.u else "rad"

# transforming image
hologram = transform_hologram(source_phase, (x_decline*unit, y_decline*unit), focal_len)
# hologram.img.show()
if algorithm == "GD":
    hologram.img.convert("RGB").save(f"holograms/{target_name}_hologram_x={x_decline}{u_name(unit)}_y={y_decline}{u_name(unit)}_lens={focal_len}_alg={algorithm}_invert={invert}_mask_relevance={mask_relevance}_unsettle={unsettle}.jpg", quality=100)
else:
    hologram.img.convert("RGB").save(f"holograms/{target_name}_hologram_x={x_decline}{u_name(unit)}_y={y_decline}{u_name(unit)}_lens={focal_len}_alg={algorithm}_invert={invert}.jpg", quality=100)

# preview of results: what goes into SLM and what it should look like
source_phase.show()
expected_target.show()
