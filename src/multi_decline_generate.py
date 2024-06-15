#TODO: redo this

import multi_decline_lib as md
from constants import slm_height as h, slm_width as w

object = md.dot
obj_size_x = 5
obj_size_y = 5
coord_generator = md.user_defined
note = "two_dots_quarterized"
spacing_param_a = 2 # num of objects on x axis for grating or x coordinate for 1-object position
spacing_param_b = 2 # num of objects on y axis for grating or y coordinate for 1-object position

ud = coord_generator == md.user_defined
spacing_params = f"{spacing_param_a}x{spacing_param_b}"
if object == md.dot:
    name = f"multidecline_{coord_generator.__name__}_{spacing_params}_{object.__name__}"
elif object == md.user_defined:
    name = f"multidecline_{coord_generator.__name__}_note_{object.__name__}_{obj_size_x}x{obj_size_y}"
else:
    name = f"multidecline_{coord_generator.__name__}_{spacing_params}_{object.__name__}_{obj_size_x}x{obj_size_y}"

if ud:
    coordinates = [(w//6, h//4), (2*w//6, h//4)]
else:
    coordinates = coord_generator(spacing_param_a, spacing_param_b)

img = md.multi_decline_img(coordinates, object, obj_size_x, obj_size_y)

img.save(f"images/{name}.png")

# [(int(c.slm_width/3), int(c.slm_height/2))]