import multi_decline_lib as md
from constants import slm_height as h, slm_width as w

object = md.ellipse
obj_size_x = 8
obj_size_y = 12
coord_generator = md.fract_position
note = "half_and_third"
spacing_param_a = 3 # num of objects on x axis for grating or x coordinate for 1-object position
spacing_param_b = 3 # num of objects on y axis for grating or y coordinate for 1-object position

ud = coord_generator == md.user_defined
spacing_params = f"{spacing_param_a}x{spacing_param_b}"
name = f"multidecline_{coord_generator.__name__}_{note if ud else spacing_params}_{object.__name__}_{obj_size_x}x{obj_size_y}"

if ud:
    coordinates = [(w//2, h//2), (w//3, h//2)]
else:
    coordinates = coord_generator(spacing_param_a, spacing_param_b)

img = md.multi_decline_img(coordinates, object, obj_size_x, obj_size_y)

img.save(f"images/{name}.jpg", quality=100)

# [(int(c.slm_width/3), int(c.slm_height/2))]