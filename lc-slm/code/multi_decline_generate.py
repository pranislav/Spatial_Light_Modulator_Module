import multi_decline_lib as md


object = md.rectangle
obj_size_x = 2
obj_size_y = 1
coord_generator = md.fract_position
spacing_param_a = 3 # num of objects on x axis for grating or x coordinate for 1-object position
spacing_param_b = 2 # num of objects on y axis for grating or y coordinate for 1-object position

name = f"multidecline_{coord_generator.__name__}_{spacing_param_a}x{spacing_param_b}_{object.__name__}_{obj_size_x}x{obj_size_y}"

coordinates = coord_generator(spacing_param_a, spacing_param_b)

img = md.multi_decline_img(coordinates, object, obj_size_x, obj_size_y)

img.save(f"images/{name}.jpg", quality=100)

# [(int(c.slm_width/3), int(c.slm_height/2))]