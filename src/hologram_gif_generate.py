from hologram_gif_lib import generate_hologram_gif

# params for GD algorithm
learning_rate = 0.005
mask_relevance = 100
tolerance = 0.0025
max_loops = 20
unsettle = 0

# params for generator
source_path = "images/moving_traps/two_circulating_traps_quarterize_radius1px"
preview = True

generate_hologram_gif(source_path, "GS",  preview, learning_rate,
                      mask_relevance, tolerance, max_loops, unsettle)