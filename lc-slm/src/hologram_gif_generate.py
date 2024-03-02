from hologram_gif_generator import generate_hologram_gif

# params for GD algorithm
learning_rate = 0.002
mask_relevance = 10
tolerance = 0.001
max_loops = 20
unsettle = 0

# params for generator
source_path = "images/moving_traps/moving_traps_4"
preview = True

generate_hologram_gif(source_path, preview, learning_rate,
                      mask_relevance, tolerance, max_loops, unsettle)