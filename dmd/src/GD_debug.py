from PIL import Image as im
from pseudoGD import *
from dmd_constants import i_size, j_size
import numpy as np
import matplotlib.pyplot as plt

def complex_to_real_phase(complex_phase):
    return (np.angle(complex_phase) + np.pi) / (2*np.pi) * 255

# img = im.fromarray(complex_to_real_phase(phase_profile_diagonal(i_size, j_size)))
# img.show()


my_image = im.open('images/lena.png').resize((i_size, j_size)).convert('L')
my_image_arr = np.array(my_image)

hologram, exp_tar_for_dmd, error_evolution = discrete_GD(my_image_arr, phase_profile_diagonal(i_size, j_size))

plt.plot(error_evolution)
plt.show()

img = im.fromarray((hologram * 255).astype(np.uint8))
img.show()

img2 = im.fromarray(exp_tar_for_dmd)
img2.show()

