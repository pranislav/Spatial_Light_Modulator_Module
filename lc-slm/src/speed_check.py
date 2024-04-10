'''comparison of calculation speed for displaying hoograms
in case when i make set of calibration holograms without reference subdomain
so i need to add it when calibrating
i can have reference hologram as numpy array right away, but the actual hologram comes as image
'''

from time import time
from timeit import timeit
import numpy as np
from PIL import Image as im
import constants as c
from holograms_for_calibration import make_hologram


subdomain_size = 32
hologram = im.fromarray(make_hologram(np.zeros((c.slm_height, c.slm_width)), (0, 0)))
reference_coordinates = (608, 448)
reference_hologram_arr = make_hologram(np.zeros((c.slm_height, c.slm_width)), reference_coordinates)


def numpy_case():
    hologram_arr = np.array(hologram)
    return im.fromarray(hologram_arr + reference_hologram_arr)

def for_case():
    hologram_arr = np.array(hologram)
    i0, j0 = reference_coordinates
    for i in range(subdomain_size):
        for j in range(subdomain_size):
            hologram_arr[i0 + i, j0 + j] = reference_hologram_arr[i0 + i, j0 + j]
    return im.fromarray(hologram_arr)


def image_case():
    i0, j0 = reference_coordinates
    for i in range(subdomain_size):
        for j in range(subdomain_size):
            hologram.putpixel((i + i0, j + j0), reference_hologram_arr[i, j])
    return hologram


# t0 = time()
# numpy_case()
# t1 = time()
# for_case()
# t2 = time()
# image_case()
# t3 = time()

# print("numpy_case: ", t1 - t0)
# print("for_case: ", t2 - t1)
# print("image_case: ", t3 - t2)


n = 100
time1 = timeit(numpy_case, number=n) # 2.3
time2 = timeit(for_case, number=n) # 0.9
time3 = timeit(image_case, number=n) # 0.1

print("numpy_case: ", time1)
print("for_case: ", time2)
print("image_case: ", time3)

