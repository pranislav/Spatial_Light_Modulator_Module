import dmd
from dmd_constants import gamma, wavelength, b
import numpy as np

beta = dmd.alpha + 2*gamma
u = wavelength/b # unit: 1 diffraction maximum (approx for small angles)

# dmd.lens_plus_decline_x_img(1, beta).show()
dmd.decline_x_img(beta + u).show()
# dmd.my_scan(127).show()