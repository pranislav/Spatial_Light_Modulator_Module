'''write a binary Gerchberg-Saxton algorithm
which creates a hologram for DMD
with screen size i_size x j_size (from dmd_constants.py).
Input for this algorithm is picture of size same as the hologram.
output is the hologram picture.
Hologram  should consist just of binary values (black and white)
and its fourier transform should be the input picture.
initial guess can be fourier transform of an input image.
'''
    
import numpy as np
from PIL import Image as im
from dmd_constants import i_size, j_size, wavelength
from dmd import conversion
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift



