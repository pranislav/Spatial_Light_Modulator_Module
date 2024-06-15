'''slm - module for work with Spatial Light Modulator

this module mainly provides functions which generate images
for slm screen.
It contains function for deflecting ray,
function, which stimulates lens
'''


from distutils.log import error
import numpy as np
from PIL import Image as im
pi = np.pi


# slm parameters settings
a = 3.6e-8    # meters
x_size = 1024    # pixels
y_size = 768    # pixels


### GENERAL FUNCTIONS AND CONSTANTS ###

wavelength = 5.32e-7    # meters
conversion_factor = 255 / (2*pi)    # converts from rad to color
cf = conversion_factor

def clear_screen(x_size: int, y_size: int) -> im:
    return im.new('L', (x_size, y_size), 0)


### DEFLECTING LIGHT RAY ###


def deflection(original_image: im, axis: str, angle: float) -> im:
    if axis != 'x' and axis != 'y': error('axis must be x or y')
    img = original_image
    for i in range(img.width):
        for j in range(img.height):
            step = i if axis == 'x' else j
            phase_shift = 2*pi*a*np.sin(angle) / wavelength * step
            color = int((img.getpixel((i, j)) + phase_shift * cf) % 255)
            img.putpixel((i, j), color)
    return img

