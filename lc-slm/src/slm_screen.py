# TODO: change from the root. this is too expensive. no getpixel, putpixel, work with arrays and convert to image just at the end
# also, that cf thing is confusing, moreover, i divide it bu pi and right then multiply by it
# also, x_decline and y_decline can be done in one function (yes, function, forget about your classy class) - get inspired with holograms_for_calibration
'''slm - module for work with Spatial Light Modulator

this module primarly provides functions which generate images
for slm screen.
It contains function for deflecting ray,
function, which stimulates lens
'''


from __future__ import annotations
import numpy as np
from PIL import Image as im
from distutils.log import error
import constants as c

pi = np.pi


# parameters
x_size = c.slm_width
y_size = c.slm_height
a = c.px_distance
conversion_factor = 255 / (2*pi)    # converts from rad to greyscale number
cf = conversion_factor


class Screen:
    '''extension of class Image'''

    def __init__(self, img: im = im.new('L', (x_size, y_size))):
        self.img = img

    def invert(self):
        x = np.array(self.img)
        return Screen(im.fromarray((-x) % 255))

    def decline(self, axis: str, angle: float) -> Screen:
        '''deflects ray in axis 'axis' by angle 'angle'

        parameters
        ----------
        axis: {'x', 'y'}
            axis in which the deflection is made
        angle: angle of deflection in radians
        '''

        if axis != 'x' and axis != 'y':
            error('axis must be x or y')
        for i in range(self.img.width):
            for j in range(self.img.height):
                step = i if axis == 'x' else j
                phase_shift = 2*pi*a*np.sin(angle) / c.wavelength * step
                shade = int(
                    (self.img.getpixel((i, j)) + phase_shift * cf) % 256)
                self.img.putpixel((i, j), shade)
        return self

    def lens(self, focal_length: float) -> Screen:
        '''simultes lens with focal length 'focal_length' in meters
        '''

        f = focal_length
        w, h = self.img.width, self.img.height
        for i in range(w):
            for j in range(h):
                r = a * np.sqrt((i - w/2)**2 + (j - h/2)**2)
                phase_shift = 2*pi*f / c.wavelength * \
                    (1 - np.sqrt(1 + r**2/f**2))
                shade = int(
                    (self.img.getpixel((i, j)) + phase_shift * cf) % 255)
                self.img.putpixel((i, j), shade)
        return self

    def axicon(self, angle: float) -> Screen:
        '''simulates axicon, a special type of lense with conical surface
        param `angle` refers to local beam deflection, not conic slope'''

        w, h = self.img.width, self.img.height
        for i in range(w):
            for j in range(h):
                r = a * np.sqrt((i - w/2)**2 + (j - h/2)**2)
                phase_shift = 2*pi*np.sin(angle) / c.wavelength * r
                shade = int(
                    (self.img.getpixel((i, j)) + phase_shift * cf) % 255)
                self.img.putpixel((i, j), shade)
        return self

    def my_dmd_trafo(self) -> Screen:
        '''transformates the image so that it appears as original
        at the DMD screen (but rotated by pi/2 rad)'''

        # self_copy = self.img.copy()

        w, h = self.img.width, self.img.height
        new = Screen(im.new('L', (w, h)))
        for i in range(w):
            for j in range(h):
                new_i = i + (j - h + 1) // 2
                new_j = - i + (j + h) // 2
                if 0 < new_i < w and 0 < new_j < h:
                    shade = self.img.getpixel((i, j))
                    new.img.putpixel((new_i, new_j), shade)
        return new
