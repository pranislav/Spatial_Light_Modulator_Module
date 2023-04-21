import numpy as np
import PIL.Image as im
from scipy.fft import ifft2

name = "multidecline_fract_position_3x3_ellipse_8x12"
img = im.open(f"images/{name}.jpg")
img = img.convert("L")
img_arr = np.array(img)
img_arr = np.sqrt(img_arr)
arr_out = (ifft2(img_arr))
phase_out = (np.angle(arr_out) + np.pi) * 255 / (2*np.pi)
img_out = im.fromarray(phase_out)
img_out.show()
