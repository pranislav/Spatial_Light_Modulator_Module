from pylablib.devices import uc480
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image as im
import calibration_lib as cl


# print(uc480.list_cameras())



cam = uc480.UC480Camera()

cl.set_exposure(cam)

frame = cam.snap()
# print(type(frame))

img = im.fromarray(frame)
img.show()
