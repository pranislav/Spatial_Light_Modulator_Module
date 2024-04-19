import numpy as np
from PIL import Image as im
from pylablib.devices import uc480
from calibration_lib import display_image_on_external_screen, set_exposure

def average_frames(hologram_path):
    cam = uc480.UC480Camera()
    display_image_on_external_screen(hologram_path)
    set_exposure((230, 250), cam)
    while True:
        num = input("enter number of frames to be averaged")
        frame = cam.snap()
        for _ in range(1, num):
            frame += cam.snap()
        im.fromarray(frame/num).show()
        if input("do you wish to adapt intensity? (y/N) "):
            adapt_intensity(cam)


def adapt_intensity(cam):
    where = input("increase (i) decrease (d)")
    if where == 'i':
      cam.set_exposure(cam.get_exposure() * 1.05)
    if where == 'd':
       cam.set_exposure(cam.get_exposure() * 1.05)