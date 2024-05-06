'''script that experimentally determines the relationship between
a value of a pixel in a hologram and real phase shift that SLM applies on the pixel
in contrast to the other one, here there are just two subdomains placed in a way
that intensity is equally divided between them
subdomains are as big as possible* so intensity evolution should be very smooth

* for case when incomming intensity is circle inscribed in slm's display'''


import explore_calibration as e
import calibration_lib as cl
import constants as c
import numpy as np
import argparse
from pylablib.devices import uc480



def main(precision):
    cam = uc480.UC480Camera()
    window = cl.create_tk_window()
    subdomain_size = int(np.sqrt(4 / 5) * c.slm__height)
    sample_list = e.make_sample_holograms((1, 1), precision)
    upper_left_corner = np.array((c.slm_width // 2 - subdomain_size, (c.slm_height - subdomain_size) // 2))
    reference = cl.add_subdomain(cl.black_hologram, sample_list[0], upper_left_corner, subdomain_size)
    hologram_set = make_hologram_set(reference, sample_list, upper_left_corner + (subdomain_size, 0))
    cl.set_exposure_wrt_reference_img(cam, window, (220, 250), hologram_set[0], 8)
    intensity_coords = cl.get_highest_intensity_coordinates_img(cam, window, hologram_set[0], 8)
    intensity_list = [[], []]
    k = 0
    while k < precision:
        frame = cam.snap()
        intensity = cl.get_intensity_coordinates(frame, intensity_coords)
        if intensity == 255:
            print("maximal intensity was reached, adapting...")
            cam.set_exposure(cam.get_exposure() * 0.9) # 10 % decrease of exposure time
            k = 0
            intensity_list = [[], []]
            continue
        phase = k * 256 // precision
        intensity_list[0].append(phase)
        intensity_list[1].append(intensity)
    fit_params = 



def make_hologram_set(reference, sample_list, coords):
    hologram_set = []
    for sample in sample_list:
        hologram_set.append(cl.add_subdomain(reference, sample, coords))
    return hologram_set
