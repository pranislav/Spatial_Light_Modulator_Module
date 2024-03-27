'''creates phase mask for LC-SLM which compensates
aberrations caused both by the modulator and whole optical path.
This mask should be added up with any projected hologram.
For each optical there should be generated its own mask.

Principle:
modulator's screen is divided into square subdomains
for each subdomain we are searching for optimal phase offset
optimal phase offset is found as following:
let us have a reference subdomain which deflects light to particular angle
for each subdomain we do following:
the subdomain deflects light to the exact same angle as the reference one
all the others subdomains are "off"
the hologram on the subdomain is shifted by a constant phase several times
the phase shift which causes best constructive interference with the reference subdomain
is chosen and written into the phase mask
the quality of the interference is decided by
measuring intensity at the end of the optical path with a camera
'''

from calibration_lib import *
import numpy as np
import sys
import os
from pylablib.devices import uc480
from random import randint


def main(path_to_holograms: str):
    subdomain_size = get_subdomain_size(f"{path_to_holograms}/0/0/0.png")
    precision = get_precision(f"{path_to_holograms}/0/0")
    cam = uc480.UC480Camera()
    window = create_tk_window()
    H, W = get_number_of_subdomains(subdomain_size)
    path_to_random_hologram = f"{path_to_holograms}/{randint(H)}/{randint(W)}/{randint(precision)}"
    set_exposure_wrt_reference_img(cam, window, path_to_random_hologram)
    phase_step = 256 / precision
    phase_mask = np.zeros((H, W))
    for i in H:
        for j in W:
            intensity = 0
            for k in precision:
                display_image_on_external_screen(window, f"{path_to_holograms}/{i}/{j}/{k}.png") # displays hologram on an external dispaly (SLM)
                frame = cam.snap()
                if get_intensity_naive(frame) > intensity:
                    intensity = get_intensity_naive(frame)
                    if intensity == 255: print("maximal intensity was reached, consider adjusting exposure time")
                    phase_mask[i, j] = k * phase_step
    name = os.path.basename(path_to_holograms)
    create_phase_mask(phase_mask, subdomain_size, name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python calibration.py <path_to_holograms>")
        sys.exit(1)

# if __name__ == "__main__":
#     # Check if the number of arguments is correct
#     if len(sys.argv) < 2 or len(sys.argv) > 4:
#         print("Usage: python calibration.py <path_to_holograms> [subdomain_size] [precision]")
#         sys.exit(1)
    
    # Get command line arguments
    path_to_holograms = sys.argv[1]
    # subdomain_size = sys.argv[2] if len(sys.argv) >= 3 else 8
    # precision = sys.argv[3] if len(sys.argv) == 4 else 8
    
    # Call main function with the parameters
    main(path_to_holograms)

