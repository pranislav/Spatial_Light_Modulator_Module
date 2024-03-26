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
import constants as c
import numpy as np
import sys
from PIL import Image as im
import subprocess
from pylablib.devices import uc480


def main(path_to_holograms: str, subdomain_size: int, precision: int):
    cam = uc480.UC480Camera()
    H, W = get_number_of_subdomains(subdomain_size)
    phase_step = 2 * np.pi() / precision
    phase_mask = np.zeros((H, W))
    for i in H:
        for j in W:
            intensity = 0
            for k in precision:
                subprocess.run(f"{path_to_holograms}/{i}/{j}/{k}.png") # displays hologram on an external dispaly (SLM)
                img = cam.snap()
                if get_intensity(img) > intensity:
                    intensity = get_intensity(img)
                    phase_mask[i, j] = k * phase_step
    create_phase_mask(phase_mask, path_to_holograms)



if __name__ == "__main__":
    # Check if the number of arguments is correct
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python calibration.py <path_to_holograms> [subdomain_size] [precision]")
        sys.exit(1)
    
    # Get command line arguments
    path_to_holograms = sys.argv[1]
    subdomain_size = sys.argv[2] if len(sys.argv) >= 3 else 8
    precision = sys.argv[3] if len(sys.argv) == 4 else 8
    
    # Call main function with the parameters
    main(path_to_holograms, subdomain_size, precision)

