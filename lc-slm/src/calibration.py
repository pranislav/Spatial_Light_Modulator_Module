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
import os
import sys
from pylablib.devices import uc480


def main(path_to_holograms: str, calibration_name: str):
    subdomain_size = get_subdomain_size(path_to_holograms)
    precision = get_precision(f"{path_to_holograms}/0/0")
    cam = uc480.UC480Camera()
    window = create_tk_window()
    H, W = get_number_of_subdomains(subdomain_size)
    path_to_reference_hologram = get_path_to_reference_hologram(path_to_holograms)
    set_exposure_wrt_reference_img(cam, window, path_to_reference_hologram)
    # coordinates = get_highest_intensity_coordinates(cam, window, path_to_reference_hologram)
    phase_step = 256 // precision
    phase_mask = np.zeros((H, W))
    i_0, j_0 = get_reference_position(path_to_reference_hologram)
    square = detect_bright_area(np.array(im.open(path_to_reference_hologram).convert("L")))
    for i in range(H):
        print(f"{i}/{H}")
        for j in range(W):
            if i == i_0 and j == j_0: continue
            intensity = 0
            for k in range(precision):
                display_image_on_external_screen(window, f"{path_to_holograms}/{i}/{j}/{k}.png") # displays hologram on an external dispaly (SLM)
                frame = cam.snap()
                if get_intensity_integral(frame, square) > intensity:
                    intensity = get_intensity_integral(frame, square)
                    # if intensity == 255: print("maximal intensity was reached, consider adjusting exposure time")
                    phase_mask[i, j] = k * phase_step
    name = f"{os.path.basename(path_to_holograms)}_{calibration_name}"
    create_phase_mask(phase_mask, subdomain_size, name)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python calibration.py <hologram_set_name> <calibration_name>")
        sys.exit(1)

    
    # Get command line arguments
    path_to_holograms = sys.argv[1]
    calibration_name = sys.argv[2]
# path_to_holograms = "lc-slm/holograms_for_calibration/size32_precision8_x1_y1"
# calibration_name = "home_trial"
    
    # Call main function with the parameters
    main(path_to_holograms, calibration_name)

